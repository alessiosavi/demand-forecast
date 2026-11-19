"""Dataset-independent gated encoder/decoder and structural episode pruning."""

from dataclasses import dataclass, fields, replace

import torch
from torch import Tensor, nn
from torch.nn.functional import linear

from demand_forecast.data.episode import EpisodeBatch
from demand_forecast.data.episode.collate import EpisodeBatchWithNaive
from demand_forecast.data.episode.schema import (
    CH_STEP,
    CH_STEP_FLAG,
    KIND_NUMBER,
    N_SUMMARY,
    NAIVE_OPTIONS,
    PATCH,
    ROLE_FORECAST,
    ROLE_KNOWN,
    ROLE_PANEL,
    ROLE_PAST,
    ROLE_PEER,
    ROLE_SIBLING,
    STEP_FAMILIES,
)
from demand_forecast.models.heads import MonotoneQuantileHead

from .blocks import DecoderBlock, EncoderBlock, _expert_parameter_match, _rows
from .config import V5Config
from .context import ExpertRouter, ScaleShift, StaticContext
from .gates import GateNet, GlobalMask, HardConcreteGate
from .tokens import CalendarEmbedding, PatchEncoder


@dataclass
class Gates:
    """Applied independent gates and the one summary mask shared by this pass."""

    row: Tensor
    cand: Tensor
    family: Tensor
    summary_mask: Tensor
    expected_l0: Tensor


@dataclass
class V5Output:
    """Scaled quantiles, applied gates, per-row expert weights and penalties.

    ``router`` holds the expert weights applied in the forward pass, including
    the null weights under ``"uniform"`` and ``"rolled"`` routing. It is ``None``
    in dense expert mode.
    """

    quantiles: Tensor
    gates: Gates
    router: Tensor | None
    aux: dict[str, Tensor]
    anchor: Tensor | None = None
    mix: Tensor | None = None


def _reference(batch: EpisodeBatch, summary: Tensor, flags: Tensor) -> tuple[Tensor, Tensor]:
    kept = batch.keep & batch.forecast_mask
    values = _rows(summary, batch.forecast_rows)
    valid = _rows(flags, batch.forecast_rows)
    reference = values.masked_fill(~kept[..., None], 0).sum(1)
    reference = reference / kept.sum(1).clamp_min(1)[..., None]
    return reference, (valid & kept[..., None]).any(1)


def _candidate_weights(gates: Tensor, indices: Tensor, mask: Tensor) -> Tensor:
    if gates.shape[1] == 0:
        return gates.new_zeros(indices.shape)
    return gates.gather(1, indices.clamp_min(0)).masked_fill(~mask, 0)


class V5Model(nn.Module):
    """Causal shared memory with forecast and known-future rows at every origin."""

    def __init__(self, config: V5Config = V5Config()) -> None:
        super().__init__()
        self.config = config
        self.global_mask = GlobalMask(config)
        self.gate_net = GateNet(config)
        self.family_logits = nn.Linear(2 * N_SUMMARY, len(STEP_FAMILIES))
        nn.init.zeros_(self.family_logits.weight)
        nn.init.constant_(self.family_logits.bias, config.gate_init)
        self.family_gate = HardConcreteGate()
        self.static_context = StaticContext(config)
        self.calendar = CalendarEmbedding(config)
        self.patch_encoder = PatchEncoder(config)
        self.scale_shift = ScaleShift(config)
        self.router = ExpertRouter(config) if config.expert_mode == "routed" else None
        self.encoder = nn.ModuleList(
            EncoderBlock(config, expert=i >= config.enc_layers - config.expert_enc_layers)
            for i in range(config.enc_layers)
        )
        self.decoder = nn.ModuleList(
            DecoderBlock(config, expert=i >= config.dec_layers - config.expert_dec_layers)
            for i in range(config.dec_layers)
        )
        self.query = nn.Parameter(torch.empty(config.d_model))
        nn.init.normal_(self.query, std=0.02)
        self.horizon_calendar = nn.Sequential(
            nn.Linear(PATCH * config.d_calendar, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
        )
        self.step_projection = nn.Linear(config.d_model, PATCH * config.d_out)
        self.head = MonotoneQuantileHead(config.d_out, config.quantiles, config.width_factor)
        if config.horizon_seed:
            self.horizon_seed_projection = nn.Linear(config.d_model, config.d_model)
            nn.init.zeros_(self.horizon_seed_projection.weight)
            nn.init.zeros_(self.horizon_seed_projection.bias)
        if config.naive_mix:
            self.mix_projection = nn.Linear(config.d_out, len(NAIVE_OPTIONS))
            nn.init.zeros_(self.mix_projection.weight)
            with torch.no_grad():
                self.mix_projection.bias.copy_(
                    self.mix_projection.bias.new_tensor((-3, 0, 0, 4.5, 0, -2))
                )

    def _mix_naive(
        self, steps_out: Tensor, batch: EpisodeBatchWithNaive, real: Tensor
    ) -> tuple[Tensor, Tensor]:
        projection = self.mix_projection
        dtype = torch.float64 if projection.weight.dtype == torch.float64 else torch.float32
        logits = linear(steps_out.to(dtype), projection.weight.to(dtype), projection.bias.to(dtype))
        seen = batch.naive_seen & real[..., None]
        has_option = seen.any(dim=-1, keepdim=True)
        # Keep empty logits finite (test_empty_naive_cell_has_finite_softmax_backward).
        logits = logits.masked_fill(~seen, -torch.inf).masked_fill(~has_option, 0)
        mix = logits.softmax(dim=-1).masked_fill(~seen, 0)
        paths = batch.naive_paths.to(dtype).masked_fill(~seen, 0)
        return (mix * paths).sum(-1), mix

    def parameter_report(self) -> dict[str, int | None]:
        """Return both parameter totals and the extra width of each dense expert block.

        Dense size and extra width are unavailable when there are no expert blocks.
        """
        total = sum(parameter.numel() for parameter in self.parameters())
        config = self.config
        if config.expert_enc_layers == config.expert_dec_layers == 0:
            return {"routed": total, "dense": None, "extra": None}
        extra, delta = _expert_parameter_match(config)
        routed = total if config.expert_mode == "routed" else total - delta
        return {"routed": routed, "dense": routed + delta, "extra": extra}

    def gates(
        self,
        batch: EpisodeBatch,
        *,
        mode: str = "learned",
        zero_candidates: Tensor | None = None,
        zero_rows: Tensor | None = None,
    ) -> Gates:
        """Choose cutoff-only gates, then force zeros without changing expected L0.

        The ``"evidence"`` mode uses the sign of builder relevance scores and
        keeps forecast, known-future and whole-panel rows open.

        ``zero_candidates`` zeros candidate gates and their inherited peer gates.
        ``zero_rows`` may force any real row except ``ROLE_FORECAST``, including
        siblings and known-future rows (overriding their gate floor).
        A forced sibling behaves like a sibling with a learned zero gate: its
        forecast slot remains in the output until ``prune`` removes it. Callers
        scoring forecasts must select kept rows (``keep & forecast_mask``).
        """
        if mode not in ("learned", "ones", "evidence"):
            raise ValueError("gate mode must be 'learned', 'ones' or 'evidence'")
        for name, mask, shape in (
            ("zero_candidates", zero_candidates, batch.cand_mask.shape),
            ("zero_rows", zero_rows, batch.row_mask.shape),
        ):
            if mask is not None and (mask.dtype != torch.bool or mask.shape != shape):
                raise ValueError(f"{name} must be a boolean mask with shape {tuple(shape)}")
        if (
            zero_rows is not None
            and (zero_rows & batch.row_mask & (batch.row_role == ROLE_FORECAST)).any()
        ):
            raise ValueError("cannot force forecast rows to zero")
        if mode != "learned":
            if mode == "evidence":
                role = batch.row_role
                always_open = (role == ROLE_FORECAST) | (role == ROLE_KNOWN) | (role == ROLE_PANEL)
                row = ((batch.row_score > 0) | always_open).to(batch.row_score)
                cand = (batch.cand_score > 0).to(batch.cand_score)
            else:
                row = torch.ones_like(batch.row_score)
                cand = torch.ones_like(batch.cand_score)
            family = batch.x.new_ones((batch.x.shape[0], len(STEP_FAMILIES)))
            summary_mask = batch.x.new_ones(N_SUMMARY)
            expected_l0 = batch.x.new_zeros(())
        else:
            # Evaluate GlobalMask once. Forward consumes this exact sample as well.
            summary_mask, _ = self.global_mask(
                batch.x.new_ones(N_SUMMARY),
                torch.ones(N_SUMMARY, dtype=torch.bool, device=batch.x.device),
            )
            summary = batch.row_summary * summary_mask
            flags = batch.row_summary_flags & (summary_mask > 0)
            ref, ref_flags = _reference(batch, summary, flags)
            row, row_l0 = self.gate_net(
                summary, flags, batch.row_score, batch.row_kind, batch.row_role, ref, ref_flags
            )
            cand, cand_l0 = self.gate_net(
                batch.cand_summary * summary_mask,
                batch.cand_flags & (summary_mask > 0),
                batch.cand_score,
                batch.cand_kind,
                torch.full_like(batch.cand_kind, ROLE_PEER),
                ref,
                ref_flags,
            )
            logits = self.family_logits(torch.cat((ref, ref_flags.to(ref.dtype)), dim=-1))
            family = self.family_gate(logits)
            independent = batch.row_mask & (
                (batch.row_role == ROLE_SIBLING)
                | (batch.row_role == ROLE_PAST)
                | (batch.row_role == ROLE_PANEL)
            )
            family_l0 = self.family_gate.expected_l0(logits)
            count = independent.sum() + batch.cand_mask.sum() + family_l0.numel()
            expected_l0 = (
                row_l0.masked_fill(~independent, 0).sum()
                + cand_l0.masked_fill(~batch.cand_mask, 0).sum()
                + family_l0.sum()
            ) / count.clamp_min(1)
            expected_l0 = expected_l0 + self.global_mask.expected_l0().mean()

        if zero_candidates is not None:
            cand = cand.masked_fill(zero_candidates, 0)
        peer_gate = _candidate_weights(cand, batch.row_cand, batch.row_mask)
        row = torch.where(batch.row_role == ROLE_PEER, peer_gate, row)
        if mode != "ones":
            row = row.masked_fill(~batch.row_mask, 0)
            cand = cand.masked_fill(~batch.cand_mask, 0)
        if zero_rows is not None:
            row = row.masked_fill(zero_rows, 0)
        return Gates(row, cand, family, summary_mask, expected_l0)

    def forward(
        self, batch: EpisodeBatch, gates: Gates | None = None, *, router: str = "learned"
    ) -> V5Output:
        """Encode and decode with learned, uniform or deterministically rolled routing.

        Rolled routing requires at least two episodes in the batch. Dense expert
        mode accepts the default ``router="learned"`` argument.
        """
        if self.router is None and router != "learned":
            raise ValueError("dense expert mode only accepts router='learned'")
        if router not in ("learned", "uniform", "rolled"):
            raise ValueError("router must be 'learned', 'uniform' or 'rolled'")
        if router == "rolled" and batch.x.shape[0] == 1:
            raise ValueError("the rolled router needs at least two episodes")
        gates = self.gates(batch) if gates is None else gates
        summary = batch.row_summary * gates.summary_mask
        flags = batch.row_summary_flags & (gates.summary_mask > 0)
        ref, ref_flags = _reference(batch, summary, flags)
        x = batch.x.clone()
        numeric = batch.row_kind == KIND_NUMBER
        for fi, span in enumerate(STEP_FAMILIES.values()):
            weight = torch.where(numeric, gates.family[:, fi, None], 1)[:, :, None, None, None]
            for offset in (CH_STEP, CH_STEP_FLAG):
                x[..., offset + span.start : offset + span.stop] = (
                    batch.x[..., offset + span.start : offset + span.stop] * weight
                )
        # Both halves of a value token are summaries (member mean and spread).
        attr_stats = torch.cat(
            (
                batch.attr_stats[..., : 2 * N_SUMMARY] * gates.summary_mask.repeat(2),
                batch.attr_stats[..., 2 * N_SUMMARY :],
            ),
            dim=-1,
        )
        attr_weight = _candidate_weights(gates.cand, batch.attr_cand, batch.attr_mask)
        attr_weight = attr_weight * batch.attr_flags
        number_weight = _candidate_weights(gates.cand, batch.static_num_cand, batch.static_num_mask)
        number_weight = number_weight * batch.static_num_flags
        context = self.static_context(
            attr_stats,
            attr_weight,
            batch.static_num,
            batch.static_num_flags,
            number_weight,
            ref,
            ref_flags,
            batch.freq_class,
        )
        if self.router is None:
            weights, balance = None, context.new_zeros(())
        else:
            weights, balance = self.router(context, summary, flags, batch.row_mask)
            if router == "uniform":
                weights = torch.full_like(weights, 1 / self.config.n_experts)
            elif router == "rolled":
                weights = weights.roll(max(1, weights.shape[0] // 2), dims=0)
        cal = self.calendar(batch.cal_phases, batch.cal_ids)[:, None]
        tokens = self.patch_encoder(
            x,
            cal.expand(*x.shape[:-1], self.config.d_calendar),
            batch.row_kind[..., None].expand(x.shape[:3]),
            batch.row_role[..., None].expand(x.shape[:3]),
        )
        tokens = self.scale_shift(tokens * gates.row[..., None, None], context)
        for block in self.encoder:
            tokens = block(tokens, batch.patch_valid, batch.row_mask, gates.row, weights)

        steps = torch.arange(batch.calh_phases.shape[2] * PATCH, device=x.device)
        step_mask = (steps[None] < batch.horizon[:, None]).reshape(
            x.shape[0], -1, PATCH
        ) & batch.horizon_mask[..., None]
        calh = self.calendar(batch.calh_phases, batch.calh_ids)
        calh = calh.masked_fill(~step_mask[:, None, ..., None], 0)
        query = self.query + self.horizon_calendar(calh.flatten(-2))
        nf = batch.forecast_rows.shape[1]
        forecast_tokens = query[:, :, None].expand(-1, -1, nf, -1, -1)
        if self.config.horizon_seed:
            last_patch = batch.origin_patches - 1
            examples = torch.arange(tokens.shape[0], device=tokens.device)[:, None, None]
            rows = batch.forecast_rows[:, None, :]
            patches = last_patch.clamp_min(0)[..., None]
            history = tokens[examples, rows, patches]
            valid = batch.patch_valid[examples, rows, patches]
            valid = (
                valid
                & ((last_patch >= 0) & batch.origin_mask)[..., None]
                & batch.forecast_mask[:, None, :]
            )
            seed = self.horizon_seed_projection(history.masked_fill(~valid[..., None], 0))
            seed = seed.masked_fill(~valid[..., None], 0)
            forecast_tokens = forecast_tokens + seed.unsqueeze(-2)
        xh = batch.xh.masked_fill(~step_mask[:, None, None, ..., None], 0)
        kinds = _rows(batch.row_kind, batch.known_rows)[:, None, :, None]
        roles = _rows(batch.row_role, batch.known_rows)[:, None, :, None]
        known_tokens = self.patch_encoder(
            xh,
            calh[:, :, None].expand(*xh.shape[:-1], self.config.d_calendar),
            kinds.expand(xh.shape[:4]),
            roles.expand(xh.shape[:4]),
        )
        dec_rows = torch.cat((batch.forecast_rows, batch.known_rows), dim=1)
        dec_mask = torch.cat((batch.forecast_mask, batch.known_mask), dim=1)
        dec_gate = _rows(gates.row, dec_rows)
        horizon_tokens = torch.cat((forecast_tokens, known_tokens), dim=2)
        h = self.scale_shift(horizon_tokens * dec_gate[:, None, :, None, None], context)
        dec_weights = _rows(weights, dec_rows) if weights is not None else None
        for block in self.decoder:
            h = block(
                h,
                tokens,
                dec_rows,
                batch.origin_patches,
                batch.patch_valid,
                batch.horizon_mask,
                dec_mask,
                dec_gate,
                dec_weights,
            )
        real = batch.origin_mask[:, :, None, None] & batch.forecast_mask[:, None, :, None]
        anchor = mix = None
        # Low precision can round the gaps between adjacent quantiles to zero.
        with torch.autocast(h.device.type, enabled=False):
            # Match the float32 weights in training and honor an explicit forecast dtype.
            states = h[:, :, :nf].to(self.step_projection.weight.dtype)
            steps_out = self.step_projection(states).unflatten(-1, (PATCH, self.config.d_out))
            steps_out = steps_out.flatten(3, 4)
            quantiles = self.head(steps_out)
            if self.config.naive_mix and isinstance(batch, EpisodeBatchWithNaive):
                real_steps = real & step_mask.flatten(1)[:, None, None, :]
                anchor, mix = self._mix_naive(steps_out, batch, real_steps)
                quantiles = torch.where(
                    batch.naive_seen.any(dim=-1, keepdim=True) & real_steps[..., None],
                    quantiles + anchor[..., None],
                    quantiles,
                )
        quantiles = quantiles.masked_fill(~real[..., None], 0)
        return V5Output(
            quantiles,
            gates,
            weights,
            {"sparsity": gates.expected_l0, "balance": balance},
            anchor,
            mix,
        )


def prune(batch: EpisodeBatch, gates: Gates) -> tuple[EpisodeBatch, Gates]:
    """Remove zero-gated candidates and rows, repadding and remapping every index."""
    row_fields = {
        "x",
        "patch_valid",
        "row_kind",
        "row_role",
        "row_summary",
        "row_summary_flags",
        "row_score",
        "row_cand",
        "row_mask",
    }
    forecast_fields = {"forecast_rows", "keep", "loc", "scale", "fallback", "forecast_mask"}
    known_fields = {"known_rows", "known_mask"}
    cand_fields = {"cand_kind", "cand_summary", "cand_flags", "cand_score", "cand_mask"}
    attr_fields = {"attr_stats", "attr_flags", "attr_cand", "attr_mask"}
    number_fields = {"static_num", "static_num_flags", "static_num_cand", "static_num_mask"}
    examples, row_gates, cand_gates = [], [], []
    for b in range(batch.x.shape[0]):
        candidates = batch.cand_mask[b] & (gates.cand[b] != 0)
        rows = batch.row_mask[b] & (gates.row[b] != 0)
        if candidates.numel():
            peer_kept = candidates[batch.row_cand[b].clamp_min(0)]
            rows = rows & ((batch.row_role[b] != ROLE_PEER) | peer_kept)
        row_indices = rows.nonzero().flatten()
        cand_indices = candidates.nonzero().flatten()
        row_map = torch.full_like(batch.row_role[b], -1)
        row_map[row_indices] = torch.arange(len(row_indices), device=rows.device)
        cand_map = torch.full_like(batch.cand_kind[b], -1)
        cand_map[cand_indices] = torch.arange(len(cand_indices), device=rows.device)
        forecast = batch.forecast_mask[b] & rows[batch.forecast_rows[b]]
        known = batch.known_mask[b] & rows[batch.known_rows[b]]
        attrs = batch.attr_mask[b]
        numbers = batch.static_num_mask[b]
        if candidates.numel():
            attrs = attrs & candidates[batch.attr_cand[b]]
            numbers = numbers & candidates[batch.static_num_cand[b]]
        masks = (rows, forecast, known, candidates, attrs, numbers)
        groups = (
            row_fields,
            forecast_fields,
            known_fields,
            cand_fields,
            attr_fields,
            number_fields,
        )
        example = {}
        for field in fields(batch):
            name, value = field.name, getattr(batch, field.name)
            if not isinstance(value, Tensor):
                example[name] = value[b]
                continue
            value = value[b]
            for names, selected in zip(groups, masks):
                if name in names:
                    value = value[selected]
                    break
            if name in ("labels", "label_mask", "naive_paths", "naive_seen"):
                value = value[:, forecast]
            if name == "xh":
                value = value[:, known]
            if name in ("forecast_rows", "known_rows"):
                value = row_map[value]
            if name in ("row_cand", "attr_cand", "static_num_cand"):
                value = (
                    torch.where(value >= 0, cand_map[value.clamp_min(0)], -1)
                    if candidates.numel()
                    else value
                )
            example[name] = value
        metadata_masks = {"row_names": rows, "cand_names": candidates, "attr_names": attrs}
        if "row_keys" in example:
            metadata_masks["row_keys"] = rows
        for name, selected in metadata_masks.items():
            example[name] = [
                value for i, value in enumerate(getattr(batch, name)[b]) if selected[i]
            ]
        examples.append(example)
        row_gates.append(gates.row[b, rows])
        cand_gates.append(gates.cand[b, candidates])

    def padded(values: list[Tensor]) -> Tensor:
        shape = tuple(max(value.shape[i] for value in values) for i in range(values[0].ndim))
        result = values[0].new_zeros((len(values), *shape))
        for i, value in enumerate(values):
            result[(i, *(slice(0, n) for n in value.shape))] = value
        return result

    result = {
        name: padded([example[name] for example in examples])
        if isinstance(getattr(batch, name), Tensor)
        else [example[name] for example in examples]
        for name in examples[0]
    }
    return replace(batch, **result), Gates(
        padded(row_gates), padded(cand_gates), gates.family, gates.summary_mask, gates.expected_l0
    )
