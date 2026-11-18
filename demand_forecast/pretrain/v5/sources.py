"""Typed panels and corpus rows through one random-generator-driven interface."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Protocol

import numpy as np

from demand_forecast.data.episode import Item, Panel
from demand_forecast.synthetic.typed import (
    GeneratedPanel,
    PanelRecipe,
    generate_typed_panel,
    sample_recipe,
    shuffle_assignment,
)

from .config import SourceConfig
from .corpus import CorpusIndex, RowCache, read_values


class PanelSource(Protocol):
    def panel(self, rng: np.random.Generator) -> tuple[Panel, dict]: ...


def apply_control(generated: GeneratedPanel, config: SourceConfig) -> Panel:
    """Return the exposed panel, with input controls addressed by the recipe seed.

    Shuffle includes nuisance twins and moves each hierarchy pair together once.
    ``None`` and ``missing_as_zero`` leave the exposed panel unchanged; the
    sampler fills missing target history after choosing the episode's anchor.
    No draws are taken from the source's caller-owned random generator.
    """
    if config.control == "shuffle":
        controlled = generated
        for record in generated.inputs:
            mechanism = record.twin if record.mechanism == "nuisance" else record.mechanism
            if mechanism not in generated.recipe.mechanisms or record.name in controlled.shuffled:
                continue
            if config.control_inputs == "attribute" and record.where != "static":
                continue
            if config.control_inputs == "column" and record.where != "dynamic":
                continue
            controlled = shuffle_assignment(controlled, record.name, seed=generated.recipe.seed)
        return controlled.panel
    if config.control == "no_dates":
        return replace(generated.panel, epoch=None)
    return generated.panel


class TypedSource:
    """Generate typed panels with optional input controls.

    ``shuffle`` requires ``max_items >= 2``; sampled recipes are clamped to
    between two items and that cap without drawing from the caller's generator.
    Both hierarchy attributes move together in one ``shuffle_assignment`` call;
    each hierarchy nuisance twin pair likewise moves together exactly once.
    This applies to both ``control_inputs='all'`` and ``'attribute'``.
    """

    def __init__(self, config: SourceConfig):
        if config.control == "shuffle" and config.max_items < 2:
            raise ValueError("control='shuffle' requires max_items >= 2")
        self.config = config

    def panel(self, rng: np.random.Generator) -> tuple[Panel, dict]:
        config = self.config
        seed = int(rng.integers(0, 2**63))
        if config.recipes:
            mechanisms = config.recipes[int(rng.integers(len(config.recipes)))]
            recipe = PanelRecipe(
                mechanisms=mechanisms,
                n_items=config.max_items,
                length=config.length,
                freq=config.freq or "D",
                n_nuisance=config.n_nuisance,
                nuisance_from=config.nuisance_from,
                singleton_share=config.singleton_share,
                heldout=config.heldout,
                complete_inputs=False,
                seed=seed,
            )
        else:
            recipe = sample_recipe(seed, heldout=config.heldout)
            n_items = min(recipe.n_items, config.max_items)
            if config.control == "shuffle":
                n_items = max(2, n_items)
            recipe = replace(recipe, n_items=n_items)
        generated = generate_typed_panel(recipe)
        return apply_control(generated, config), {
            "source": "typed",
            "generated": generated,
            "recipe": recipe,
            "control": config.control,
            "index": {item.item_id: i for i, item in enumerate(generated.panel.items)},
        }


class CorpusSource:
    def __init__(self, config: SourceConfig, root: Path | None = None):
        location = config.root if config.root is not None else root
        if location is None:
            raise ValueError("corpus source requires a root")
        self.config = config
        self.index = CorpusIndex.build(Path(location))
        if not self.index.rows:
            raise ValueError(f"corpus has no admitted rows: {location}")
        self.cache = RowCache()

    def panel(self, rng: np.random.Generator) -> tuple[Panel, dict]:
        values = read_values(self.index, int(rng.integers(self.index.rows)), self.cache)
        targets, covariates = values["targets"], values["covariates"]
        designated = int(rng.integers(len(targets)))
        others = np.delete(np.arange(len(targets)), designated)
        selected = [designated, *rng.choice(others, min(15, len(others)), replace=False)]
        columns = rng.choice(len(covariates), min(32, len(covariates)), replace=False)
        roles = {f"cov_{i:03d}": str(rng.choice(("known", "past"))) for i in columns}
        item = Item(
            item_id=f"{values['file']}:{values['row']}",
            start=0,
            targets={f"target_{i:03d}": targets[i] for i in selected},
            dynamic={f"cov_{i:03d}": covariates[i] for i in columns},
            static={},
        )
        panel = Panel("corpus", (item,), roles, values["freq"], None, False)
        return panel, {
            "source": "corpus",
            "file": values["file"],
            "row": values["row"],
            "target": f"target_{designated:03d}",
        }


class SourceMix:
    def __init__(self, sources, *, corpus_root: Path | None = None):
        configs = tuple(sources)
        weights = np.array([source.weight for source in configs], dtype=np.float64)
        if (
            not len(weights)
            or not np.all(np.isfinite(weights))
            or np.any(weights < 0)
            or not weights.sum() > 0
        ):
            raise ValueError("source weights must be finite, nonnegative and have a positive sum")
        self._weights = weights / weights.sum()
        self.sources: tuple[PanelSource, ...] = tuple(
            self._source(config, corpus_root) for config in configs
        )

    @staticmethod
    def _source(config: SourceConfig, corpus_root: Path | None) -> PanelSource:
        if config.kind == "typed":
            return TypedSource(config)
        if config.kind == "corpus":
            return CorpusSource(config, root=corpus_root)
        raise NotImplementedError("upstream source is reserved for Task P8")

    @property
    def corpus_digest(self) -> str | None:
        digests = [
            source.index.digest for source in self.sources if isinstance(source, CorpusSource)
        ]
        if not digests:
            return None
        if len(set(digests)) == 1:
            return digests[0]
        return listing_digest_for_sources(digests)

    def panel(self, rng: np.random.Generator) -> tuple[Panel, dict]:
        return self.sources[int(rng.choice(len(self.sources), p=self._weights))].panel(rng)


def listing_digest_for_sources(digests: list[str]) -> str:
    from .corpus import listing_digest

    return listing_digest((str(i), 0, digest) for i, digest in enumerate(digests))
