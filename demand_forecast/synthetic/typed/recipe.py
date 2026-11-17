"""Recipes and the public records of an equation-driven typed panel."""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from demand_forecast.data.episode import Panel
from demand_forecast.data.frequency import to_offset

MECHANISMS = (
    "multi_target",
    "calendar",
    "seasonal_lag",
    "intermittent",
    "level_shift",
    "missing_vs_zero",
    "static_category",
    "multi_label",
    "hierarchy",
    "static_number",
    "whole_number_code",
    "dynamic_category_mean",
    "dynamic_category_spread",
    "dynamic_category_timing",
    "known_driver",
    "planned_event",
    "delayed_past_driver",
    "event_response",
    "label_lag",
)

RANGES: dict[str, dict[str, tuple[tuple[float, float], tuple[float, float]]]] = {
    "multi_target": {"lag": ((0, 3), (4, 6))},
    "calendar": {"amplitude": ((0.1, 0.4), (0.5, 0.8))},
    "seasonal_lag": {"amplitude": ((0.5, 1.5), (1.7, 2.3))},
    "intermittent": {"p": ((0.15, 0.7), (0.05, 0.12))},
    "level_shift": {"size": ((0.15, 0.4), (0.5, 0.8))},
    "missing_vs_zero": {"rebound": ((0.5, 1.5), (1.7, 2.3))},
    "static_category": {"amplitude": ((0.2, 0.6), (0.7, 1.0))},
    "multi_label": {"size": ((0.2, 0.5), (0.6, 0.9))},
    "hierarchy": {"amplitude": ((0.15, 0.4), (0.5, 0.8))},
    "static_number": {"noise": ((0.3, 0.7), (0.8, 1.1)), "trend": ((0.3, 0.8), (0.9, 1.3))},
    "whole_number_code": {"p": ((0.3, 0.9), (0.1, 0.25))},
    "dynamic_category_mean": {"effect": ((0.3, 0.6), (0.7, 1.0))},
    "dynamic_category_spread": {"effect": ((0.4, 1.0), (1.2, 1.6))},
    "dynamic_category_timing": {"pulse": ((0.5, 1.2), (1.4, 2.0))},
    "known_driver": {"beta": ((0.3, 0.8), (0.9, 1.3))},
    "planned_event": {"size": ((0.4, 1.2), (1.4, 2.0))},
    "delayed_past_driver": {"beta": ((0.4, 0.9), (1.0, 1.4)), "lag_periods": ((0, 1), (1, 2))},
    "event_response": {"size": ((0.3, 1.0), (1.2, 1.8))},
    "label_lag": {"beta": ((0.4, 0.9), (1.0, 1.4))},
}

SHARES: dict[str, float] = {
    "static_category": 0.50,
    "multi_label": 0.30,
    "hierarchy": 0.25,
    "static_number": 0.30,
    "whole_number_code": 0.20,
    "dynamic_category": 0.40,
    "known_driver": 0.40,
    "planned_event": 0.40,
    "delayed_past_driver": 0.35,
    "event_response": 0.20,
    "label_lag": 0.20,
    "calendar": 0.50,
    "multi_target": 0.25,
    "missing_vs_zero": 0.20,
    "seasonal_lag": 0.30,
    "intermittent": 0.25,
    "level_shift": 0.20,
}


@dataclass(frozen=True)
class PanelRecipe:
    """The mechanisms, grid distribution and seed of one panel."""

    mechanisms: tuple[str, ...] = ()
    n_items: int = 64
    length: tuple[int, int] = (64, 2048)
    freq: str = "D"
    n_nuisance: int = 0
    nuisance_from: tuple[str, ...] | None = None
    singleton_share: float = 0.0
    heldout: bool = False
    complete_inputs: bool = False
    seed: int = 0

    def __post_init__(self) -> None:
        if any(m not in MECHANISMS for m in self.mechanisms):
            raise ValueError("Unknown mechanism in recipe")
        if len(set(self.mechanisms)) != len(self.mechanisms):
            raise ValueError("Repeated mechanism in recipe")
        if sum(m.startswith("dynamic_category_") for m in self.mechanisms) > 1:
            raise ValueError("At most one dynamic_category_* mechanism is allowed")
        if self.n_items < 1:
            raise ValueError("n_items must be at least one")
        if self.length[0] < 8 or self.length[0] > self.length[1]:
            raise ValueError("length must be ordered with a minimum of eight")
        if not 0 <= self.singleton_share <= 1:
            raise ValueError("singleton_share must lie in [0, 1]")
        if self.nuisance_from is not None and any(
            m not in self.mechanisms for m in self.nuisance_from
        ):
            raise ValueError("nuisance_from must name mechanisms in the recipe")
        offset = to_offset(self.freq)
        if offset.n <= 0:
            raise ValueError("freq must have a positive step")
        panel = Panel(
            "recipe", (), {}, self.freq, offset.rollforward(pd.Timestamp("2000-01-01")), True
        )
        if "seasonal_lag" in self.mechanisms and panel.long_lag is None:
            raise ValueError("seasonal_lag requires a frequency with a long lag")


@dataclass(frozen=True)
class InputRecord:
    """An input's location, kind, role and optional nuisance provenance."""

    name: str
    mechanism: str
    where: str
    kind: str
    role: str | None
    twin: str | None = None
    process: dict[str, object] | None = None


@dataclass(frozen=True)
class GeneratedPanel:
    """Observed and complete panels together with their generating equations."""

    panel: Panel
    complete: Panel
    recipe: PanelRecipe
    inputs: tuple[InputRecord, ...]
    params: dict[str, dict[str, object]]
    parts: dict[str, dict[str, np.ndarray]]
    scalars: dict[str, dict[str, np.ndarray]]
    short_items: tuple[str, ...]
    hosts: dict[str, str]
    availability: dict[str, float]
    grid_length: int
    period: int
    long_lag: int | None
    shuffled: tuple[str, ...] = ()
    donors: dict[str, dict[str, tuple[str, int]]] = field(default_factory=dict)

    def inputs_of(self, mechanism: str) -> tuple[InputRecord, ...]:
        return tuple(record for record in self.inputs if record.mechanism == mechanism)


def sample_recipe(seed: int, *, heldout: bool = False) -> PanelRecipe:
    """Sample independent mechanism presences and the proposed grid mixture."""
    from demand_forecast.synthetic.typed.draft import stream

    rng = stream(seed, "recipe")
    empty = rng.random() < 0.1
    freq = str(rng.choice(("h", "D", "W-MON", "ME"), p=(0.25, 0.40, 0.20, 0.15)))
    chosen = set()
    if not empty:
        for name in sorted(SHARES):
            if rng.random() < SHARES[name]:
                if name == "dynamic_category":
                    name = str(
                        rng.choice(
                            (
                                "dynamic_category_mean",
                                "dynamic_category_spread",
                                "dynamic_category_timing",
                            )
                        )
                    )
                chosen.add(name)
    probe = Panel("recipe", (), {}, freq, None, True)
    if probe.long_lag is None:
        chosen.discard("seasonal_lag")
    return PanelRecipe(
        mechanisms=tuple(m for m in MECHANISMS if m in chosen),
        n_items=int(rng.integers(8, 97)),
        length={"ME": (24, 240), "W-MON": (52, 1040)}.get(freq, (64, 2048)),
        freq=freq,
        n_nuisance=int(rng.integers(0, 4)),
        singleton_share=0.0 if rng.random() < 0.7 else float(rng.uniform(0.1, 0.5)),
        heldout=heldout,
        seed=seed,
    )
