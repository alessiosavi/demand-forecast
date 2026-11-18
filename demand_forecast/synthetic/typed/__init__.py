"""Equation-driven panels with typed inputs and recorded latent contributions."""

from demand_forecast.synthetic.typed.controls import (
    necessary_items,
    quiet_origins,
    shuffle_assignment,
)
from demand_forecast.synthetic.typed.generate import generate_typed_panel
from demand_forecast.synthetic.typed.recipe import (
    MECHANISMS,
    RANGES,
    SHARES,
    GeneratedPanel,
    InputRecord,
    PanelRecipe,
    sample_recipe,
)

__all__ = [
    "MECHANISMS",
    "RANGES",
    "SHARES",
    "GeneratedPanel",
    "InputRecord",
    "PanelRecipe",
    "generate_typed_panel",
    "necessary_items",
    "quiet_origins",
    "sample_recipe",
    "shuffle_assignment",
]
