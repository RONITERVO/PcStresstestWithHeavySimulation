"""World metadata contracts for Garage Life Lab."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Optional, Sequence

import numpy as np

SeedFieldFactory = Callable[[int, int, int], np.ndarray]


@dataclass(frozen=True)
class WorldSpec:
    id: str
    display_name: str
    window_title: str
    sim_shader: str
    display_shader: str
    seed_field: SeedFieldFactory
    default_overrides: Mapping[str, object] = field(default_factory=dict)
    preview_image: Optional[str] = None
    stability_notes: Sequence[str] = field(default_factory=tuple)
    hud_subtitle: str = "3D VOLUMETRIC STRESS"
    preview_palette: Sequence[str] = field(default_factory=tuple)
    uses_audio: bool = False
