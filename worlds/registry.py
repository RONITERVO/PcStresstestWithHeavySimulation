"""Stable registry for selectable Garage Life Lab worlds."""
from __future__ import annotations

from typing import Iterable, Optional

from . import (
    audio_reactive_3d,
    living_sketchbook_3d,
    minecraft_3d,
    minecraft_long_term_3d,
    muddy_asteroid_planet_3d,
    neural_plane_3d,
    original_2d,
    original_3d,
    original_tuned_3d,
    sketchbook_ink_islands_3d,
    sketchbook_visualizer_3d,
    static_sandstorm_3d,
    tsunami_land_3d,
)
from .spec import WorldSpec

DEFAULT_WORLD_ID = "minecraft-long-term-3d"

_WORLD_ORDER: tuple[WorldSpec, ...] = (
    minecraft_long_term_3d.SPEC,
    minecraft_3d.SPEC,
    living_sketchbook_3d.SPEC,
    sketchbook_visualizer_3d.SPEC,
    sketchbook_ink_islands_3d.SPEC,
    audio_reactive_3d.SPEC,
    static_sandstorm_3d.SPEC,
    tsunami_land_3d.SPEC,
    muddy_asteroid_planet_3d.SPEC,
    neural_plane_3d.SPEC,
    original_3d.SPEC,
    original_tuned_3d.SPEC,
    original_2d.SPEC,
)
_WORLDS_BY_ID = {world.id: world for world in _WORLD_ORDER}


def iter_worlds() -> Iterable[WorldSpec]:
    return _WORLD_ORDER


def world_ids() -> tuple[str, ...]:
    return tuple(world.id for world in _WORLD_ORDER)


def get_world(world_id: Optional[str]) -> WorldSpec:
    if not world_id:
        return _WORLDS_BY_ID[DEFAULT_WORLD_ID]
    try:
        return _WORLDS_BY_ID[world_id]
    except KeyError as exc:
        valid = ", ".join(world_ids())
        raise ValueError(f"Unknown world '{world_id}'. Valid worlds: {valid}") from exc
