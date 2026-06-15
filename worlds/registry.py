"""Stable registry for selectable Garage Life Lab worlds.

The project now has one intentional long-term default world.  Older experiments
remain available through explicit selection and through the legacy helpers below
so launcher UI can keep them in a separate tab without losing compatibility.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Iterable, Optional

from . import (
    audio_reactive_3d,
    Final_minecraft_3D,
    future_future_minecraft,
    future_minecraft,
    living_sketchbook_3d,
    minecraft_3d,
    minecraft_definitive_overworld_3d,
    minecraft_first_future,
    minecraft_fluid,
    minecraft_long_term_3d,
    minecraft_overworld_prime_3d,
    minecraft_perfect_ecosystem_3d,
    minecraft_voxels_3d,
    minecraft_voxelv2_3d,
    muddy_asteroid_planet_3d,
    neural_plane_3d,
    original_2d,
    original_3d,
    original_tuned_3d,
    sketchbook_ink_islands_3d,
    sketchbook_visualizer_3d,
    static_sandstorm_3d,
    true_minecraft,
    tsunami_land_3d,
)
from .spec import WorldSpec

DEFAULT_WORLD_ID: str = minecraft_definitive_overworld_3d.SPEC.id

_PRIMARY_WORLDS: tuple[WorldSpec, ...] = (
    minecraft_definitive_overworld_3d.SPEC,
)

_LEGACY_SOURCE_WORLDS: tuple[WorldSpec, ...] = (
    audio_reactive_3d.SPEC,
    Final_minecraft_3D.SPEC,
    future_future_minecraft.SPEC,
    future_minecraft.SPEC,
    living_sketchbook_3d.SPEC,
    minecraft_3d.SPEC,
    minecraft_first_future.SPEC,
    minecraft_fluid.SPEC,
    minecraft_long_term_3d.SPEC,
    minecraft_overworld_prime_3d.SPEC,
    minecraft_perfect_ecosystem_3d.SPEC,
    minecraft_voxels_3d.SPEC,
    minecraft_voxelv2_3d.SPEC,
    muddy_asteroid_planet_3d.SPEC,
    neural_plane_3d.SPEC,
    original_2d.SPEC,
    original_3d.SPEC,
    original_tuned_3d.SPEC,
    sketchbook_ink_islands_3d.SPEC,
    sketchbook_visualizer_3d.SPEC,
    static_sandstorm_3d.SPEC,
    true_minecraft.SPEC,
    tsunami_land_3d.SPEC,
)


def _as_legacy(world: WorldSpec) -> WorldSpec:
    notes = tuple(dict.fromkeys((*world.stability_notes, "legacy")))
    return replace(world, stability_notes=notes)


_LEGACY_WORLDS: tuple[WorldSpec, ...] = tuple(
    sorted((_as_legacy(world) for world in _LEGACY_SOURCE_WORLDS), key=lambda world: world.id)
)
_WORLD_ORDER: tuple[WorldSpec, ...] = (*_PRIMARY_WORLDS, *_LEGACY_WORLDS)
_WORLDS_BY_ID = {world.id: world for world in _WORLD_ORDER}
_LEGACY_WORLD_IDS = frozenset(world.id for world in _LEGACY_WORLDS)


def iter_worlds(include_legacy: bool = True) -> Iterable[WorldSpec]:
    """Return worlds in launcher order: primary default first, legacy after."""
    if include_legacy:
        return _WORLD_ORDER
    return _PRIMARY_WORLDS


def iter_primary_worlds() -> Iterable[WorldSpec]:
    """Return only worlds intended for the main/default tab."""
    return _PRIMARY_WORLDS


def iter_legacy_worlds() -> Iterable[WorldSpec]:
    """Return worlds intended for the legacy tab."""
    return _LEGACY_WORLDS


def world_ids(include_legacy: bool = True) -> tuple[str, ...]:
    return tuple(world.id for world in iter_worlds(include_legacy=include_legacy))


def is_legacy_world(world_id: str) -> bool:
    return world_id in _LEGACY_WORLD_IDS


def get_world(world_id: Optional[str]) -> WorldSpec:
    lookup_id = world_id or DEFAULT_WORLD_ID
    try:
        return _WORLDS_BY_ID[lookup_id]
    except KeyError as exc:
        valid = ", ".join(world_ids())
        raise ValueError(f"Unknown world '{lookup_id}'. Valid worlds: {valid}") from exc
