"""Garage Life Lab command router."""
from __future__ import annotations

import sys

from worlds.registry import DEFAULT_WORLD_ID, is_legacy_world, iter_worlds


def _print_worlds() -> None:
    for world in iter_worlds():
        badges: list[str] = []
        if world.id == DEFAULT_WORLD_ID:
            badges.append("default")
        if is_legacy_world(world.id):
            badges.append("legacy")
        badges.extend(note for note in world.stability_notes if note not in badges)
        suffix = f" ({', '.join(badges)})" if badges else ""
        print(f"{world.id}\t{world.display_name}{suffix}")


def main() -> None:
    if "--list-worlds" in sys.argv:
        _print_worlds()
        return

    # Keep lookup/listing usable in source-only handoffs where GPU dependencies
    # may not be installed yet.  The renderer imports moderngl only when needed.
    from engine import main as run_engine

    run_engine()


if __name__ == "__main__":
    main()
