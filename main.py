"""Garage Life Lab command router."""
from __future__ import annotations

import sys

from engine import main as run_engine
from worlds.registry import iter_worlds


def _print_worlds() -> None:
    for world in iter_worlds():
        notes = ", ".join(world.stability_notes)
        suffix = f" ({notes})" if notes else ""
        print(f"{world.id}\t{world.display_name}{suffix}")


def main() -> None:
    if "--list-worlds" in sys.argv:
        _print_worlds()
        return
    run_engine()


if __name__ == "__main__":
    main()
