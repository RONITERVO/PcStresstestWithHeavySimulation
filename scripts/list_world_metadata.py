"""List registered Garage Life Lab world metadata for validation runs."""
from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from worlds.registry import iter_worlds  # noqa: E402
from worlds.spec import WorldSpec  # noqa: E402


def powershell_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)


def smoke_command(world_id: str) -> str:
    smoke_script = r".\scripts\smoke_worlds.ps1"
    report_path = f"logs\\validation\\{safe_name(world_id)}-smoke.md"
    return (
        f"& {powershell_quote(smoke_script)} "
        f"-World {powershell_quote(world_id)} "
        f"-DisableThermalHold "
        f"-TimeoutSeconds 20 "
        f"-ReportPath {powershell_quote(report_path)}"
    )


def is_minecraft_candidate(world: WorldSpec) -> bool:
    text = f"{world.id} {world.display_name}".lower()
    return "minecraft" in text


def selected_worlds(args: argparse.Namespace) -> list[WorldSpec]:
    worlds = list(iter_worlds())
    if args.world:
        by_id = {world.id: world for world in worlds}
        missing = [world_id for world_id in args.world if world_id not in by_id]
        if missing:
            raise SystemExit(f"Unknown world id(s): {', '.join(missing)}")
        worlds = [by_id[world_id] for world_id in args.world]
    if args.minecraft_only:
        worlds = [world for world in worlds if is_minecraft_candidate(world)]
    return worlds


def world_row(world: WorldSpec) -> dict[str, object]:
    return {
        "id": world.id,
        "display_name": world.display_name,
        "stability_notes": list(world.stability_notes),
        "uses_audio": world.uses_audio,
        "hud_subtitle": world.hud_subtitle,
        "preview_image": world.preview_image or "",
        "preview_palette": list(world.preview_palette),
        "default_overrides": dict(world.default_overrides),
        "low_smoke_command": smoke_command(world.id),
    }


def compact(value: object) -> str:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    return str(value)


def markdown_cell(value: object, max_length: int = 700) -> str:
    text = compact(value).replace("\r\n", " ").replace("\n", " ").replace("\r", " ")
    text = text.strip()
    if len(text) > max_length:
        text = text[: max_length - 3] + "..."
    return text.replace("\\", "\\\\").replace("|", "\\|").replace("`", "\\`")


def markdown_code(value: object) -> str:
    text = compact(value)
    return "<code>" + html.escape(text).replace("|", "&#124;") + "</code>"


def print_table(rows: Iterable[dict[str, object]]) -> None:
    columns = (
        "id",
        "display_name",
        "stability_notes",
        "uses_audio",
        "preview_image",
        "preview_palette",
        "default_overrides",
    )
    materialized = [{column: compact(row[column]) for column in columns} for row in rows]
    widths = {
        column: max(len(column), *(len(row[column]) for row in materialized))
        for column in columns
    }
    print("  ".join(column.ljust(widths[column]) for column in columns))
    print("  ".join("-" * widths[column] for column in columns))
    for row in materialized:
        print("  ".join(row[column].ljust(widths[column]) for column in columns))


def print_markdown(rows: list[dict[str, object]]) -> None:
    print("# Garage Life Lab World Metadata")
    print()
    print("| World ID | Display name | Notes | Audio | Preview image | Preview palette | Default overrides |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for row in rows:
        notes = ", ".join(row["stability_notes"]) or "standard"
        print(
            f"| {markdown_cell(row['id'])} | {markdown_cell(row['display_name'])} | "
            f"{markdown_cell(notes)} | {markdown_cell(row['uses_audio'])} | "
            f"{markdown_cell(row['preview_image'] or 'palette fallback')} | "
            f"{markdown_code(row['preview_palette'])} | "
            f"{markdown_code(row['default_overrides'])} |"
        )

    print()
    print("## Low Smoke Commands")
    print()
    print("```powershell")
    for row in rows:
        print(row["low_smoke_command"])
    print("```")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="List world metadata and reproducible low-smoke commands."
    )
    parser.add_argument("--world", action="append", help="World id to include. Repeatable.")
    parser.add_argument(
        "--minecraft-only",
        action="store_true",
        help="Only include worlds whose id or display name contains Minecraft.",
    )
    parser.add_argument(
        "--format",
        choices=("table", "markdown", "json"),
        default="table",
        help="Output format.",
    )
    parser.add_argument(
        "--python",
        default=r".\.venv\Scripts\python.exe",
        help="Deprecated; retained for compatibility because smoke commands use the harness.",
    )
    args = parser.parse_args()

    rows = [world_row(world) for world in selected_worlds(args)]
    if args.format == "json":
        print(json.dumps(rows, indent=2, sort_keys=True))
    elif args.format == "markdown":
        print_markdown(rows)
    else:
        print_table(rows)


if __name__ == "__main__":
    main()
