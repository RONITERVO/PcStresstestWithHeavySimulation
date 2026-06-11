"""Garage Life Lab Windows launcher."""
from __future__ import annotations

import os
import subprocess
import sys
import time
import tkinter as tk
from dataclasses import replace
from pathlib import Path
from tkinter import ttk
from typing import Callable, Optional

from garage_life_presets import (
    CREATE_NO_WINDOW,
    HardwareInfo,
    LaunchPreset,
    build_presets,
    detect_hardware,
    preset_by_key,
    recommended_preset_key,
)
from worlds.registry import DEFAULT_WORLD_ID, get_world, iter_worlds
from worlds.spec import WorldSpec


APP_DIR = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
SOURCE_DIR = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent
LOG_DIR = SOURCE_DIR / "logs"
REQUIRED_IMPORT_CHECK = "import moderngl, moderngl_window, glfw, numpy, pyrr"


class GarageLifeLauncher:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Garage Life Lab")
        self.root.geometry("1120x720")
        self.root.minsize(980, 640)
        self.root.configure(bg="#05070a")
        self._apply_window_chrome()

        self.hw = detect_hardware(
            screen_width=self.root.winfo_screenwidth(),
            screen_height=self.root.winfo_screenheight(),
            python_executable=sys.executable,
        )
        self.presets = build_presets(self.hw)
        self.worlds = list(iter_worlds())
        self.world_names_by_id = {world.id: world.display_name for world in self.worlds}
        self.world_ids_by_name = {world.display_name: world.id for world in self.worlds}
        self.python_command = select_python_command()
        self.selected_world_id = tk.StringVar(value=DEFAULT_WORLD_ID)
        self.selected_world_name = tk.StringVar(value=self.world_names_by_id[DEFAULT_WORLD_ID])
        self.selected_key = tk.StringVar(value=recommended_preset_key(self.hw))
        self.status_var = tk.StringVar(value="Ready")
        self.advanced_open = tk.BooleanVar(value=False)
        self.process: Optional[subprocess.Popen[str]] = None
        self.log_path: Optional[Path] = None
        self.preview_tick = 0

        self.fields: dict[str, tk.StringVar] = {}
        self.command_var = tk.StringVar()
        self.summary_var = tk.StringVar()
        self.world_summary_var = tk.StringVar()
        self.hardware_var = tk.StringVar(value=self._hardware_summary())
        self.preview_image: Optional[tk.PhotoImage] = None

        self._configure_style()
        self._build_layout()
        self._load_selected_preset()
        self._draw_preview()
        self._poll_process()

    def _apply_window_chrome(self) -> None:
        if os.name == "nt":
            try:
                self.root.attributes("-alpha", 0.96)
            except tk.TclError:
                pass

    def _configure_style(self) -> None:
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TFrame", background="#05070a")
        style.configure("Panel.TFrame", background="#11151d")
        style.configure("TLabel", background="#05070a", foreground="#e7edf7")
        style.configure("Muted.TLabel", background="#05070a", foreground="#92a0b2")
        style.configure("Panel.TLabel", background="#11151d", foreground="#e7edf7")
        style.configure("MutedPanel.TLabel", background="#11151d", foreground="#97a3b5")
        style.configure("Card.TRadiobutton", background="#11151d", foreground="#e7edf7")
        style.map("Card.TRadiobutton", background=[("active", "#17202c")])
        style.configure("Primary.TButton", font=("Segoe UI", 12, "bold"), padding=(18, 12))
        style.configure("TButton", font=("Segoe UI", 10), padding=(12, 8))
        style.configure("TCheckbutton", background="#11151d", foreground="#e7edf7")
        style.configure("TEntry", fieldbackground="#080b10", foreground="#e7edf7")
        style.configure("TSpinbox", fieldbackground="#080b10", foreground="#e7edf7")

    def _build_layout(self) -> None:
        shell = ttk.Frame(self.root, padding=18)
        shell.pack(fill="both", expand=True)
        shell.columnconfigure(0, weight=3)
        shell.columnconfigure(1, weight=2)
        shell.rowconfigure(1, weight=1)

        header = ttk.Frame(shell)
        header.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 14))
        header.columnconfigure(0, weight=1)
        ttk.Label(
            header,
            text="Garage Life Lab",
            font=("Segoe UI", 24, "bold"),
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(
            header,
            text="One click starts the show. Advanced mode shows every knob before launch.",
            style="Muted.TLabel",
            font=("Segoe UI", 10),
        ).grid(row=1, column=0, sticky="w", pady=(2, 0))
        ttk.Label(
            header,
            textvariable=self.hardware_var,
            style="Muted.TLabel",
            justify="right",
        ).grid(row=0, column=1, rowspan=2, sticky="e")

        left = ttk.Frame(shell, style="Panel.TFrame", padding=18)
        left.grid(row=1, column=0, sticky="nsew", padx=(0, 14))
        left.columnconfigure(0, weight=1)
        left.rowconfigure(3, weight=1)

        ttk.Label(
            left,
            text="Choose World",
            style="Panel.TLabel",
            font=("Segoe UI", 13, "bold"),
        ).grid(row=0, column=0, sticky="w", pady=(0, 10))

        world_panel = ttk.Frame(left, style="Panel.TFrame")
        world_panel.grid(row=1, column=0, sticky="ew", pady=(0, 16))
        world_panel.columnconfigure(0, weight=1)
        self.world_combo = ttk.Combobox(
            world_panel,
            textvariable=self.selected_world_name,
            values=[world.display_name for world in self.worlds],
            state="readonly",
            font=("Segoe UI", 11),
        )
        self.world_combo.grid(row=0, column=0, sticky="ew")
        self.world_combo.bind("<<ComboboxSelected>>", self._world_changed)
        ttk.Label(
            world_panel,
            textvariable=self.world_summary_var,
            style="MutedPanel.TLabel",
            wraplength=600,
        ).grid(row=1, column=0, sticky="w", pady=(6, 0))

        ttk.Label(
            left,
            text="Choose PC Load",
            style="Panel.TLabel",
            font=("Segoe UI", 13, "bold"),
        ).grid(row=2, column=0, sticky="w", pady=(0, 10))

        cards = ttk.Frame(left, style="Panel.TFrame")
        cards.grid(row=3, column=0, sticky="nsew")
        cards.columnconfigure(0, weight=1)

        for index, preset in enumerate(self.presets):
            card = ttk.Frame(cards, style="Panel.TFrame", padding=(8, 8))
            card.grid(row=index, column=0, sticky="ew", pady=(0, 8))
            card.columnconfigure(1, weight=1)
            rb = ttk.Radiobutton(
                card,
                style="Card.TRadiobutton",
                variable=self.selected_key,
                value=preset.key,
                command=self._load_selected_preset,
            )
            rb.grid(row=0, column=0, rowspan=3, sticky="n", padx=(0, 8))
            label = preset.name
            if preset.key == recommended_preset_key(self.hw):
                label += "  Recommended"
            ttk.Label(
                card,
                text=label,
                style="Panel.TLabel",
                font=("Segoe UI", 12, "bold"),
            ).grid(row=0, column=1, sticky="w")
            ttk.Label(
                card,
                text=preset.audience,
                style="MutedPanel.TLabel",
            ).grid(row=1, column=1, sticky="w")
            ttk.Label(
                card,
                text=preset.description,
                style="MutedPanel.TLabel",
                wraplength=540,
            ).grid(row=2, column=1, sticky="w")

        advanced_header = ttk.Frame(left, style="Panel.TFrame")
        advanced_header.grid(row=4, column=0, sticky="ew", pady=(12, 4))
        advanced_header.columnconfigure(1, weight=1)
        ttk.Checkbutton(
            advanced_header,
            text="Advanced controls",
            variable=self.advanced_open,
            command=self._toggle_advanced,
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(
            advanced_header,
            text="Exact parameters remain editable for developers.",
            style="MutedPanel.TLabel",
        ).grid(row=0, column=1, sticky="e")

        self.advanced_frame = ttk.Frame(left, style="Panel.TFrame")
        self.advanced_frame.grid(row=5, column=0, sticky="ew")
        self._build_advanced_controls()

        right = ttk.Frame(shell, style="Panel.TFrame", padding=18)
        right.grid(row=1, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)

        ttk.Label(
            right,
            text="Live preview",
            style="Panel.TLabel",
            font=("Segoe UI", 13, "bold"),
        ).grid(row=0, column=0, sticky="w", pady=(0, 10))
        self.preview = tk.Canvas(
            right,
            height=280,
            bg="#06090d",
            highlightthickness=0,
            relief="flat",
        )
        self.preview.grid(row=1, column=0, sticky="nsew")

        ttk.Label(
            right,
            textvariable=self.summary_var,
            style="MutedPanel.TLabel",
            justify="left",
            wraplength=390,
        ).grid(row=2, column=0, sticky="ew", pady=(14, 8))

        ttk.Label(
            right,
            text="Exact launch",
            style="Panel.TLabel",
            font=("Segoe UI", 11, "bold"),
        ).grid(row=3, column=0, sticky="w", pady=(8, 4))
        command_box = tk.Text(
            right,
            height=5,
            wrap="word",
            bg="#080b10",
            fg="#c7f7ff",
            insertbackground="#c7f7ff",
            relief="flat",
            padx=10,
            pady=10,
            font=("Cascadia Mono", 9),
        )
        command_box.grid(row=4, column=0, sticky="ew")
        command_box.configure(state="disabled")
        self.command_box = command_box

        controls = ttk.Frame(right, style="Panel.TFrame")
        controls.grid(row=5, column=0, sticky="ew", pady=(14, 0))
        controls.columnconfigure(0, weight=1)
        ttk.Button(
            controls,
            text="Start Show",
            style="Primary.TButton",
            command=self.start_show,
        ).grid(row=0, column=0, sticky="ew", padx=(0, 8))
        ttk.Button(
            controls,
            text="Stop",
            command=self.stop_show,
        ).grid(row=0, column=1, sticky="e")
        ttk.Label(
            right,
            textvariable=self.status_var,
            style="MutedPanel.TLabel",
        ).grid(row=6, column=0, sticky="w", pady=(10, 0))
        self._toggle_advanced()

    def _build_advanced_controls(self) -> None:
        specs = [
            ("width", "Width"),
            ("height", "Height"),
            ("tile_size", "Tile"),
            ("substeps", "Substeps"),
            ("cpu_workers", "CPU workers"),
            ("cpu_matrix", "CPU matrix"),
            ("glow", "Glow"),
            ("exposure", "Exposure"),
            ("gamma", "Gamma"),
            ("contour_contrast", "Contours"),
            ("ray_steps", "Ray steps"),
            ("fx_intensity", "FX"),
            ("camera_speed", "Camera"),
            ("max_gpu_temp", "GPU C"),
            ("max_cpu_temp", "CPU C"),
        ]
        for index, (key, label) in enumerate(specs):
            row = index // 3
            col = (index % 3) * 2
            var = tk.StringVar()
            self.fields[key] = var
            ttk.Label(
                self.advanced_frame,
                text=label,
                style="MutedPanel.TLabel",
            ).grid(row=row, column=col, sticky="w", padx=(0, 6), pady=3)
            entry = ttk.Entry(
                self.advanced_frame,
                textvariable=var,
                width=9,
            )
            entry.grid(row=row, column=col + 1, sticky="w", padx=(0, 16), pady=3)
            var.trace_add("write", self._advanced_changed)

    def _hardware_summary(self) -> str:
        gpu_memory = ""
        if self.hw.gpu.memory_mb:
            gpu_memory = f" {self.hw.gpu.memory_mb // 1024} GB"
        python_line = "Bundled runtime" if getattr(sys, "frozen", False) else "Python: " + (
            " ".join(self.python_command) if self.python_command else "setup needed"
        )
        return (
            f"{self.hw.gpu.name}{gpu_memory}\n"
            f"{self.hw.cpu_name}\n"
            f"{self.hw.logical_cpus} threads, {self.hw.ram_gb} GB RAM\n"
            f"{python_line}"
        )

    def _selected_base_preset(self) -> LaunchPreset:
        return preset_by_key(self.presets, self.selected_key.get())

    def _selected_world(self) -> WorldSpec:
        return get_world(self.selected_world_id.get())

    def _world_changed(self, *_: object) -> None:
        world_name = self.selected_world_name.get()
        self.selected_world_id.set(self.world_ids_by_name.get(world_name, DEFAULT_WORLD_ID))
        self._refresh_command()
        self._draw_preview()

    def _load_selected_preset(self) -> None:
        preset = self._selected_base_preset()
        for key, value in preset.__dict__.items():
            if key in self.fields:
                self.fields[key].set(str(value))
        self._refresh_command()
        self._draw_preview()

    def _toggle_advanced(self) -> None:
        if self.advanced_open.get():
            self.advanced_frame.grid()
        else:
            self.advanced_frame.grid_remove()
        self._refresh_command()

    def _advanced_changed(self, *_: object) -> None:
        self._refresh_command()
        self._draw_preview()

    def _current_preset(self) -> LaunchPreset:
        preset = self._selected_base_preset()
        if not self.advanced_open.get():
            return preset

        values: dict[str, object] = {}
        integer_fields = {"width", "height", "tile_size", "substeps", "ray_steps", "cpu_workers", "cpu_matrix"}
        float_fields = {"glow", "exposure", "gamma", "contour_contrast", "fx_intensity", "camera_speed", "max_gpu_temp", "max_cpu_temp"}
        for key in integer_fields:
            values[key] = _safe_int(self.fields[key].get(), getattr(preset, key), minimum=0)
        for key in float_fields:
            values[key] = _safe_float(self.fields[key].get(), getattr(preset, key), minimum=0.0)
        values["width"] = max(640, int(values["width"]))
        values["height"] = max(360, int(values["height"]))
        values["tile_size"] = max(2, int(values["tile_size"]))
        values["substeps"] = max(1, int(values["substeps"]))
        values["ray_steps"] = max(32, min(160, int(values["ray_steps"])))
        values["fx_intensity"] = max(0.2, min(1.6, float(values["fx_intensity"])))
        values["camera_speed"] = max(0.05, min(2.0, float(values["camera_speed"])))
        return replace(preset, **values)

    def _refresh_command(self) -> None:
        preset = self._current_preset()
        world = self._selected_world()
        heat_score = preset.substeps * max(1, 20 - min(18, preset.tile_size)) + preset.ray_steps * 2 + preset.cpu_workers * 12
        notes = ", ".join(world.stability_notes) if world.stability_notes else "standard"
        self.world_summary_var.set(f"{world.display_name}: {notes}.")
        self.summary_var.set(
            f"{world.display_name} / {preset.short_name}: {preset.width}x{preset.height}, tile {preset.tile_size}, "
            f"{preset.substeps} sim steps, {preset.ray_steps} ray steps, FX {preset.fx_intensity:.1f}, "
            f"{preset.cpu_workers} CPU workers. "
            f"Thermal hold: GPU {preset.max_gpu_temp:.0f}C, CPU {preset.max_cpu_temp:.0f}C. "
            f"Estimated load: {_load_label(heat_score)}."
        )
        try:
            command_preview = display_command(launch_command(preset, world, self.python_command))
        except RuntimeError as exc:
            command_preview = str(exc)
        self.command_var.set(command_preview)
        self.command_box.configure(state="normal")
        self.command_box.delete("1.0", "end")
        self.command_box.insert("1.0", command_preview)
        self.command_box.configure(state="disabled")

    def _draw_preview(self) -> None:
        self.preview.delete("all")
        width = max(1, self.preview.winfo_width())
        height = max(1, self.preview.winfo_height())
        if width < 10 or height < 10:
            self.root.after(50, self._draw_preview)
            return
        preset = self._current_preset()
        world = self._selected_world()
        self.preview_tick += 1
        image_drawn = False
        if world.preview_image:
            image_path = APP_DIR / world.preview_image
            if image_path.exists():
                try:
                    self.preview_image = tk.PhotoImage(file=str(image_path))
                    self.preview.create_image(width // 2, height // 2, image=self.preview_image)
                    image_drawn = True
                except tk.TclError:
                    self.preview_image = None
        if not image_drawn:
            cell = max(8, min(26, preset.tile_size + 3))
            palette = list(world.preview_palette) or ["#06101f", "#0b3143", "#106569", "#1db38b", "#80e0b5", "#ff7ba5", "#ffe083"]
            for y in range(0, height, cell):
                for x in range(0, width, cell):
                    wave = (x // cell * 3 + y // cell * 5 + self.preview_tick) % len(palette)
                    color = palette[wave]
                    if (x + y + self.preview_tick * cell) % (cell * 7) == 0:
                        color = "#e7fff5"
                    if preset.fx_intensity > 1.0 and (x * 3 + y + self.preview_tick * 11) % (cell * 11) == 0:
                        color = "#ff70b6"
                    self.preview.create_rectangle(x, y, x + cell + 1, y + cell + 1, fill=color, outline="")
        self.preview.create_rectangle(0, 0, width, height, fill="#020409", stipple="gray25", outline="")
        self.preview.create_text(
            20,
            22,
            anchor="nw",
            fill="#d7f6ff",
            font=("Segoe UI", 15, "bold"),
            text=world.display_name,
        )
        self.preview.create_text(
            20,
            54,
            anchor="nw",
            fill="#9fb2c7",
            font=("Segoe UI", 10),
            text=f"{preset.name}  {preset.width}x{preset.height}  sim {preset.substeps}  rays {preset.ray_steps}",
        )
        self.preview.create_text(
            20,
            height - 48,
            anchor="nw",
            fill="#7ee7b0",
            font=("Cascadia Mono", 10, "bold"),
            text=f"GPU HOLD {preset.max_gpu_temp:.0f}C  CPU HOLD {preset.max_cpu_temp:.0f}C",
        )
        self.root.after(900, self._draw_preview)

    def start_show(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.status_var.set("Already running. Stop the current show first.")
            return

        preset = self._current_preset()
        world = self._selected_world()
        LOG_DIR.mkdir(exist_ok=True)
        self.log_path = LOG_DIR / "launcher-last-run.log"
        log_file = self.log_path.open("w", encoding="utf-8")
        try:
            command = launch_command(preset, world, self.python_command)
        except RuntimeError as exc:
            log_file.write(str(exc) + "\n")
            log_file.close()
            self.status_var.set(str(exc))
            return
        log_file.write(f"Started {time.ctime()}\n")
        log_file.write(f"World: {world.display_name} ({world.id})\n")
        log_file.write(f"Load preset: {preset.name} ({preset.key})\n")
        log_file.write("Command: " + display_command(command) + "\n\n")
        log_file.flush()
        try:
            self.process = subprocess.Popen(
                command,
                cwd=str(APP_DIR),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                creationflags=CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
        except OSError as exc:
            log_file.close()
            self.status_var.set(f"Could not start: {exc}")
            return

        self.status_var.set(f"Running {world.display_name} / {preset.name}. Log: {self.log_path}")

    def stop_show(self) -> None:
        if self.process is None or self.process.poll() is not None:
            self.status_var.set("No running show to stop.")
            return
        self.process.terminate()
        self.status_var.set("Stopping the show...")
        self.root.after(1800, self._kill_if_needed)

    def _kill_if_needed(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.kill()
            self.status_var.set("Show stopped.")

    def _poll_process(self) -> None:
        if self.process is not None:
            code = self.process.poll()
            if code is not None:
                if code == 0:
                    self.status_var.set("Show closed.")
                else:
                    suffix = f" See {self.log_path}." if self.log_path else ""
                    self.status_var.set(f"Show exited with code {code}.{suffix}")
                self.process = None
        self.root.after(1000, self._poll_process)


def _safe_int(value: str, fallback: int, minimum: int = 0) -> int:
    try:
        parsed = int(float(value))
    except (TypeError, ValueError):
        return fallback
    return max(minimum, parsed)


def _safe_float(value: str, fallback: float, minimum: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return fallback
    return max(minimum, parsed)


def _load_label(score: int) -> str:
    if score >= 800:
        return "maximum"
    if score >= 580:
        return "high"
    if score >= 300:
        return "medium"
    return "low"


def display_command(command: list[str]) -> str:
    return " ".join(_quote_process_arg(part) for part in command)


def _quote_process_arg(part: str) -> str:
    if not part:
        return '""'
    if any(ch.isspace() for ch in part):
        return '"' + part.replace('"', '\\"') + '"'
    return part


def select_python_command() -> Optional[list[str]]:
    if getattr(sys, "frozen", False):
        return [sys.executable]

    candidates: list[list[str]] = []
    venv_scripts = SOURCE_DIR / ".venv" / "Scripts"
    candidates.extend([
        [str(venv_scripts / "pythonw.exe")],
        [str(venv_scripts / "python.exe")],
    ])

    current = Path(sys.executable)
    if current.exists():
        if current.name.lower() == "python.exe":
            pythonw = current.with_name("pythonw.exe")
            candidates.append([str(pythonw)])
        candidates.append([str(current)])

    candidates.extend([
        ["pyw", "-3.9"],
        ["py", "-3.9"],
        ["pythonw"],
        ["python"],
    ])

    seen: set[tuple[str, ...]] = set()
    for candidate in candidates:
        key = tuple(part.lower() for part in candidate)
        if key in seen:
            continue
        seen.add(key)
        if python_can_run_simulation(candidate):
            return candidate
    return None


def python_can_run_simulation(command: list[str]) -> bool:
    if not _command_exists(command):
        return False
    try:
        result = subprocess.run(
            command + ["-c", REQUIRED_IMPORT_CHECK],
            cwd=str(APP_DIR),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=8.0,
            creationflags=CREATE_NO_WINDOW if os.name == "nt" else 0,
            check=False,
        )
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        return False
    return result is not None and result.returncode == 0


def _command_exists(command: list[str]) -> bool:
    first = command[0]
    if "\\" in first or "/" in first:
        return Path(first).exists()
    try:
        result = subprocess.run(
            ["where" if os.name == "nt" else "which", first],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=CREATE_NO_WINDOW if os.name == "nt" else 0,
            check=False,
        )
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


def launch_command(
    preset: LaunchPreset,
    world: WorldSpec,
    python_command: Optional[list[str]] = None,
) -> list[str]:
    world_args = ["--world", world.id]
    if getattr(sys, "frozen", False):
        return [sys.executable, "--run-sim"] + world_args + preset.args()

    command = python_command or select_python_command()
    if not command:
        raise RuntimeError(
            "Garage Life Lab needs Python 3.9 with the graphics packages installed. "
            "Run build_windows_app.ps1 or install requirements into .venv."
        )
    return command + [str(APP_DIR / "main.py")] + world_args + preset.args()


def run_simulation(argv: list[str]) -> None:
    import main as simulation_main

    sys.argv = [str(APP_DIR / "main.py")] + argv
    simulation_main.main()


def main() -> None:
    if "--run-sim" in sys.argv:
        index = sys.argv.index("--run-sim")
        run_simulation(sys.argv[index + 1 :])
        return

    root = tk.Tk()
    GarageLifeLauncher(root)
    root.mainloop()


if __name__ == "__main__":
    main()
