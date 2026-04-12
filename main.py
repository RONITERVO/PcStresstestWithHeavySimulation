"""Garage Life Lab: 1080p bio-simulation space heater."""
from __future__ import annotations

import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import moderngl
import moderngl_window as mglw
import numpy as np
from moderngl_window import geometry


CREATE_NO_WINDOW = 0x08000000
CPU_SENSOR_POWERSHELL = """
$ErrorActionPreference = 'Stop'
$namespaces = @('root\\LibreHardwareMonitor', 'root\\OpenHardwareMonitor')
$preferredPattern = 'CPU Package|Tctl/Tdie|Tdie|Core Max|CPU CCD|CPU Die|Package'
foreach ($ns in $namespaces) {
    try {
        $sensors = Get-CimInstance -Namespace $ns -ClassName Sensor |
            Where-Object { $_.SensorType -eq 'Temperature' } |
            ForEach-Object {
                [PSCustomObject]@{
                    Name = [string]$_.Name
                    Value = [double]$_.Value
                }
            }
        if ($sensors) {
            $preferred = $sensors |
                Where-Object { $_.Name -match $preferredPattern } |
                Sort-Object Value -Descending |
                Select-Object -First 1
            if (-not $preferred) {
                $preferred = $sensors |
                    Sort-Object Value -Descending |
                    Select-Object -First 1
            }
            if ($preferred) {
                $value = $preferred.Value.ToString('0.0', [System.Globalization.CultureInfo]::InvariantCulture)
                Write-Output ('{0}|{1}' -f $preferred.Name, $value)
                exit 0
            }
        }
    } catch {
    }
}
exit 1
""".strip()

FONT_3X5 = {
    " ": ("000", "000", "000", "000", "000"),
    "-": ("000", "000", "111", "000", "000"),
    ".": ("000", "000", "000", "000", "010"),
    ":": ("000", "010", "000", "010", "000"),
    "?": ("111", "001", "011", "000", "010"),
    "0": ("111", "101", "101", "101", "111"),
    "1": ("010", "110", "010", "010", "111"),
    "2": ("111", "001", "111", "100", "111"),
    "3": ("111", "001", "111", "001", "111"),
    "4": ("101", "101", "111", "001", "001"),
    "5": ("111", "100", "111", "001", "111"),
    "6": ("111", "100", "111", "101", "111"),
    "7": ("111", "001", "010", "010", "010"),
    "8": ("111", "101", "111", "101", "111"),
    "9": ("111", "101", "111", "001", "111"),
    "A": ("111", "101", "111", "101", "101"),
    "B": ("110", "101", "110", "101", "110"),
    "C": ("111", "100", "100", "100", "111"),
    "D": ("110", "101", "101", "101", "110"),
    "E": ("111", "100", "110", "100", "111"),
    "F": ("111", "100", "110", "100", "100"),
    "G": ("111", "100", "101", "101", "111"),
    "H": ("101", "101", "111", "101", "101"),
    "I": ("111", "010", "010", "010", "111"),
    "J": ("001", "001", "001", "101", "111"),
    "K": ("101", "101", "110", "101", "101"),
    "L": ("100", "100", "100", "100", "111"),
    "M": ("101", "111", "111", "101", "101"),
    "N": ("101", "111", "111", "111", "101"),
    "O": ("111", "101", "101", "101", "111"),
    "P": ("111", "101", "111", "100", "100"),
    "Q": ("111", "101", "101", "111", "011"),
    "R": ("110", "101", "110", "101", "101"),
    "S": ("111", "100", "111", "001", "111"),
    "T": ("111", "010", "010", "010", "010"),
    "U": ("101", "101", "101", "101", "111"),
    "V": ("101", "101", "101", "101", "010"),
    "W": ("101", "101", "111", "111", "101"),
    "X": ("101", "101", "010", "101", "101"),
    "Y": ("101", "101", "010", "010", "010"),
    "Z": ("111", "001", "010", "100", "111"),
}


@dataclass(frozen=True)
class ThermalHoldState:
    lines: Sequence[str]
    log_path: Path


def _sanitize_text(line: str) -> str:
    return "".join(ch if ch in FONT_3X5 else "?" for ch in line.upper())


def _draw_glyph(
    canvas: np.ndarray,
    glyph: Sequence[str],
    x: int,
    y: int,
    scale: int,
    color: Sequence[int],
) -> None:
    height, width, _ = canvas.shape
    for row_index, row in enumerate(glyph):
        for col_index, cell in enumerate(row):
            if cell != "1":
                continue
            y0 = y + row_index * scale
            x0 = x + col_index * scale
            y1 = min(y0 + scale, height)
            x1 = min(x0 + scale, width)
            if x1 > 0 and y1 > 0 and x0 < width and y0 < height:
                canvas[max(y0, 0):y1, max(x0, 0):x1] = color


def _draw_text_line(
    canvas: np.ndarray,
    line: str,
    y: int,
    scale: int,
    color: Sequence[int],
    shadow: bool = True,
) -> None:
    sanitized = _sanitize_text(line)
    glyph_width = 3 * scale
    spacing = scale
    line_width = max(0, len(sanitized) * (glyph_width + spacing) - spacing)
    x = max((canvas.shape[1] - line_width) // 2, scale * 3)
    if shadow:
        shadow_offset = max(1, scale // 3)
        shadow_color = np.clip(np.array(color, dtype=np.int16) // 4, 0, 255).astype(np.uint8)
        cursor = x + shadow_offset
        for char in sanitized:
            _draw_glyph(canvas, FONT_3X5.get(char, FONT_3X5["?"]), cursor, y + shadow_offset, scale, shadow_color)
            cursor += glyph_width + spacing
    cursor = x
    for char in sanitized:
        _draw_glyph(canvas, FONT_3X5.get(char, FONT_3X5["?"]), cursor, y, scale, color)
        cursor += glyph_width + spacing


def build_hold_frame(lines: Sequence[str], size: Sequence[int]) -> np.ndarray:
    width = max(int(size[0]), 320)
    height = max(int(size[1]), 180)
    x_gradient = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
    y_gradient = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    stripe = 0.5 + 0.5 * np.sin(x_gradient * 18.0 + y_gradient * 11.0)

    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[..., 0] = np.clip(18.0 + 40.0 * (1.0 - y_gradient) + stripe * 12.0, 0, 255).astype(np.uint8)
    frame[..., 1] = np.clip(11.0 + 20.0 * (1.0 - y_gradient) + stripe * 8.0, 0, 255).astype(np.uint8)
    frame[..., 2] = np.clip(10.0 + 16.0 * (1.0 - y_gradient) + stripe * 7.0, 0, 255).astype(np.uint8)

    margin = max(24, min(width, height) // 12)
    max_chars = max((len(_sanitize_text(line)) for line in lines), default=1)
    usable_width = max(width - 2 * margin, 120)
    usable_height = max(height - 2 * margin, 80)
    scale_by_width = usable_width // max(1, max_chars * 4 - 1)
    scale_by_height = usable_height // max(1, len(lines) * 7 - 2)
    scale = max(2, min(scale_by_width, scale_by_height, 32))
    line_height = 5 * scale
    line_gap = 2 * scale
    total_height = len(lines) * line_height + max(0, len(lines) - 1) * line_gap
    start_y = max((height - total_height) // 2, margin)

    for index, line in enumerate(lines):
        if index == 0:
            color = (255, 132, 78)
        elif index >= len(lines) - 2:
            color = (204, 214, 220)
        else:
            color = (248, 238, 228)
        line_y = start_y + index * (line_height + line_gap)
        _draw_text_line(frame, line, line_y, scale, color)

    return frame


class GarageHeatShow(mglw.WindowConfig):
    """GPU-heavy biosphere simulation with CPU burners and thermal hold."""

    title = "Garage Life Lab - Biosphere"
    gl_version = (4, 5)
    resource_dir = Path(__file__).parent
    window_size = (1920, 1080)
    aspect_ratio = window_size[0] / window_size[1]
    samples = 0
    fullscreen = False
    vsync = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.args = getattr(type(self), "argv", None)
        if self.args is None:
            raise RuntimeError("GarageHeatShow requires command-line arguments")

        self.base_title = type(self).title
        self.stop_event = threading.Event()
        self.thermal_thread_stop = threading.Event()
        self.telemetry_lock = threading.Lock()
        self.latest_temperatures: Dict[str, float] = {}
        self.cpu_sensor_name: Optional[str] = None
        self.cpu_sensor_retry_after = 0.0
        self.gpu_sensor_failures = 0
        self.next_title_refresh_at = 0.0
        self.thermal_hold: Optional[ThermalHoldState] = None
        self.hold_texture = None
        self.cpu_threads: List[threading.Thread] = []

        self.ctx.disable(moderngl.DEPTH_TEST)
        self.quad = geometry.quad_fs()
        self.update_program = self.load_program(
            vertex_shader="shaders/quad.vert",
            fragment_shader="shaders/life_update.frag",
        )
        self.display_program = self.load_program(
            vertex_shader="shaders/quad.vert",
            fragment_shader="shaders/life_display.frag",
        )
        self.message_program = self.load_program(
            vertex_shader="shaders/quad.vert",
            fragment_shader="shaders/present_texture.frag",
        )

        self.update_program["stateTex"].value = 0
        self.display_program["stateTex"].value = 0
        self.message_program["displayTex"].value = 0

        if (self.args.width, self.args.height) != self.window_size:
            self.wnd.resize(self.args.width, self.args.height)

        self._init_simulation_resources()
        self._sync_static_uniforms()

        if self.args.cpu_workers > 0:
            self._spin_cpu_workers()
        if not self.args.no_thermal_hold:
            self._spin_thermal_watchdog()

    def _run_command(self, command: Sequence[str], timeout: float) -> Optional[subprocess.CompletedProcess[str]]:
        try:
            return subprocess.run(
                list(command),
                capture_output=True,
                text=True,
                timeout=timeout,
                creationflags=CREATE_NO_WINDOW,
                check=False,
            )
        except (FileNotFoundError, OSError, subprocess.SubprocessError):
            return None

    def _read_gpu_temp(self) -> Optional[float]:
        result = self._run_command(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            timeout=3.0,
        )
        if result is None or result.returncode != 0:
            return None
        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if not lines:
            return None
        try:
            return float(lines[0])
        except ValueError:
            return None

    def _read_cpu_temp(self) -> Optional[float]:
        now = time.monotonic()
        if now < self.cpu_sensor_retry_after:
            return None
        result = self._run_command(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", CPU_SENSOR_POWERSHELL],
            timeout=4.0,
        )
        if result is None or result.returncode != 0:
            self.cpu_sensor_name = None
            self.cpu_sensor_retry_after = now + 60.0
            return None
        line = result.stdout.strip().splitlines()[-1].strip() if result.stdout.strip() else ""
        if "|" not in line:
            self.cpu_sensor_name = None
            self.cpu_sensor_retry_after = now + 60.0
            return None
        name, value = line.split("|", 1)
        try:
            temperature = float(value)
        except ValueError:
            self.cpu_sensor_name = None
            self.cpu_sensor_retry_after = now + 60.0
            return None
        self.cpu_sensor_name = name.strip() or "CPU"
        self.cpu_sensor_retry_after = now + max(1.0, float(self.args.thermal_poll_seconds))
        return temperature

    def _spin_thermal_watchdog(self) -> None:
        thread = threading.Thread(
            target=self._thermal_watchdog,
            name="thermal-watchdog",
            daemon=True,
        )
        thread.start()

    def _thermal_watchdog(self) -> None:
        poll_interval = max(1.0, float(self.args.thermal_poll_seconds))
        while not self.thermal_thread_stop.is_set() and self.thermal_hold is None:
            gpu_temp = self._read_gpu_temp() if self.args.max_gpu_temp > 0 else None
            cpu_temp = self._read_cpu_temp() if self.args.max_cpu_temp > 0 else None

            with self.telemetry_lock:
                if gpu_temp is None:
                    self.latest_temperatures.pop("GPU", None)
                else:
                    self.latest_temperatures["GPU"] = gpu_temp
                if cpu_temp is None:
                    self.latest_temperatures.pop("CPU", None)
                else:
                    self.latest_temperatures["CPU"] = cpu_temp

            reasons: List[str] = []
            notes: List[str] = []

            if self.args.max_gpu_temp > 0:
                if gpu_temp is None:
                    self.gpu_sensor_failures += 1
                    if self.gpu_sensor_failures >= 3:
                        reasons.append("GPU SENSOR OFFLINE")
                else:
                    self.gpu_sensor_failures = 0
                    if gpu_temp > self.args.max_gpu_temp:
                        reasons.append(
                            f"GPU {gpu_temp:.1f}C OVER LIMIT {self.args.max_gpu_temp:.1f}C"
                        )

            if self.args.max_cpu_temp > 0:
                if cpu_temp is None:
                    notes.append("CPU SENSOR OFFLINE")
                elif cpu_temp > self.args.max_cpu_temp:
                    reasons.append(
                        f"CPU {cpu_temp:.1f}C OVER LIMIT {self.args.max_cpu_temp:.1f}C"
                    )

            if reasons:
                self._trigger_thermal_hold(reasons, notes)
                return

            if self.thermal_thread_stop.wait(poll_interval):
                return

    def _write_thermal_logs(self, reasons: Sequence[str], notes: Sequence[str]) -> Path:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_dir = self.resource_dir / "logs"
        log_dir.mkdir(exist_ok=True)
        log_path = log_dir / "thermal_events.log"
        last_event_path = log_dir / "last_thermal_hold.txt"

        log_lines = [f"[{timestamp}] THERMAL HOLD", *reasons]
        if notes:
            log_lines.extend(notes)
        gpu_temp = self.latest_temperatures.get("GPU")
        cpu_temp = self.latest_temperatures.get("CPU")
        if gpu_temp is not None:
            log_lines.append(f"LAST GPU TEMP {gpu_temp:.1f}C")
        if cpu_temp is not None:
            sensor_name = self.cpu_sensor_name or "CPU"
            log_lines.append(f"LAST {sensor_name.upper()} TEMP {cpu_temp:.1f}C")
        log_lines.append("LOADS STOPPED TO COOL SYSTEM")
        log_lines.append("")

        with log_path.open("a", encoding="utf-8") as handle:
            handle.write("\n".join(log_lines))
        last_event_path.write_text("\n".join(log_lines[:-1]), encoding="utf-8")
        return log_path

    def _build_hold_texture(self, lines: Sequence[str]) -> None:
        frame = build_hold_frame(lines, self.wnd.buffer_size)
        if self.hold_texture is not None:
            self.hold_texture.release()
        self.hold_texture = self.ctx.texture(
            self.wnd.buffer_size,
            3,
            data=frame.tobytes(),
            alignment=1,
        )
        self.hold_texture.filter = (moderngl.NEAREST, moderngl.NEAREST)

    def _trigger_thermal_hold(self, reasons: Sequence[str], notes: Sequence[str]) -> None:
        if self.thermal_hold is not None:
            return
        log_path = self._write_thermal_logs(reasons, notes)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        hold_lines = [
            "THERMAL HOLD",
            *reasons,
            *notes,
            "LOADS STOPPED TO COOL SYSTEM",
            timestamp,
            "SEE LOGS THERMAL EVENTS LOG",
            "PRESS ESC TO EXIT",
        ]
        self.thermal_hold = ThermalHoldState(lines=hold_lines, log_path=log_path)
        self.stop_event.set()
        self.thermal_thread_stop.set()
        self.wnd.title = f"{self.base_title} | THERMAL HOLD"
        self._build_hold_texture(hold_lines)

    def _refresh_window_title(self) -> None:
        now = time.monotonic()
        if now < self.next_title_refresh_at or self.thermal_hold is not None:
            return
        self.next_title_refresh_at = now + 1.0
        parts: List[str] = []
        with self.telemetry_lock:
            gpu_temp = self.latest_temperatures.get("GPU")
            cpu_temp = self.latest_temperatures.get("CPU")
        if gpu_temp is not None:
            parts.append(f"GPU {gpu_temp:.0f}C")
        if cpu_temp is not None:
            parts.append(f"CPU {cpu_temp:.0f}C")
        elif self.args.max_cpu_temp > 0 and not self.args.no_thermal_hold:
            parts.append("CPU SENSOR OFFLINE")
        if parts:
            self.wnd.title = f"{self.base_title} | " + " | ".join(parts)
        else:
            self.wnd.title = self.base_title

    def _init_simulation_resources(self) -> None:
        if hasattr(self, "state_textures"):
            for fbo in getattr(self, "framebuffers", []):
                fbo.release()
            for tex in self.state_textures:
                tex.release()
        buffer_size = self.wnd.buffer_size
        self.state_textures = [
            self.ctx.texture(buffer_size, 4, dtype="f4")
            for _ in range(2)
        ]
        for tex in self.state_textures:
            tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            tex.repeat_x = True
            tex.repeat_y = True
        self.framebuffers = [
            self.ctx.framebuffer(color_attachments=[tex])
            for tex in self.state_textures
        ]
        self.active_state = 0
        self._seed_field()
        self._update_resolution_uniforms()

    def _update_resolution_uniforms(self) -> None:
        buffer_size = self.wnd.buffer_size
        resolution = np.array([buffer_size[0], buffer_size[1]], dtype="f4")
        self.update_program["resolution"].write(resolution)
        self.display_program["resolution"].write(resolution)

    def _seed_field(self) -> None:
        width, height = self.state_textures[0].size
        field = np.zeros((height, width, 4), dtype=np.float32)
        field[..., 0] = 0.96
        field[..., 1] = 0.02

        rng = np.random.default_rng(2025)
        y_indices, x_indices = np.meshgrid(
            np.arange(height),
            np.arange(width),
            indexing="ij",
        )
        x_norm = x_indices / max(width - 1, 1)
        y_norm = y_indices / max(height - 1, 1)

        macro_wave = (
            0.48
            + 0.18 * np.sin(x_norm * 8.0 + 0.7)
            + 0.15 * np.cos(y_norm * 6.0 - 1.3)
            + 0.12 * np.sin((x_norm + y_norm) * 10.0)
        )
        field[..., 3] = np.clip(
            macro_wave + rng.random((height, width), dtype=np.float32) * 0.14,
            0.0,
            1.0,
        )
        field[..., 2] = rng.random((height, width), dtype=np.float32) * 0.22

        mega_blob_count = max(6, (width * height) // 900000)
        micro_blob_count = max(90, (width * height) // 90000)

        for _ in range(mega_blob_count):
            cx = rng.integers(0, width)
            cy = rng.integers(0, height)
            radius = rng.integers(
                max(90, min(width, height) // 10),
                max(180, min(width, height) // 3),
            )
            mask = (x_indices - cx) ** 2 + (y_indices - cy) ** 2 <= radius ** 2
            field[..., 1][mask] = np.maximum(field[..., 1][mask], rng.uniform(0.45, 1.0))
            field[..., 0][mask] = np.minimum(field[..., 0][mask], rng.uniform(0.0, 0.35))
            field[..., 3][mask] = np.clip(field[..., 3][mask] + rng.uniform(0.12, 0.32), 0.0, 1.0)

        for _ in range(micro_blob_count):
            cx = rng.integers(0, width)
            cy = rng.integers(0, height)
            radius = rng.integers(8, 52)
            mask = (x_indices - cx) ** 2 + (y_indices - cy) ** 2 <= radius ** 2
            field[..., 1][mask] = np.maximum(field[..., 1][mask], rng.uniform(0.18, 0.7))
            field[..., 0][mask] = np.minimum(field[..., 0][mask], rng.uniform(0.05, 0.7))

        current_bands = np.sin(x_norm * 13.0 + np.cos(y_norm * 9.0) * 2.2)
        current_mask = np.clip((current_bands - 0.72) * 2.3, 0.0, 1.0)
        field[..., 1] += current_mask * 0.16
        field[..., 0] -= current_mask * 0.18
        field[..., 2] += np.clip(1.0 - np.abs(current_bands), 0.0, 1.0) * 0.04

        field[..., 0] = np.clip(field[..., 0] + rng.standard_normal((height, width), dtype=np.float32) * 0.02, 0.0, 1.0)
        field[..., 1] = np.clip(field[..., 1], 0.0, 1.0)
        field[..., 2] = np.clip(field[..., 2], 0.0, 1.0)
        field[..., 3] = np.clip(field[..., 3], 0.0, 1.0)

        for tex in self.state_textures:
            tex.write(field.tobytes())

    def _sync_static_uniforms(self) -> None:
        self.update_program["diffU"].value = self.args.diff_u
        self.update_program["diffV"].value = self.args.diff_v
        self.update_program["dt"].value = self.args.time_step
        self.update_program["laplaceScale"].value = self.args.laplace_scale
        self.update_program["noiseStrength"].value = self.args.noise_strength
        self.update_program["parameterDrift"].value = self.args.param_drift

        self.display_program["exposure"].value = self.args.exposure
        self.display_program["glow"].value = self.args.glow
        self.display_program["gamma"].value = self.args.gamma
        self.display_program["contourContrast"].value = self.args.contour_contrast

    def _spin_cpu_workers(self) -> None:
        for worker_id in range(self.args.cpu_workers):
            thread = threading.Thread(
                target=self._cpu_burner,
                args=(worker_id,),
                name=f"cpu-burner-{worker_id}",
                daemon=True,
            )
            thread.start()
            self.cpu_threads.append(thread)

    def _cpu_burner(self, worker_id: int) -> None:
        matrix_n = self.args.cpu_matrix
        rng = np.random.default_rng(worker_id + 42)
        a = rng.standard_normal((matrix_n, matrix_n), dtype=np.float32)
        b = rng.standard_normal((matrix_n, matrix_n), dtype=np.float32)
        while not self.stop_event.is_set():
            np.matmul(a, b, out=a)
            norm = np.linalg.norm(a)
            if norm > 0:
                a /= norm
            real = rng.standard_normal(matrix_n * 8, dtype=np.float32)
            imag = rng.standard_normal(matrix_n * 8, dtype=np.float32)
            signal = (real + 1j * imag).astype(np.complex64)
            _ = np.fft.fft(signal)
            a, b = b, a

    def render(self, time_value: float, frame_time: float) -> None:
        if self.thermal_hold is not None:
            if self.hold_texture is None or self.hold_texture.size != self.wnd.buffer_size:
                self._build_hold_texture(self.thermal_hold.lines)
            self.ctx.screen.use()
            self.ctx.viewport = (0, 0, *self.wnd.buffer_size)
            self.hold_texture.use(location=0)
            self.quad.render(self.message_program)
            return

        if self.state_textures[0].size != self.wnd.buffer_size:
            self._init_simulation_resources()
            self._sync_static_uniforms()

        self._refresh_window_title()

        animated_time = time_value * self.args.anim_speed
        substeps = max(1, self.args.substeps)
        for _ in range(substeps):
            self._step_simulation(animated_time)
        self._render_display(animated_time)

    def _step_simulation(self, animated_time: float) -> None:
        current = self.state_textures[self.active_state]
        next_index = 1 - self.active_state
        target_fbo = self.framebuffers[next_index]
        target_fbo.use()
        self.ctx.viewport = (0, 0, *current.size)
        current.use(location=0)
        self.update_program["time"].value = animated_time
        self.update_program["feed"].value = self.args.feed
        self.update_program["kill"].value = self.args.kill
        self.quad.render(self.update_program)
        self.active_state = next_index

    def _render_display(self, animated_time: float) -> None:
        self.ctx.screen.use()
        self.ctx.viewport = (0, 0, *self.wnd.buffer_size)
        self.state_textures[self.active_state].use(location=0)
        self.display_program["time"].value = animated_time
        self.display_program["colorShift"].value = (
            animated_time * self.args.color_shift_speed
        ) % 10.0
        self.quad.render(self.display_program)

    def resize(self, width: int, height: int):  # type: ignore[override]
        if self.thermal_hold is not None:
            self._build_hold_texture(self.thermal_hold.lines)
            return
        self._init_simulation_resources()
        self._sync_static_uniforms()

    def destroy(self) -> None:
        self.stop_event.set()
        self.thermal_thread_stop.set()
        for thread in self.cpu_threads:
            thread.join(timeout=1.0)
        self.cpu_threads.clear()
        if self.hold_texture is not None:
            self.hold_texture.release()
            self.hold_texture = None
        super().destroy()

    @classmethod
    def add_arguments(cls, parser) -> None:  # type: ignore[override]
        parser.add_argument("--width", type=int, default=cls.window_size[0], help="Render width")
        parser.add_argument("--height", type=int, default=cls.window_size[1], help="Render height")
        parser.add_argument("--feed", type=float, default=0.029, help="Gray-Scott feed rate")
        parser.add_argument("--kill", type=float, default=0.057, help="Gray-Scott kill rate")
        parser.add_argument("--diff-u", type=float, default=0.16, help="Diffusion rate for U")
        parser.add_argument("--diff-v", type=float, default=0.08, help="Diffusion rate for V")
        parser.add_argument("--time-step", dest="time_step", type=float, default=1.0, help="Simulation time step")
        parser.add_argument("--substeps", type=int, default=12, help="Simulation steps per frame")
        parser.add_argument("--laplace-scale", type=float, default=1.0, help="Global laplacian multiplier")
        parser.add_argument("--noise-strength", type=float, default=0.015, help="Stochastic noise injected each step")
        parser.add_argument("--param-drift", type=float, default=0.004, help="Sinusoidal feed/kill drift amplitude")
        parser.add_argument("--anim-speed", type=float, default=1.0, help="Global animation multiplier")
        parser.add_argument("--color-shift-speed", type=float, default=0.05, help="Palette cycle speed")
        parser.add_argument("--exposure", type=float, default=1.4, help="Display exposure")
        parser.add_argument("--glow", type=float, default=1.1, help="Display glow factor")
        parser.add_argument("--gamma", type=float, default=1.2, help="Display gamma correction")
        parser.add_argument("--contour-contrast", type=float, default=0.75, help="Contour emphasis strength")
        parser.add_argument("--cpu-workers", type=int, default=0, help="CPU burner thread count")
        parser.add_argument("--cpu-matrix", type=int, default=896, help="CPU burner matrix size")
        parser.add_argument("--max-cpu-temp", type=float, default=75.0, help="Hold the show if the CPU exceeds this temperature in Celsius")
        parser.add_argument("--max-gpu-temp", type=float, default=70.0, help="Hold the show if the GPU exceeds this temperature in Celsius")
        parser.add_argument("--thermal-poll-seconds", type=float, default=5.0, help="Sensor poll interval in seconds")
        parser.add_argument("--no-thermal-hold", action="store_true", help="Disable the thermal watchdog and hold screen")


def main() -> None:
    mglw.run_window_config(GarageHeatShow)


if __name__ == "__main__":
    main()
