"""Garage Life Lab: 1080p bio-simulation space heater (3D Raymarched Volumetric Edition)."""
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

from worlds.registry import DEFAULT_WORLD_ID, get_world, world_ids
from worlds.spec import WorldSpec

try:
    import pyaudio

    HAS_AUDIO = True
except ImportError:
    pyaudio = None
    HAS_AUDIO = False

CREATE_NO_WINDOW = 0x08000000
CHUNKS_PER_SEC = 60
SAMPLE_RATE = 44100
CHUNK_SIZE = SAMPLE_RATE // CHUNKS_PER_SEC
FFT_BINS = 128
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

VERT_SHADER = """
#version 450
in vec2 in_position;
out vec2 uv;
out vec2 v_uv;
void main() {
    uv = in_position * 0.5 + 0.5;
    v_uv = uv;
    gl_Position = vec4(in_position, 0.0, 1.0);
}
"""



MSG_FRAG_SHADER = """
#version 450
in vec2 uv;
out vec4 fragColor;
uniform sampler2D displayTex;
void main() {
    fragColor = texture(displayTex, uv);
}
"""

DEFAULT_ARGUMENTS = {
    "width": 1920,
    "height": 1080,
    "feed": 0.035,
    "kill": 0.060,
    "diff_u": 0.16,
    "diff_v": 0.08,
    "time_step": 1.0,
    "substeps": 8,
    "laplace_scale": 1.0,
    "noise_strength": 0.015,
    "param_drift": 0.004,
    "anim_speed": 1.0,
    "color_shift_speed": 0.05,
    "exposure": 1.4,
    "glow": 1.1,
    "gamma": 1.2,
    "contour_contrast": 0.75,
    "ray_steps": 96,
    "fx_intensity": 1.0,
    "camera_speed": 1.0,
    "cpu_workers": 0,
    "cpu_matrix": 896,
    "tile_size": 12,
    "max_cpu_temp": 75.0,
    "max_gpu_temp": 70.0,
    "thermal_poll_seconds": 5.0,
    "hud_scale": 1.0,
    "quit_after": 0.0,
}


def _smoothstep(edge0: float, edge1: float, x: float) -> float:
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return float(t * t * (3.0 - 2.0 * t))


class GenerativeAudioEngine:
    """Small procedural audio source for worlds that expose audio uniforms."""

    def __init__(self) -> None:
        self.active = True
        self.time_value = 0.0
        self.fft_data = np.zeros(FFT_BINS, dtype=np.float32)
        self.wave_data = np.zeros(FFT_BINS, dtype=np.float32)
        self.energy = 0.0
        self.bass = 0.0
        self.treble = 0.0
        self.chords = [
            [32.70, 65.41, 98.00, 155.56, 196.00],
            [27.50, 55.00, 82.41, 138.59, 164.81],
            [21.83, 43.65, 65.41, 103.83, 130.81],
            [36.71, 73.42, 110.00, 174.61, 220.00],
        ]
        self.current_chord_idx = 0
        self.phase = np.zeros(5, dtype=np.float32)
        self.lock = threading.Lock()
        self._pa = None
        self._stream = None
        self.simulated = True
        if HAS_AUDIO:
            self._start_audio_stream()
        else:
            self._start_simulation_thread()

    def _start_audio_stream(self) -> None:
        try:
            self._pa = pyaudio.PyAudio()
            self._stream = self._pa.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=SAMPLE_RATE,
                output=True,
                frames_per_buffer=CHUNK_SIZE,
                stream_callback=self._audio_callback,
            )
            self._stream.start_stream()
            self.simulated = False
        except Exception:
            self._pa = None
            self._stream = None
            self._start_simulation_thread()

    def _start_simulation_thread(self) -> None:
        self.simulated = True
        thread = threading.Thread(target=self._simulate_audio, name="audio-simulator", daemon=True)
        thread.start()

    def _generate_chunk(self) -> np.ndarray:
        dt = CHUNK_SIZE / SAMPLE_RATE
        t_seq = np.linspace(self.time_value, self.time_value + dt, CHUNK_SIZE, endpoint=False)
        self.time_value += dt

        macro_time = self.time_value * 0.15
        next_chord_idx = (self.current_chord_idx + 1) % len(self.chords)
        blend = _smoothstep(0.8, 1.0, macro_time % 1.0)
        if (macro_time % 1.0) < 0.05 and macro_time > self.current_chord_idx + 1:
            self.current_chord_idx = next_chord_idx

        chord_a = np.array(self.chords[self.current_chord_idx], dtype=np.float32)
        chord_b = np.array(self.chords[next_chord_idx], dtype=np.float32)
        freqs = chord_a * (1.0 - blend) + chord_b * blend

        out = np.zeros(CHUNK_SIZE, dtype=np.float32)
        for index, freq in enumerate(freqs):
            fm = np.sin(2.0 * np.pi * (freq * 0.5) * t_seq) * (
                1.2 + 0.8 * np.sin(self.time_value * 0.2 + index)
            )
            phase_inc = 2.0 * np.pi * freq * t_seq + fm
            amp = 0.12 + 0.08 * np.sin(self.time_value * 0.4 + index * 1.6)
            out += np.sin(phase_inc + self.phase[index]) * amp
            self.phase[index] = (self.phase[index] + 2.0 * np.pi * freq * dt) % (2.0 * np.pi)

        kick_env = max(0.0, np.sin(self.time_value * np.pi * 2.0 * 1.5)) ** 24.0
        out += np.sin(2.0 * np.pi * 45.0 * t_seq - kick_env * 15.0) * kick_env * 0.6
        noise_env = max(0.0, np.sin(self.time_value * np.pi * 0.25)) ** 4.0
        out += np.random.normal(0.0, 0.01 + 0.03 * noise_env, CHUNK_SIZE)
        out = np.clip(out, -1.0, 1.0).astype(np.float32)

        with self.lock:
            window = np.hanning(CHUNK_SIZE)
            fft_complex = np.fft.rfft(out * window)
            fft_mag = np.abs(fft_complex[:FFT_BINS]) / (CHUNK_SIZE / 2)
            self.fft_data = self.fft_data * 0.6 + fft_mag.astype(np.float32) * 0.4
            wave_downsampled = out[:: max(1, CHUNK_SIZE // FFT_BINS)][:FFT_BINS]
            self.wave_data = self.wave_data * 0.5 + wave_downsampled * 0.5
            self.energy = float(np.mean(self.fft_data))
            self.bass = float(np.mean(self.fft_data[:8]))
            self.treble = float(np.mean(self.fft_data[64:]))
        return out

    def _audio_callback(self, in_data, frame_count, time_info, status):
        return self._generate_chunk().tobytes(), pyaudio.paContinue

    def _simulate_audio(self) -> None:
        while self.active:
            self._generate_chunk()
            time.sleep(1.0 / CHUNKS_PER_SEC)

    def get_data(self) -> tuple[np.ndarray, np.ndarray, float, float, float]:
        with self.lock:
            return (
                self.fft_data.copy(),
                self.wave_data.copy(),
                self.energy,
                self.bass,
                self.treble,
            )

    def destroy(self) -> None:
        self.active = False
        try:
            if self._stream is not None:
                self._stream.stop_stream()
                self._stream.close()
            if self._pa is not None:
                self._pa.terminate()
        except Exception:
            pass


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
    x: Optional[int] = None,
    align: str = "center",
) -> None:
    sanitized = _sanitize_text(line)
    glyph_width = 3 * scale
    spacing = scale
    line_width = max(0, len(sanitized) * (glyph_width + spacing) - spacing)
    if x is None:
        if align == "right":
            x = canvas.shape[1] - line_width - scale * 3
        elif align == "left":
            x = scale * 3
        else:
            x = (canvas.shape[1] - line_width) // 2
    elif align == "right":
        x -= line_width
    elif align == "center":
        x -= line_width // 2
    x = max(int(x), scale * 2)
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


def _text_width(line: str, scale: int) -> int:
    sanitized = _sanitize_text(line)
    if not sanitized:
        return 0
    return len(sanitized) * (4 * scale) - scale


def _fill_rect(
    canvas: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    color: Sequence[int],
) -> None:
    height, width, _ = canvas.shape
    left = max(0, min(width, int(x0)))
    right = max(0, min(width, int(x1)))
    top = max(0, min(height, int(y0)))
    bottom = max(0, min(height, int(y1)))
    if right > left and bottom > top:
        canvas[top:bottom, left:right] = color


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


def build_hud_frame(
    left_lines: Sequence[str],
    right_lines: Sequence[str],
    size: Sequence[int],
    hud_scale: float,
) -> np.ndarray:
    width = max(int(size[0]), 320)
    height = max(int(size[1]), 180)
    frame = np.zeros((height, width, 4), dtype=np.uint8)
    margin = max(14, min(width, height) // 44)
    gap = max(8, margin // 2)
    base_scale = int(round((min(width, height) / 360.0) * max(0.5, hud_scale)))
    scale = max(2, min(base_scale, 6))
    title_scale = max(scale + 1, 3)
    line_gap = max(3, scale)
    pad_x = scale * 4
    pad_y = scale * 3

    left_title = left_lines[:1]
    left_body = left_lines[1:]
    left_width = max(
        [_text_width(line, title_scale) for line in left_title]
        + [_text_width(line, scale) for line in left_body]
        + [scale * 32]
    )
    right_width = max([_text_width(line, scale) for line in right_lines] + [scale * 34])
    left_panel_width = min(width - margin * 2, left_width + pad_x * 2)
    right_panel_width = min(width - margin * 2, right_width + pad_x * 2)
    left_panel_height = (
        pad_y * 2
        + (5 * title_scale if left_title else 0)
        + (line_gap * 2 if left_title and left_body else 0)
        + len(left_body) * (5 * scale + line_gap)
    )
    left_panel_height = max(left_panel_height, scale * 20)
    right_panel_height = pad_y * 2 + len(right_lines) * (5 * scale + line_gap)
    right_panel_height = max(right_panel_height, scale * 20)

    left_x = margin
    left_y = margin
    right_x = width - margin - right_panel_width
    right_y = margin
    if right_x < left_x + left_panel_width + gap:
        right_x = margin
        right_y = margin + left_panel_height + gap

    panel_color = (6, 10, 14, 172)
    line_color = (36, 214, 192, 210)
    title_color = (238, 252, 247, 245)
    body_color = (190, 222, 224, 230)
    warn_color = (255, 186, 87, 238)

    _fill_rect(frame, left_x, left_y, left_x + left_panel_width, left_y + left_panel_height, panel_color)
    _fill_rect(frame, left_x, left_y, left_x + scale, left_y + left_panel_height, line_color)
    cursor_y = left_y + pad_y
    if left_title:
        _draw_text_line(
            frame,
            left_title[0],
            cursor_y,
            title_scale,
            title_color,
            x=left_x + pad_x,
            align="left",
        )
        cursor_y += 5 * title_scale + line_gap * 2
    for line in left_body:
        _draw_text_line(
            frame,
            line,
            cursor_y,
            scale,
            body_color,
            x=left_x + pad_x,
            align="left",
        )
        cursor_y += 5 * scale + line_gap

    _fill_rect(frame, right_x, right_y, right_x + right_panel_width, right_y + right_panel_height, panel_color)
    _fill_rect(frame, right_x, right_y, right_x + scale, right_y + right_panel_height, line_color)
    cursor_y = right_y + pad_y
    for line in right_lines:
        color = warn_color if "OFFLINE" in line or "OFF" in line else body_color
        _draw_text_line(
            frame,
            line,
            cursor_y,
            scale,
            color,
            x=right_x + pad_x,
            align="left",
        )
        cursor_y += 5 * scale + line_gap

    return frame


class GarageHeatShow(mglw.WindowConfig):
    """GPU-heavy 3D volumetric world simulation with CPU burners and thermal hold."""

    title = "Garage Life Lab - 3D Bio-World"
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

        self.world = get_world(getattr(self.args, "world", DEFAULT_WORLD_ID))
        self._apply_world_defaults(self.world)
        self.base_title = self.world.window_title
        self.wnd.title = self.base_title
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
        self.hud_texture = None
        self.next_hud_refresh_at = 0.0
        self.started_at = time.monotonic()
        self.fps_estimate = 0.0
        self.offscreen_texture = None
        self.offscreen_framebuffer = None
        self.cpu_threads: List[threading.Thread] = []
        self.audio: Optional[GenerativeAudioEngine] = None
        self.audio_fft_tex = None
        self.audio_wave_tex = None

        self.ctx.disable(moderngl.DEPTH_TEST)
        self.quad = geometry.quad_fs()

        self.update_program = self.ctx.program(
            vertex_shader=VERT_SHADER,
            fragment_shader=self.world.sim_shader,
        )
        self.display_program = self.ctx.program(
            vertex_shader=VERT_SHADER,
            fragment_shader=self.world.display_shader,
        )
        self.message_program = self.ctx.program(
            vertex_shader=VERT_SHADER,
            fragment_shader=MSG_FRAG_SHADER,
        )

        self.update_program["stateTex"].value = 0
        self.display_program["stateTex"].value = 0
        self.message_program["displayTex"].value = 0
        if self.world.uses_audio:
            self._init_audio_resources()

        if (self.args.width, self.args.height) != self.window_size:
            self.wnd.resize(self.args.width, self.args.height)

        self._init_simulation_resources()
        self._sync_static_uniforms()

        if self.args.cpu_workers > 0:
            self._spin_cpu_workers()
        if not self.args.no_thermal_hold:
            self._spin_thermal_watchdog()

    def _init_audio_resources(self) -> None:
        self.audio = GenerativeAudioEngine()
        self.audio_fft_tex = self.ctx.texture((FFT_BINS, 1), 1, dtype="f4")
        self.audio_fft_tex.filter = (moderngl.LINEAR, moderngl.NEAREST)
        self.audio_wave_tex = self.ctx.texture((FFT_BINS, 1), 1, dtype="f4")
        self.audio_wave_tex.filter = (moderngl.LINEAR, moderngl.NEAREST)
        self._set_uniform(self.update_program, "audioFft", 1)
        self._set_uniform(self.display_program, "audioFft", 1)
        self._set_uniform(self.display_program, "audioWave", 2)

    def _refresh_audio_textures(self):
        if self.audio is None or self.audio_fft_tex is None or self.audio_wave_tex is None:
            return None
        fft_data, wave_data, energy, bass, treble = self.audio.get_data()
        self.audio_fft_tex.write(fft_data.tobytes())
        self.audio_wave_tex.write(wave_data.tobytes())
        return energy, bass, treble

    def _apply_world_defaults(self, world: WorldSpec) -> None:
        defaults = dict(DEFAULT_ARGUMENTS)
        defaults.update(world.default_overrides)
        for key, value in defaults.items():
            if getattr(self.args, key, None) is None:
                setattr(self.args, key, value)

    def _uniform(self, program, name: str):
        try:
            return program[name]
        except KeyError:
            return None

    def _set_uniform(self, program, name: str, value) -> None:
        uniform = self._uniform(program, name)
        if uniform is not None:
            uniform.value = value

    def _write_uniform(self, program, name: str, value) -> None:
        uniform = self._uniform(program, name)
        if uniform is not None:
            uniform.write(value)

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
            data=np.flipud(frame).tobytes(),
            alignment=1,
        )
        self.hold_texture.filter = (moderngl.NEAREST, moderngl.NEAREST)

    def _display_target(self):
        if self.ctx.screen is not None:
            return self.ctx.screen
        if (
            self.offscreen_texture is None
            or self.offscreen_framebuffer is None
            or self.offscreen_texture.size != self.wnd.buffer_size
        ):
            if self.offscreen_framebuffer is not None:
                self.offscreen_framebuffer.release()
            if self.offscreen_texture is not None:
                self.offscreen_texture.release()
            self.offscreen_texture = self.ctx.texture(self.wnd.buffer_size, 4)
            self.offscreen_framebuffer = self.ctx.framebuffer(
                color_attachments=[self.offscreen_texture]
            )
        return self.offscreen_framebuffer

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

    def _audio_status_line(self) -> Optional[str]:
        if not self.world.uses_audio or self.audio is None:
            return None
        return "AUDIO SIMULATED" if self.audio.simulated else "AUDIO ONLINE"

    def _format_uptime(self) -> str:
        total_seconds = max(0, int(time.monotonic() - self.started_at))
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        return f"UP {hours:02d}:{minutes:02d}:{seconds:02d}"

    def _temperature_line(self, label: str, value: Optional[float], limit: float) -> str:
        if limit <= 0:
            return f"{label} HOLD OFF"
        if value is None:
            return f"{label} SENSOR OFFLINE"
        return f"{label} {value:.0f}C LIMIT {limit:.0f}C"

    def _hud_lines(self) -> Sequence[Sequence[str]]:
        width, height = self.wnd.buffer_size
        tile_size = max(2, int(self.args.tile_size))
        tiles_x = max(1, int(np.ceil(width / tile_size)))
        tiles_y = max(1, int(np.ceil(height / tile_size)))
        with self.telemetry_lock:
            gpu_temp = self.latest_temperatures.get("GPU")
            cpu_temp = self.latest_temperatures.get("CPU")
        if self._uniform(self.display_program, "raySteps") is not None:
            sim_line = f"SIM STEP {self.args.substeps} RAYMARCH {self.args.ray_steps}"
        else:
            sim_line = f"GRID {tiles_x}X{tiles_y} STEP {self.args.substeps}"
        left_lines = [
            "GARAGE LIFE LAB",
            self.world.hud_subtitle,
            f"{width}X{height} MAP {tiles_x}X{tiles_y}",
            sim_line,
            f"FX {self.args.fx_intensity:.1f} CAM {self.args.camera_speed:.1f}",
            f"CPU WORKERS {self.args.cpu_workers}",
        ]
        right_lines = [
            self._temperature_line("GPU", gpu_temp, self.args.max_gpu_temp),
            self._temperature_line("CPU", cpu_temp, self.args.max_cpu_temp),
            f"FPS {self.fps_estimate:.0f}" if self.fps_estimate > 0 else "FPS --",
        ]
        audio_status = self._audio_status_line()
        if audio_status is not None:
            right_lines.append(audio_status)
        right_lines.extend([
            self._format_uptime(),
            "THERMAL HOLD OFF" if self.args.no_thermal_hold else "THERMAL HOLD ARMED",
        ])
        return left_lines, right_lines

    def _build_hud_texture(self) -> None:
        if self.args.no_hud:
            return
        now = time.monotonic()
        if (
            self.hud_texture is not None
            and self.hud_texture.size == self.wnd.buffer_size
            and now < self.next_hud_refresh_at
        ):
            return
        left_lines, right_lines = self._hud_lines()
        frame = build_hud_frame(left_lines, right_lines, self.wnd.buffer_size, self.args.hud_scale)
        data = np.flipud(frame).tobytes()
        if self.hud_texture is None or self.hud_texture.size != self.wnd.buffer_size:
            if self.hud_texture is not None:
                self.hud_texture.release()
            self.hud_texture = self.ctx.texture(
                self.wnd.buffer_size,
                4,
                data=data,
                alignment=1,
            )
            self.hud_texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
        else:
            self.hud_texture.write(data)
        self.next_hud_refresh_at = now + 0.5

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
        self._write_uniform(self.update_program, "resolution", resolution)
        self._write_uniform(self.display_program, "resolution", resolution)
        tile_size = float(max(2, self.args.tile_size))
        self._set_uniform(self.update_program, "tileSize", tile_size)
        self._set_uniform(self.display_program, "tileSize", tile_size)

    def _seed_field(self) -> None:
        width_px, height_px = self.state_textures[0].size
        tile_size = max(2, int(self.args.tile_size))
        field = self.world.seed_field(width_px, height_px, tile_size)
        expected_shape = (height_px, width_px, 4)
        if field.shape != expected_shape:
            raise ValueError(
                f"World {self.world.id} seed field shape {field.shape} != {expected_shape}"
            )
        field = np.ascontiguousarray(field.astype(np.float32, copy=False))
        for tex in self.state_textures:
            tex.write(field.tobytes())

    def _sync_static_uniforms(self) -> None:
        self._set_uniform(self.update_program, "diffU", self.args.diff_u)
        self._set_uniform(self.update_program, "diffV", self.args.diff_v)
        self._set_uniform(self.update_program, "dt", self.args.time_step)
        self._set_uniform(self.update_program, "laplaceScale", self.args.laplace_scale)
        self._set_uniform(self.update_program, "noiseStrength", self.args.noise_strength)
        self._set_uniform(self.update_program, "parameterDrift", self.args.param_drift)

        self._set_uniform(self.display_program, "exposure", self.args.exposure)
        self._set_uniform(self.display_program, "glow", self.args.glow)
        self._set_uniform(self.display_program, "gamma", self.args.gamma)
        self._set_uniform(self.display_program, "contourContrast", self.args.contour_contrast)
        self._set_uniform(self.display_program, "cameraSpeed", self.args.camera_speed)
        self._set_uniform(self.display_program, "fxIntensity", self.args.fx_intensity)
        self._set_uniform(
            self.display_program,
            "raySteps",
            int(max(32, min(160, self.args.ray_steps))),
        )

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
        if self.args.quit_after > 0 and time.monotonic() - self.started_at >= self.args.quit_after:
            self.wnd.close()
            return

        if frame_time > 0:
            current_fps = 1.0 / frame_time
            if self.fps_estimate <= 0:
                self.fps_estimate = current_fps
            else:
                self.fps_estimate = self.fps_estimate * 0.92 + current_fps * 0.08

        if self.thermal_hold is not None:
            if self.hold_texture is None or self.hold_texture.size != self.wnd.buffer_size:
                self._build_hold_texture(self.thermal_hold.lines)
            self._display_target().use()
            self.ctx.viewport = (0, 0, *self.wnd.buffer_size)
            self.hold_texture.use(location=0)
            self.quad.render(self.message_program)
            return

        if self.state_textures[0].size != self.wnd.buffer_size:
            self._init_simulation_resources()
            self._sync_static_uniforms()

        self._refresh_window_title()
        audio_values = self._refresh_audio_textures()

        animated_time = time_value * self.args.anim_speed
        substeps = max(1, self.args.substeps)
        for _ in range(substeps):
            self._step_simulation(animated_time, audio_values)
        self._render_display(animated_time, audio_values)

    def _step_simulation(self, animated_time: float, audio_values) -> None:
        current = self.state_textures[self.active_state]
        next_index = 1 - self.active_state
        target_fbo = self.framebuffers[next_index]
        target_fbo.use()
        self.ctx.viewport = (0, 0, *current.size)
        current.use(location=0)
        if audio_values is not None and self.audio_fft_tex is not None:
            energy, bass, treble = audio_values
            self.audio_fft_tex.use(location=1)
            self._set_uniform(self.update_program, "audioEnergy", energy)
            self._set_uniform(self.update_program, "audioBass", bass)
            self._set_uniform(self.update_program, "audioTreble", treble)
        self._set_uniform(self.update_program, "time", animated_time)
        self._set_uniform(self.update_program, "feed", self.args.feed)
        self._set_uniform(self.update_program, "kill", self.args.kill)
        self.quad.render(self.update_program)
        self.active_state = next_index

    def _render_display(self, animated_time: float, audio_values) -> None:
        self._display_target().use()
        self.ctx.viewport = (0, 0, *self.wnd.buffer_size)
        self.state_textures[self.active_state].use(location=0)
        if audio_values is not None and self.audio_fft_tex is not None and self.audio_wave_tex is not None:
            energy, bass, treble = audio_values
            self.audio_fft_tex.use(location=1)
            self.audio_wave_tex.use(location=2)
            self._set_uniform(self.display_program, "audioEnergy", energy)
            self._set_uniform(self.display_program, "audioBass", bass)
            self._set_uniform(self.display_program, "audioTreble", treble)
        self._set_uniform(self.display_program, "time", animated_time)
        self._set_uniform(self.display_program, "colorShift", (
            animated_time * self.args.color_shift_speed
        ) % 10.0)
        self.quad.render(self.display_program)
        self._render_hud()

    def _render_hud(self) -> None:
        if self.args.no_hud:
            return
        self._build_hud_texture()
        if self.hud_texture is None:
            return
        self._display_target().use()
        self.ctx.viewport = (0, 0, *self.wnd.buffer_size)
        self.hud_texture.use(location=0)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        self.quad.render(self.message_program)
        self.ctx.disable(moderngl.BLEND)

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
        if self.audio is not None:
            self.audio.destroy()
            self.audio = None
        if self.audio_fft_tex is not None:
            self.audio_fft_tex.release()
            self.audio_fft_tex = None
        if self.audio_wave_tex is not None:
            self.audio_wave_tex.release()
            self.audio_wave_tex = None
        if self.hold_texture is not None:
            self.hold_texture.release()
            self.hold_texture = None
        if self.hud_texture is not None:
            self.hud_texture.release()
            self.hud_texture = None
        if self.offscreen_framebuffer is not None:
            self.offscreen_framebuffer.release()
            self.offscreen_framebuffer = None
        if self.offscreen_texture is not None:
            self.offscreen_texture.release()
            self.offscreen_texture = None
        super().destroy()

    @classmethod
    def add_arguments(cls, parser) -> None:  # type: ignore[override]
        parser.add_argument("--world", default=DEFAULT_WORLD_ID, choices=world_ids(), help="World/style to render")
        parser.add_argument("--width", type=int, default=None, help="Render width")
        parser.add_argument("--height", type=int, default=None, help="Render height")
        parser.add_argument("--feed", type=float, default=None, help="Gray-Scott base feed rate")
        parser.add_argument("--kill", type=float, default=None, help="Gray-Scott base kill rate")
        parser.add_argument("--diff-u", type=float, default=None, help="Diffusion rate for U")
        parser.add_argument("--diff-v", type=float, default=None, help="Diffusion rate for V")
        parser.add_argument("--time-step", dest="time_step", type=float, default=None, help="Simulation time step")
        parser.add_argument("--substeps", type=int, default=None, help="Simulation steps per frame")
        parser.add_argument("--laplace-scale", type=float, default=None, help="Global laplacian multiplier")
        parser.add_argument("--noise-strength", type=float, default=None, help="Stochastic noise injected each step")
        parser.add_argument("--param-drift", type=float, default=None, help="Sinusoidal feed/kill drift amplitude")
        parser.add_argument("--anim-speed", type=float, default=None, help="Global animation multiplier")
        parser.add_argument("--color-shift-speed", type=float, default=None, help="Palette cycle speed")
        parser.add_argument("--exposure", type=float, default=None, help="Display exposure")
        parser.add_argument("--glow", type=float, default=None, help="Display glow factor")
        parser.add_argument("--gamma", type=float, default=None, help="Display gamma correction")
        parser.add_argument("--contour-contrast", type=float, default=None, help="Contour emphasis strength")
        parser.add_argument("--ray-steps", type=int, default=None, help="Maximum raymarch steps per pixel")
        parser.add_argument("--fx-intensity", type=float, default=None, help="Cinematic glow, aurora, terrain, and material intensity")
        parser.add_argument("--camera-speed", type=float, default=None, help="Cinematic camera speed multiplier")
        parser.add_argument("--cpu-workers", type=int, default=None, help="CPU burner thread count")
        parser.add_argument("--cpu-matrix", type=int, default=None, help="CPU burner matrix size")
        parser.add_argument("--tile-size", type=int, default=None, help="Base resolution downscale factor for fluid sim")
        parser.add_argument("--max-cpu-temp", type=float, default=None, help="Hold the show if the CPU exceeds this temperature in Celsius")
        parser.add_argument("--max-gpu-temp", type=float, default=None, help="Hold the show if the GPU exceeds this temperature in Celsius")
        parser.add_argument("--thermal-poll-seconds", type=float, default=None, help="Sensor poll interval in seconds")
        parser.add_argument("--no-thermal-hold", action="store_true", help="Disable the thermal watchdog and hold screen")
        parser.add_argument("--no-hud", action="store_true", help="Hide the in-frame show status overlay")
        parser.add_argument("--hud-scale", type=float, default=None, help="Scale the in-frame show status overlay")
        parser.add_argument("--quit-after", type=float, default=None, help="Exit after this many seconds; intended for smoke tests")


def main() -> None:
    mglw.run_window_config(GarageHeatShow)


if __name__ == "__main__":
    main()
