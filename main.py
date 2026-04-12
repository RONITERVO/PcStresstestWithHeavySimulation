"""Garage Life Lab: 1080p bio-simulation space heater."""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import List

import moderngl
import moderngl_window as mglw
import numpy as np
from moderngl_window import geometry


class GarageHeatShow(mglw.WindowConfig):
    """GPU-heavy reaction-diffusion life simulation with CPU burners."""

    title = "Garage Life Lab - BioHeat"
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

        self.update_program["stateTex"].value = 0
        self.display_program["stateTex"].value = 0

        if (self.args.width, self.args.height) != self.window_size:
            self.wnd.resize(self.args.width, self.args.height)

        self._init_simulation_resources()
        self._sync_static_uniforms()

        self.stop_event = threading.Event()
        self.cpu_threads: List[threading.Thread] = []
        if self.args.cpu_workers > 0:
            self._spin_cpu_workers()

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

    def _seed_field(self) -> None:
        width, height = self.state_textures[0].size
        field = np.zeros((height, width, 4), dtype=np.float32)
        field[..., 0] = 1.0  # U
        rng = np.random.default_rng(2025)
        y_indices, x_indices = np.meshgrid(
            np.arange(height), np.arange(width), indexing="ij"
        )
        blob_count = max(40, (width * height) // 120000)
        for _ in range(blob_count):
            cx = rng.integers(0, width)
            cy = rng.integers(0, height)
            radius = rng.integers(12, 120)
            mask = (x_indices - cx) ** 2 + (y_indices - cy) ** 2 <= radius ** 2
            field[..., 1][mask] = 1.0
            field[..., 0][mask] = 0.0
        field[..., 2] = rng.random((height, width))
        field[..., 3] = rng.random((height, width))
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
        if self.state_textures[0].size != self.wnd.buffer_size:
            self._init_simulation_resources()
            self._sync_static_uniforms()
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
        feed = self.args.feed
        kill = self.args.kill
        self.update_program["feed"].value = feed
        self.update_program["kill"].value = kill
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
        self._init_simulation_resources()
        self._sync_static_uniforms()

    def destroy(self) -> None:
        self.stop_event.set()
        for thread in self.cpu_threads:
            thread.join(timeout=1.0)
        self.cpu_threads.clear()
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


def main() -> None:
    mglw.run_window_config(GarageHeatShow)


if __name__ == "__main__":
    main()
