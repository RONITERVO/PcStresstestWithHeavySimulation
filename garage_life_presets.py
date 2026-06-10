"""Hardware detection and launch presets for Garage Life Lab."""
from __future__ import annotations

import ctypes
import os
import platform
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence


CREATE_NO_WINDOW = 0x08000000


@dataclass(frozen=True)
class GpuInfo:
    name: str
    memory_mb: int


@dataclass(frozen=True)
class HardwareInfo:
    cpu_name: str
    logical_cpus: int
    ram_gb: int
    gpu: GpuInfo
    screen_width: int
    screen_height: int
    python_executable: str
    nvidia_smi_available: bool


@dataclass(frozen=True)
class LaunchPreset:
    key: str
    name: str
    short_name: str
    audience: str
    description: str
    width: int
    height: int
    tile_size: int
    substeps: int
    cpu_workers: int
    cpu_matrix: int
    glow: float
    exposure: float
    gamma: float
    contour_contrast: float
    max_gpu_temp: float
    max_cpu_temp: float
    hud_scale: float = 1.0
    no_thermal_hold: bool = False
    no_hud: bool = False

    def args(self) -> list[str]:
        args = [
            "-wnd",
            "glfw",
            "--width",
            str(self.width),
            "--height",
            str(self.height),
            "--tile-size",
            str(self.tile_size),
            "--substeps",
            str(self.substeps),
            "--cpu-workers",
            str(self.cpu_workers),
            "--cpu-matrix",
            str(self.cpu_matrix),
            "--glow",
            f"{self.glow:.2f}",
            "--exposure",
            f"{self.exposure:.2f}",
            "--gamma",
            f"{self.gamma:.2f}",
            "--contour-contrast",
            f"{self.contour_contrast:.2f}",
            "--max-gpu-temp",
            f"{self.max_gpu_temp:.0f}",
            "--max-cpu-temp",
            f"{self.max_cpu_temp:.0f}",
            "--hud-scale",
            f"{self.hud_scale:.2f}",
        ]
        if self.no_thermal_hold:
            args.append("--no-thermal-hold")
        if self.no_hud:
            args.append("--no-hud")
        return args

    def command_preview(self) -> str:
        return " ".join(_quote_arg(arg) for arg in (["main.py"] + self.args()))


def _quote_arg(arg: str) -> str:
    if re.search(r"\s", arg):
        return f'"{arg}"'
    return arg


def run_hidden(command: Sequence[str], timeout: float = 4.0) -> Optional[subprocess.CompletedProcess[str]]:
    try:
        return subprocess.run(
            list(command),
            capture_output=True,
            text=True,
            timeout=timeout,
            creationflags=CREATE_NO_WINDOW if os.name == "nt" else 0,
            check=False,
        )
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        return None


def system_ram_gb() -> int:
    if os.name == "nt":
        class MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatusEx()
        status.dwLength = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return max(1, round(status.ullTotalPhys / (1024 ** 3)))
    return 0


def detect_cpu_name() -> str:
    if os.name == "nt":
        result = run_hidden(
            [
                "powershell",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                "(Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty Name)",
            ],
            timeout=4.0,
        )
        if result and result.returncode == 0:
            name = result.stdout.strip()
            if name:
                return compact_label(name)
    return platform.processor() or "Unknown CPU"


def detect_gpu() -> tuple[GpuInfo, bool]:
    result = run_hidden(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total",
            "--format=csv,noheader,nounits",
        ],
        timeout=4.0,
    )
    if result and result.returncode == 0:
        best = GpuInfo(name="NVIDIA GPU", memory_mb=0)
        for line in result.stdout.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 2:
                continue
            try:
                memory_mb = int(float(parts[1]))
            except ValueError:
                memory_mb = 0
            candidate = GpuInfo(name=compact_label(parts[0]), memory_mb=memory_mb)
            if candidate.memory_mb >= best.memory_mb:
                best = candidate
        return best, True

    if os.name == "nt":
        result = run_hidden(
            [
                "powershell",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                "Get-CimInstance Win32_VideoController | Sort-Object AdapterRAM -Descending | Select-Object -First 1 -ExpandProperty Name",
            ],
            timeout=4.0,
        )
        if result and result.returncode == 0 and result.stdout.strip():
            return GpuInfo(name=compact_label(result.stdout.strip()), memory_mb=0), False

    return GpuInfo(name="Unknown GPU", memory_mb=0), False


def detect_hardware(screen_width: int, screen_height: int, python_executable: str) -> HardwareInfo:
    gpu, nvidia_smi_available = detect_gpu()
    return HardwareInfo(
        cpu_name=detect_cpu_name(),
        logical_cpus=max(1, os.cpu_count() or 1),
        ram_gb=system_ram_gb(),
        gpu=gpu,
        screen_width=max(800, screen_width),
        screen_height=max(600, screen_height),
        python_executable=python_executable,
        nvidia_smi_available=nvidia_smi_available,
    )


def compact_label(value: str, limit: int = 64) -> str:
    text = " ".join(value.replace("(TM)", "").replace("(R)", "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "..."


def screen_size_for_preset(hw: HardwareInfo, width: int, height: int) -> tuple[int, int]:
    safe_width = min(width, hw.screen_width)
    safe_height = min(height, hw.screen_height)
    if safe_width < 1280 or safe_height < 720:
        return max(960, safe_width), max(540, safe_height)
    return safe_width, safe_height


def gpu_score(gpu: GpuInfo) -> int:
    name = gpu.name.upper()
    memory = gpu.memory_mb
    if any(token in name for token in ["RTX 5090", "RTX 5080", "RTX 5070", "RTX 4090", "RTX 4080", "RX 7900"]):
        return 4
    if any(token in name for token in ["RTX 4070", "RTX 3080", "RTX 3090", "RX 7800", "RX 6800", "RX 6900"]):
        return 3
    if any(token in name for token in ["RTX 4060", "RTX 3070", "RTX 3060", "RTX 2070", "RTX 2060", "RX 7600", "RX 6700", "RX 6600"]):
        return 2
    if any(token in name for token in ["GTX 1660", "GTX 1650", "GTX 1060", "RX 580", "RX 570", "RTX 3050"]):
        return 1
    if memory >= 12000:
        return 3
    if memory >= 8000:
        return 2
    if memory >= 4000:
        return 1
    return 0


def recommended_preset_key(hw: HardwareInfo) -> str:
    score = gpu_score(hw.gpu)
    if score >= 4 and hw.logical_cpus >= 16 and hw.ram_gb >= 32:
        return "ultra"
    if score >= 3 and hw.logical_cpus >= 12 and hw.ram_gb >= 16:
        return "high"
    if score >= 2 and hw.logical_cpus >= 8 and hw.ram_gb >= 16:
        return "balanced"
    return "safe"


def build_presets(hw: HardwareInfo) -> list[LaunchPreset]:
    safe_w, safe_h = screen_size_for_preset(hw, 1280, 720)
    full_w, full_h = screen_size_for_preset(hw, 1920, 1080)
    cpu = hw.logical_cpus
    return [
        LaunchPreset(
            key="safe",
            name="Safe Start",
            short_name="Low",
            audience="Older laptops, small GPUs, first launch",
            description="Starts gently, proves the app works, and keeps the machine responsive.",
            width=safe_w,
            height=safe_h,
            tile_size=18,
            substeps=10,
            cpu_workers=max(0, min(4, cpu // 4)),
            cpu_matrix=512,
            glow=1.05,
            exposure=1.25,
            gamma=1.20,
            contour_contrast=0.60,
            max_gpu_temp=72,
            max_cpu_temp=78,
            hud_scale=0.95,
        ),
        LaunchPreset(
            key="balanced",
            name="Balanced Show",
            short_name="Medium",
            audience="Mainstream 1080p gaming PCs",
            description="The default for a broad public release: readable, warm, and hard to misconfigure.",
            width=full_w,
            height=full_h,
            tile_size=14,
            substeps=18,
            cpu_workers=max(2, min(8, cpu // 2)),
            cpu_matrix=768,
            glow=1.20,
            exposure=1.35,
            gamma=1.18,
            contour_contrast=0.72,
            max_gpu_temp=75,
            max_cpu_temp=82,
            hud_scale=1.0,
        ),
        LaunchPreset(
            key="high",
            name="Performance Show",
            short_name="High",
            audience="RTX 3070/4070 class, RX 6800 class, strong desktops",
            description="More detail and heat for machines with strong cooling and a real discrete GPU.",
            width=full_w,
            height=full_h,
            tile_size=10,
            substeps=32,
            cpu_workers=max(4, min(16, (cpu * 2) // 3)),
            cpu_matrix=1024,
            glow=1.32,
            exposure=1.45,
            gamma=1.15,
            contour_contrast=0.82,
            max_gpu_temp=78,
            max_cpu_temp=86,
            hud_scale=1.0,
        ),
        LaunchPreset(
            key="ultra",
            name="Max Heat Lab",
            short_name="Ultra",
            audience="RTX 5070/4080/4090 class and high-core CPUs",
            description="Aggressive stress and display mode. Use when cooling, noise, and power are intentional.",
            width=full_w,
            height=full_h,
            tile_size=8,
            substeps=48,
            cpu_workers=max(8, min(24, (cpu * 3) // 4)),
            cpu_matrix=1536,
            glow=1.40,
            exposure=1.55,
            gamma=1.12,
            contour_contrast=0.90,
            max_gpu_temp=80,
            max_cpu_temp=88,
            hud_scale=1.0,
        ),
    ]


def preset_by_key(presets: Iterable[LaunchPreset], key: str) -> LaunchPreset:
    for preset in presets:
        if preset.key == key:
            return preset
    return next(iter(presets))


def repo_root() -> Path:
    return Path(__file__).resolve().parent
