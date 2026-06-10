# Garage Life Lab

Garage Life Lab is a long-run tile world for a large garage display. It keeps the machine hot with a full-screen GPU simulation plus optional CPU burners, but the screen now reads as a world map instead of abstract noise: oceans, coasts, forests, deserts, mountains, cities, clouds, contour bands, a day-night cycle, and an in-frame status overlay.

## Highlights
- Tile-based world simulation designed to stay readable from across a garage.
- In-frame show HUD with resolution, tile grid, worker load, FPS, uptime, temperature limits, and thermal hold state.
- Full-screen GPU workload retained for sustained heat.
- Optional NumPy CPU burners for extra room heat.
- Thermal watchdog with on-screen hold state and persistent logs.

## Requirements
- Windows 10 or 11.
- Python 3.9+.
- NVIDIA driver with `nvidia-smi` on `PATH`.
- LibreHardwareMonitor or OpenHardwareMonitor if you want the CPU temp cutoff to work automatically.

## Start Here

Normal users should launch the Windows app UI, not PowerShell flags:

```powershell
.\Start-GarageLifeLab.cmd
```

The first screen detects the PC, recommends a preset, shows a live preview of the world, and displays the exact launch command before anything starts.

## Presets

- **Safe Start / Low**: older laptops, integrated or small GPUs, first launch checks.
- **Balanced Show / Medium**: mainstream 1080p gaming PCs, the public-release default.
- **Performance Show / High**: strong desktops with real discrete GPUs.
- **Max Heat Lab / Ultra**: RTX 5070/4080/4090-class systems and high-core CPUs with cooling to match.

The app keeps developer-level control under **Advanced controls**, where every generated parameter can still be edited before launch.

Preset design targets the broad Windows gaming-PC middle first: 1080p screens, 16-32 GB RAM, 6-8 CPU cores, and DirectX 12-class GPUs. The launcher auto-selects a tier from GPU class, CPU thread count, RAM, and screen size, then shows the exact generated launch before starting.

| Tier | Typical PC | What changes |
| --- | --- | --- |
| Low | Small GPU, laptop, or first launch | 720p, larger tiles, low CPU workers |
| Medium | Mainstream 1080p desktop | 1080p, balanced detail and heat |
| High | Strong discrete GPU | More GPU steps, smaller tiles, more CPU work |
| Ultra | RTX 5070/4080/4090 class | Aggressive detail, heat, and CPU burn |

## Setup From Source
```powershell
cd D:\Projects\Simulations\realtimeSimulation
py -3.9 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Build Windows App

```powershell
.\build_windows_app.ps1
```

The release executable is written to:

```text
dist\GarageLifeLab\GarageLifeLab.exe
```

`Start-GarageLifeLab.cmd` prefers that built app when it exists and falls back to the source launcher otherwise.

## Direct Run

Advanced users can still run the simulation directly:

```powershell
.\.venv\Scripts\python main.py --width 1920 --height 1080 --tile-size 12 --substeps 24 --cpu-workers 16 --cpu-matrix 1024 --glow 1.3 --exposure 1.45
```

## Lower Detail Test
```powershell
.\.venv\Scripts\python main.py --width 1920 --height 1080 --tile-size 14 --substeps 16 --cpu-workers 8 --cpu-matrix 768
```

## Thermal Hold
- Default limits are `CPU 75C` and `GPU 70C`.
- If a limit is crossed, the app stops the heavy workloads and switches to a full-screen hold message instead of closing.
- The reason is written to `logs/thermal_events.log` and `logs/last_thermal_hold.txt`.
- If the app cannot read a CPU sensor, it shows `CPU SENSOR OFFLINE` in the title bar and still enforces the GPU limit.

## Useful Flags
- `--tile-size 12`
- `--substeps 24`
- `--cpu-workers 16`
- `--cpu-matrix 1024`
- `--max-cpu-temp 75`
- `--max-gpu-temp 70`
- `--thermal-poll-seconds 5`
- `--no-thermal-hold`
- `--no-hud`
- `--hud-scale 1.0`

## Notes
- Smaller `--tile-size` means more visible detail per screen, larger values mean chunkier cells that read better from farther away.
- Keep V-Sync off if the goal is sustained load.
- Restarting reseeds the world and gives you a new long-run map.
