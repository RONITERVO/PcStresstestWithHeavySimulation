# Garage Life Lab

Garage Life Lab is a long-run tile world for a large garage display. It keeps the machine hot with a full-screen GPU simulation plus optional CPU burners, but the screen now reads as a world map instead of abstract noise: oceans, coasts, forests, deserts, mountains, cities, clouds, contour bands, and a day-night cycle.

## Highlights
- Tile-based world simulation designed to stay readable from across a garage.
- Full-screen GPU workload retained for sustained heat.
- Optional NumPy CPU burners for extra room heat.
- Thermal watchdog with on-screen hold state and persistent logs.

## Requirements
- Windows 10 or 11.
- Python 3.9+.
- NVIDIA driver with `nvidia-smi` on `PATH`.
- LibreHardwareMonitor or OpenHardwareMonitor if you want the CPU temp cutoff to work automatically.

## Setup
```powershell
cd d:\realtimeSimulation
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Recommended Run
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

## Notes
- Smaller `--tile-size` means more visible detail per screen, larger values mean chunkier cells that read better from farther away.
- Keep V-Sync off if the goal is sustained load.
- Restarting reseeds the world and gives you a new long-run map.
