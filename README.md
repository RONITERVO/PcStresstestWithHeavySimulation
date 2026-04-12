# Garage Life Lab

Garage Life Lab turns a 1080p display into a heavy biosphere simulation that can warm a room while staying readable from across a garage door. The RTX 5070 drives the simulation and world rendering, while optional NumPy workers keep the Ryzen 9 busy on the CPU side.

## Highlights
- Biosphere render with coastlines, biomes, clouds, contour lines, and day-night lighting.
- GPU-heavy reaction-diffusion core with extra climate memory for larger world structure.
- Optional CPU burner threads for additional heat.
- Thermal watchdog that stops the workloads, keeps the window open, and logs why it tripped.

## Requirements
- Windows 10 or 11.
- Python 3.9+.
- NVIDIA driver with `nvidia-smi` available on `PATH`.
- For CPU temperature cutoff: LibreHardwareMonitor or OpenHardwareMonitor exposing WMI sensors.

## Setup
```powershell
cd d:\realtimeSimulation
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Quick Start
```powershell
.\.venv\Scripts\python main.py --width 1920 --height 1080 --substeps 16 --cpu-workers 8 --cpu-matrix 768
```

## Aggressive Heat Session
```powershell
.\.venv\Scripts\python main.py --width 1920 --height 1080 --substeps 28 --feed 0.025 --kill 0.059 --noise-strength 0.02 --cpu-workers 16 --cpu-matrix 1024 --glow 1.4 --exposure 1.7
```

## Thermal Hold
- Default hold thresholds are `CPU 75C` and `GPU 70C`.
- If the GPU goes over limit, or the GPU sensor disappears repeatedly, the app stops the loads and switches to a full-screen hold message.
- The hold screen stays visible instead of silently closing, so you can see what happened later.
- Events are written to `logs/thermal_events.log` and `logs/last_thermal_hold.txt`.
- If no CPU sensor source is available, the app will show `CPU SENSOR OFFLINE` in the title bar and only enforce the GPU limit.

## Useful Flags
- `--max-cpu-temp 75`
- `--max-gpu-temp 70`
- `--thermal-poll-seconds 5`
- `--cpu-workers 16`
- `--cpu-matrix 1024`
- `--substeps 28`
- `--no-thermal-hold`

## Notes
- Keep V-Sync disabled if you want sustained load.
- This is still a deliberate stress workload. Use sensible airflow and power delivery.
- If visuals ever get too static, reseeding by restarting the app will generate a different world.
