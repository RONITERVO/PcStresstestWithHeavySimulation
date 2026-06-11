# Garage Life Lab

Garage Life Lab is a long-run selectable-world show for a large garage display. It keeps the machine hot with a full-screen GPU simulation plus optional CPU burners, while the launcher separates the world you want to see from how hard the PC should work.

## Highlights
- Selectable worlds: Audio Reactive 3D, Sahara Sandstorm, Tsunami Land, Muddy Asteroid Planet, Neural Plane, Original 3D, Original Tuned 3D, and Original 2D.
- 3D raymarched worlds and a legacy 2D tile world designed to stay readable from across a garage.
- In-frame show HUD with resolution, tile grid, worker load, FPS, uptime, temperature limits, and thermal hold state.
- Full-screen GPU workload retained for sustained heat.
- Live camera steering in 3D worlds with WASD, arrow keys, mouse drag, and wheel zoom.
- Optional NumPy CPU burners for extra room heat.
- Thermal watchdog with on-screen hold state and persistent logs.

## Requirements
- Windows 10 or 11.
- Python 3.9+.
- NVIDIA driver with `nvidia-smi` on `PATH`.
- LibreHardwareMonitor or OpenHardwareMonitor if you want the CPU temp cutoff to work automatically.
- Optional: `pyaudio` for real audio output in Audio Reactive 3D. Without it, the world uses simulated audio data for visuals.

## Start Here

Normal users should launch the Windows app UI, not PowerShell flags:

```powershell
.\Start-GarageLifeLab.cmd
```

The first screen detects the PC, lets you choose a world, recommends a PC load preset, shows a preview thumbnail, and displays the exact launch command before anything starts.

## Worlds

World selection is separate from PC load:

- **Audio Reactive 3D**: bio-world driven by generated audio FFT/wave data, with simulated visual input when PyAudio is unavailable.
- **Sahara Sandstorm**: desert dune world with sandstorm visuals.
- **Tsunami Land**: flooded terrain experiment.
- **Muddy Asteroid Planet**: muddy planetary surface experiment.
- **Neural Plane**: neural/matrix containment experiment.
- **Original 3D**: legacy 3D raymarched bio-world.
- **Original Tuned 3D**: current tuned default.
- **Original 2D**: legacy tile-world shader.

List world IDs from the command line:

```powershell
.\.venv\Scripts\python main.py --list-worlds
```

## Presets

- **Safe Start / Low**: older laptops, integrated or small GPUs, first launch checks.
- **Balanced Show / Medium**: mainstream 1080p gaming PCs, the public-release default.
- **Performance Show / High**: strong desktops with real discrete GPUs.
- **Max Heat Lab / Ultra**: RTX 5070/4080/4090-class systems and high-core CPUs with cooling to match.

The app keeps developer-level control under **Advanced controls**, where every generated parameter can still be edited before launch.

Preset design targets the broad Windows gaming-PC middle first: 1080p screens, 16-32 GB RAM, 6-8 CPU cores, and DirectX 12-class GPUs. The launcher auto-selects a tier from GPU class, CPU thread count, RAM, and screen size, then shows the exact generated launch before starting. Higher tiers raise simulation steps, raymarch steps, cinematic FX intensity, and CPU burner load.

## Live Camera Controls

3D worlds keep their cinematic path, but you can steer on top of it while the simulation runs:

- `W` / `S`: move forward and backward.
- `A` / `D`: strafe left and right.
- `Q` / `E` or `Page Down` / `Page Up`: move down and up.
- Arrow keys: look left, right, up, and down.
- Mouse drag: steer the camera.
- Mouse wheel or `+` / `-`: zoom in and out.
- `Shift`: faster movement; `Ctrl`: slower movement.
- `M`: toggle captured mouse-look mode; `R`: reset camera offset, look, and zoom.

| Tier | Typical PC | What changes |
| --- | --- | --- |
| Low | Small GPU, laptop, or first launch | 720p, 48 ray steps, low CPU workers |
| Medium | Mainstream 1080p desktop | 1080p, 72 ray steps, balanced cinematic FX |
| High | Strong discrete GPU | More sim steps, 104 ray steps, stronger glow |
| Ultra | RTX 5070/4080/4090 class | 140 ray steps, max FX, aggressive heat |

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
.\.venv\Scripts\python main.py --world static-sandstorm-3d --width 1920 --height 1080 --tile-size 10 --substeps 32 --ray-steps 104 --fx-intensity 1.1 --cpu-workers 16 --cpu-matrix 1024 --glow 1.32 --exposure 1.45
```

## Lower Detail Test
```powershell
.\.venv\Scripts\python main.py --width 1920 --height 1080 --tile-size 14 --substeps 16 --cpu-workers 8 --cpu-matrix 768
```

## Smoke Worlds

Run every registered world at low settings and fail on startup, shader, or compiler errors:

```powershell
.\scripts\smoke_worlds.ps1
```

## Thermal Hold
- Default limits are `CPU 75C` and `GPU 70C`.
- If a limit is crossed, the app stops the heavy workloads and switches to a full-screen hold message instead of closing.
- The reason is written to `logs/thermal_events.log` and `logs/last_thermal_hold.txt`.
- If the app cannot read a CPU sensor, it shows `CPU SENSOR OFFLINE` in the title bar and still enforces the GPU limit.

## Useful Flags
- `--tile-size 12`
- `--world static-sandstorm-3d`
- `--substeps 24`
- `--ray-steps 104`
- `--fx-intensity 1.0`
- `--camera-speed 1.0`
- `--camera-move-speed 8.0`
- `--camera-look-speed 1.0`
- `--camera-zoom-speed 0.12`
- `--no-camera-controls`
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
- Historical full-file snapshots live in `docs\world-snapshots`; the runnable app uses the shared engine plus `worlds\*.py`.
