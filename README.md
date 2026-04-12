# Garage Life Lab (1080p)

The research garage now doubles as a bio-inspired heat chamber. Instead of 8K fractals, this build drives a 1920×1080 panel with a brutal reaction-diffusion (Gray-Scott / Lenia hybrid) life simulation. The RTX 5070 handles the ping-pong GPU solver while the Ryzen 9 7950X runs matrix/FFT burners, soaking the room in both photons and BTUs.

## Highlights
- **Life simulation core** – multi-step Gray-Scott updates per frame with stochastic drift and palette cycling.
- **Purposeful 1080p** – the display is locked to FHD for the garage screen; heat now comes from raw substeps and shader math instead of cheating with resolution.
- **CPU saturation threads** – configurable matmul/FFT burners ensure every CCD on the 7950X contributes to the lab climate.
- **Operational playbook** – runbook + safety tips keep the living art show sustainable.

## Requirements
- Windows 10/11 with the latest NVIDIA driver (RTX 5070).
- Python 3.9+ with `pip` (a venv is recommended).
- Visual C++ build tools or compatible wheels for NumPy/Moderngl.
- A 1920×1080 display/TV mounted to the garage door.

## Setup
```powershell
cd d:\realtimeSimulation
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Quick Start (Lab Baseline)
```powershell
python main.py --width 1920 --height 1080 --substeps 16 --cpu-workers 8 --cpu-matrix 768
```

## Aggressive Heat Session
```powershell
python main.py --width 1920 --height 1080 --substeps 28 --feed 0.025 --kill 0.059 --noise-strength 0.02 --cpu-workers 16 --cpu-matrix 1024 --glow 1.4 --exposure 1.7
```
> **Reminder:** Keep V-Sync disabled globally so the GPU never idles.

## CLI Options
| Flag | Default | Description |
|------|---------|-------------|
| `--width` / `--height` | 1920 / 1080 | Simulation resolution (keep at 1080p for the garage panel). |
| `--feed`, `--kill` | 0.029 / 0.057 | Gray-Scott reaction parameters. |
| `--diff-u`, `--diff-v` | 0.16 / 0.08 | Diffusion multipliers for the U & V fields. |
| `--time-step` | 1.0 | Base integration step size. |
| `--substeps` | 12 | Simulation iterations per rendered frame (higher = more GPU heat). |
| `--laplace-scale` | 1.0 | Kernel multiplier controlling wave speed. |
| `--noise-strength` | 0.015 | Random perturbation amplitude to keep lifeforms churning. |
| `--param-drift` | 0.004 | Sinusoidal modulation of feed/kill for emergent phases. |
| `--anim-speed` | 1.0 | Global time multiplier for both simulation and color drift. |
| `--color-shift-speed` | 0.05 | Palette cycling rate. |
| `--exposure`, `--glow`, `--gamma`, `--contour-contrast` | 1.4 / 1.1 / 1.2 / 0.75 | Display grading controls. |
| `--cpu-workers` | 0 | Number of CPU burner threads. |
| `--cpu-matrix` | 896 | Matrix dimension per CPU worker (bigger = more heat). |

All parameters can be tweaked between runs for different organism vibes or thermal setpoints.

## Thermal & Safety Checklist
1. **Monitor temps** (HWInfo64/MSI Afterburner). Keep GPU < 90 °C, CPU < 85 °C.
2. **Airflow** – cracked garage door, pedestal fan, or ducting ensures fumes/heat escape.
3. **Power budget** – expect 500–650 W draw with high substeps + CPU workers; use a dedicated circuit.
4. **Emergency stop** – `Esc` or closing the window halts GPU+CPU workloads instantly.
5. **Session cadence** – consider 20‑minute blasts with short cool-downs if temps climb.

## Show Ideas
- Describe experiments as "digital petri dishes" swimming across the door panel for lab tours.
- Dial `--param-drift` and `--noise-strength` live to morph the organisms with the music.
- Capture runs via OBS (NVENC) for documentation—just note the slight extra GPU load.

## Troubleshooting
- **Black screen** – confirm the monitor is really 1920×1080 and that Windows scaling is 100%; drop `--substeps` if the GPU throttles.
- **Driver resets** – lower `--substeps`, `--laplace-scale`, or disable some CPU workers.
- **CPU overload** – decrease `--cpu-workers` or `--cpu-matrix` for responsiveness.
- **Stale visuals** – tap `--feed`, `--kill`, or `--noise-strength` to kick the ecosystem into a new phase.

Push responsibly, iterate on the lifeforms, and enjoy the heated lab ambience.
