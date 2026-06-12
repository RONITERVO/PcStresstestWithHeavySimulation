# Minecraft Candidate Validation

Use this lightweight path before changing the default world or doing any long heat runs. It is intended for Windows plus ModernGL in an interactive desktop session.
Run commands from the repository root unless a script path is shown; `scripts\smoke_worlds.ps1` restores your original directory when it exits.

## Default Decision

Keep `minecraft-long-term-3d` as the default launcher world for now. `minecraft-perfect-ecosystem-3d` has focused low-smoke, visual-capture, fixed-camera dynamics, and bounded heavier local evidence, but it remains a candidate until a current full Minecraft comparison run and a candidate-specific launcher preview asset are available.

## Evidence Artifacts

`logs\validation` screenshots and logs are local run outputs and are intentionally ignored by git. Treat paths in this document as provenance for this workstation; reviewers on a fresh checkout should regenerate them with the commands below or request the artifacts separately. The tracked evidence is the command/result summary recorded here.

## Current Candidate Evidence

2026-06-12 step 5 rendering, lighting, and stress pass for `minecraft-perfect-ecosystem-3d`:

- Renderer tuning added a high-FX stress path while preserving low-tier safety: water uses block-snapped ripple normals, cloud terrain shadows darken sun lighting, the primary march takes slightly smaller steps only at high `fxIntensity`, volumetric glow now scales down at low `fxIntensity`, and water can run a bounded 18-step scene reflection only when `fxIntensity > 1.04` and effective `ray_steps >= 104`.
- Current direct CLI default overrides are tuned for a heavy but bounded flagship candidate: `ray_steps=152`, `fx_intensity=1.28`, `contour_contrast=1.14`, `camera_speed=0.72`, `tile_size=8`, `substeps=10`, `glow=1.22`, `exposure=1.28`, and `gamma=1.16`.
- Stress-scaling note: in the current engine, `tile_size` changes the seeded ecosystem grid and block/detail scale, but the state textures still run at full window resolution. Low/Medium safety therefore comes primarily from resolution, `substeps`, `ray_steps`, `fx_intensity`, and CPU burner settings; do not treat larger `tile_size` alone as a thermal safety control.
- Launcher preset note: the launcher appends explicit tier arguments after `--world`, so Low/Medium/High/Ultra preset values intentionally override most per-world defaults. On the local validation machine, detected as AMD Ryzen 9 7950X, 32 logical CPUs, 63 GB RAM, NVIDIA GeForce RTX 5070 with 12227 MB VRAM and `nvidia-smi=True`, the effective candidate tiers were:
  - Low/Safe: `1280x720`, `tile_size=18`, `substeps=10`, `cpu_workers=4`, `cpu_matrix=512`, `ray_steps=48`, `fx_intensity=0.65`, `max_gpu_temp=72`, `max_cpu_temp=78`.
  - Medium/Balanced: `1920x1080`, `tile_size=14`, `substeps=18`, `cpu_workers=8`, `cpu_matrix=768`, `ray_steps=72`, `fx_intensity=0.85`, `max_gpu_temp=75`, `max_cpu_temp=82`.
  - High/Performance: `1920x1080`, `tile_size=10`, `substeps=32`, `cpu_workers=16`, `cpu_matrix=1024`, `ray_steps=104`, `fx_intensity=1.10`, `max_gpu_temp=78`, `max_cpu_temp=86`.
  - Ultra/Max Heat: `1920x1080`, `tile_size=8`, `substeps=48`, `cpu_workers=24`, `cpu_matrix=1536`, `ray_steps=140`, `fx_intensity=1.30`, `max_gpu_temp=80`, `max_cpu_temp=88`.
- Python syntax/registry preflight passed with `.\.venv\Scripts\python.exe -m py_compile worlds\minecraft_perfect_ecosystem_3d.py`.
- Focused low GLFW smoke passed after the renderer changes: `logs\validation\minecraft-perfect-ecosystem-step5-low-smoke.md`. Exact command: `.\scripts\smoke_worlds.ps1 -World minecraft-perfect-ecosystem-3d -Width 640 -Height 360 -TileSize 24 -Substeps 1 -RaySteps 32 -FxIntensity 0.3 -CpuWorkers 0 -QuitAfter 3 -TimeoutSeconds 30 -ReportPath logs\validation\minecraft-perfect-ecosystem-step5-low-smoke.md`. Result: `PASS`, exit code `0`, 3.58 seconds, thermal hold enabled with GPU 70C / CPU 75C limits.
- Bounded heavier local run passed after the renderer changes. Exact command: `.\.venv\Scripts\python.exe -B main.py --world minecraft-perfect-ecosystem-3d -wnd glfw --width 1280 --height 720 --tile-size 8 --substeps 24 --ray-steps 160 --fx-intensity 1.45 --cpu-workers 4 --cpu-matrix 768 --max-gpu-temp 80 --max-cpu-temp 86 --thermal-poll-seconds 1 --no-hud --quit-after 10`. Result: exit code `0`, 10.61 seconds wall time, `logs\validation\minecraft-perfect-ecosystem-step5-heavy-20260612-184715.stderr.log` reports ModernGL 5.8.2 on NVIDIA GeForce RTX 5070 / driver 591.86 and `Duration: 9.69s @ 42.53 FPS`. `logs\thermal_events.log` was not updated during the run.

2026-06-12 step 4 ecosystem pass for `minecraft-perfect-ecosystem-3d`:

- Seed shape/range check passed at 640x360, tile 8: `(360, 640, 4)`, `float32`, channel range `0.0..1.0`.
- Seed coverage after terrain-aware drainage and biome updates: water/moisture `R > 0.55` about 29.1%, visible vegetation `G > 0.35` about 12.9%, dense vegetation `G > 0.50` about 4.7%, mountains `B > 0.60` about 23.2%, light/settlement/ore/fire `A > 0.45` about 1.9%.
- Focused GLFW smoke passed: `logs\validation\minecraft-perfect-ecosystem-step4-smoke.md`.
- Window-capture run at the candidate's tuned default workload produced `logs\validation\screenshots\minecraft-perfect-ecosystem-window-20260612-183202.png`, with process evidence in the neighboring stdout/stderr logs. The frame is nonblank and not clipped: mean RGB about `[148.47, 159.40, 142.48]`, bright fraction about 3.3%, saturated-color fraction about 48.7%.
- Visual review of that capture shows connected block terrain with a visible water basin/lowland channel, shore/wetland color transitions, readable block seams, trees/shrubs/reeds, cave-like dark openings, lit settlement/torch/ore cues, paths/clearings, and distinct highland/lowland material bands.
- A paired fixed-camera capture produced `logs\validation\screenshots\minecraft-perfect-ecosystem-fixed-20260612-183557-t3.png` and `logs\validation\screenshots\minecraft-perfect-ecosystem-fixed-20260612-183557-t9.png`. The paired run reported about 57.80 FPS on the local RTX 5070; mean absolute image difference was about 57.44 with about 88.7% of pixels changing above an 8-bit threshold of 16 over the six-second interval, confirming visible world dynamics while the camera was commanded to stay still.
- Growth/decay feedback is implemented in the simulation shader through suitability-based biomass growth, wetland and seed-bank recovery, flood/alpine/fire damage, canopy moisture retention, root-reduced erosion, soil return, bounded fire spread, and capped settlement/ore/fire light reinforcement.

## 1. List Candidate Metadata

```powershell
.\.venv\Scripts\python.exe scripts\list_world_metadata.py --minecraft-only
.\.venv\Scripts\python.exe scripts\list_world_metadata.py --minecraft-only --format markdown
```

The metadata lister shows each registered candidate, stability notes, audio usage, default overrides, and reproducible low-smoke commands.

Generate a notes-ready command manifest without opening a ModernGL window:

```powershell
.\scripts\smoke_worlds.ps1 -MinecraftOnly -KeepGoing -DryRun -ReportPath logs\validation\minecraft-candidates-command-manifest.md
```

## 2. Preflight The Display

Before comparing many worlds, prove the local OpenGL window path with one known-safe world and a real process timeout:

```powershell
.\scripts\smoke_worlds.ps1 -World minecraft-3d -TimeoutSeconds 20 -ReportPath logs\validation\display-preflight.md
```

If this reports `BLOCKED` or `TIMEOUT`, treat the run as a display/GPU/session problem and do not retry every candidate until the recovery steps below pass. Shader compile errors are recorded as `FAIL` and should be treated as world regressions.

## 3. Smoke Selected Worlds

Run every registered world, preserving the original tiny-smoke behavior. This compatibility path still disables thermal hold and uses `--cpu-workers 0`, so it should only be used for startup/shader checks:

```powershell
.\scripts\smoke_worlds.ps1
```

Run only Minecraft-like candidates and write a notes-ready report. Filtered comparison runs keep thermal hold armed with conservative limits unless you pass `-DisableThermalHold`:

```powershell
.\scripts\smoke_worlds.ps1 -MinecraftOnly -KeepGoing -TimeoutSeconds 20 -ReportPath logs\validation\minecraft-candidates-smoke.md
```

Run a focused comparison set:

```powershell
.\scripts\smoke_worlds.ps1 -World minecraft-long-term-3d,minecraft-3d -TimeoutSeconds 20 -ReportPath logs\validation\focused-minecraft-smoke.md
```

The report records exact command lines, pass/fail/blocked/timeout/thermal-hold status, elapsed seconds, and a blank visual/performance notes table. Fill the notes immediately after reviewing the window or captured output.
Raw stdout and stderr logs are kept under `logs\validation\smoke-*`.

## 4. Targeted Candidate Check

For a new candidate, keep the first proof short and low-cost:

```powershell
$candidate = "minecraft-perfect-ecosystem-3d"
.\scripts\smoke_worlds.ps1 -World $candidate -Width 640 -Height 360 -TileSize 24 -Substeps 1 -RaySteps 32 -FxIntensity 0.3 -CpuWorkers 0 -QuitAfter 3 -TimeoutSeconds 30 -ReportPath logs\validation\targeted-minecraft-candidate.md
```

Record:

| Field | Notes |
| --- | --- |
| World id |  |
| Git branch / commit |  |
| Command |  |
| Result | pass / fail / blocked |
| Exit code or timeout |  |
| FPS or responsiveness |  |
| Visual read | terrain, water, plants, sky, caves, villages, living motion |
| Artifacts | shader errors, blank frame, NaNs, flicker, bad camera, noisy materials |
| Evidence | report path, stdout/stderr logs, screenshot path if captured manually |
| Thermal or display blockers |  |
| Recovery actions |  |
| Follow-up |  |

## 5. Stress Sanity, Not Automation

Do not use scheduled validation for long heat runs, and do not include stress commands in the harness report. After low smoke and visual checks pass, a human can opt into a bounded stress sanity profile with thermal hold enabled:

```powershell
$candidate = "minecraft-perfect-ecosystem-3d"
.\.venv\Scripts\python.exe -B main.py --world $candidate -wnd glfw --width 1920 --height 1080 --tile-size 8 --substeps 24 --ray-steps 160 --fx-intensity 1.3 --cpu-workers 8 --cpu-matrix 1024 --max-gpu-temp 80 --max-cpu-temp 88 --quit-after 300
```

Stop early if the desktop becomes unresponsive, fans ramp unexpectedly, or thermal hold appears before useful visual evidence is collected.

## Display Or Hardware Blockers

If smoke fails before the world opens:

- Close the window if it exists, press `Ctrl+C` in the terminal if the script is still attached, and wait for the script timeout to kill the child Python process. A visible GLFW window can linger briefly while cleanup completes.
- If a Python process survives a timed-out smoke, identify only the matching Garage Life Lab process before stopping it:

```powershell
Get-CimInstance Win32_Process |
    Where-Object {
        $_.Name -match "^(python|pythonw|py)\.exe$" -and
        $_.CommandLine -like "*realtimeSimulation*main.py*"
    } |
    Select-Object ProcessId, Name, CommandLine

Stop-Process -Id <ProcessId> -Force
```

- Leave the generated `logs\validation\smoke-*` stdout/stderr files and report in place.
- Verify the command is running in a real Windows desktop session, not a headless service session.
- Check that the NVIDIA driver and `nvidia-smi` are available when thermal monitoring is expected.
- Confirm `.venv` dependencies are installed from `requirements.txt`.
- Let the machine cool down if fans or temperatures stayed high after a failed run.
- Retry one known-safe world with low settings:

```powershell
.\scripts\smoke_worlds.ps1 -World minecraft-3d -TimeoutSeconds 20 -ReportPath logs\validation\display-recovery-smoke.md
```

- If the failure is a shader compile error, save the report and the stderr tail as a world regression. If it is a display/context error, document it as a local hardware/display blocker and rerun candidate comparison only after the known-safe world passes.
