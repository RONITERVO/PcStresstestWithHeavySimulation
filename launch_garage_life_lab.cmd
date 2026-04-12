@echo off
REM Launches Garage Life Lab with the tile-world 1080p preset
setlocal
cd /d %~dp0

if exist .venv\Scripts\python.exe (
    set PYTHON=.venv\Scripts\python.exe
) else (
    set PYTHON=python
)

%PYTHON% main.py -wnd glfw --width 1920 --height 1080 --tile-size 12 --substeps 24 --cpu-workers 16 --cpu-matrix 1024 --glow 1.3 --exposure 1.45
