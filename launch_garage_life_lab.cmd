@echo off
REM Launches Garage Life Lab with the baseline 1080p preset
setlocal
cd /d %~dp0

if exist .venv\Scripts\python.exe (
    set PYTHON=.venv\Scripts\python.exe
) else (
    set PYTHON=python
)

%PYTHON% main.py --width 1920 --height 1080 --substeps 16 --cpu-workers 8 --cpu-matrix 768
