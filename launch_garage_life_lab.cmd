@echo off
REM Launches the Garage Life Lab Windows UI
setlocal
cd /d %~dp0

if exist .venv\Scripts\pythonw.exe (
    start "" .venv\Scripts\pythonw.exe launcher.py
    exit /b
)

if exist .venv\Scripts\python.exe (
    start "" .venv\Scripts\python.exe launcher.py
    exit /b
)

where pyw.exe >nul 2>nul
if %errorlevel%==0 (
    start "" pyw -3.9 launcher.py
    exit /b
)

start "" py -3.9 launcher.py
