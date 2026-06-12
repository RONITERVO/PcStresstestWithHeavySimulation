@echo off
REM User-friendly Windows entrypoint for Garage Life Lab.
setlocal
cd /d %~dp0

if not exist dist\GarageLifeLab\GarageLifeLab.exe goto launch_source

powershell -NoProfile -ExecutionPolicy Bypass -File scripts\is_bundled_launcher_fresh.ps1 >nul 2>nul
if %errorlevel%==0 (
    start "" dist\GarageLifeLab\GarageLifeLab.exe
    exit /b
)

:launch_source
if exist dist\GarageLifeLab\GarageLifeLab.exe echo Bundled app is stale; launching source launcher.
call launch_garage_life_lab.cmd
