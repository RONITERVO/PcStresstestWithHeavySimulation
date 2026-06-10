@echo off
REM User-friendly Windows entrypoint for Garage Life Lab.
setlocal
cd /d %~dp0

if exist dist\GarageLifeLab\GarageLifeLab.exe (
    start "" dist\GarageLifeLab\GarageLifeLab.exe
    exit /b
)

call launch_garage_life_lab.cmd
