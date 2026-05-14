@echo off
setlocal
cd /d "%~dp0"
powershell -ExecutionPolicy Bypass -File "%~dp0run_demo.ps1"
if errorlevel 1 (
    echo.
    echo Demo startup failed. Press any key to close this window.
    pause >nul
)
endlocal
