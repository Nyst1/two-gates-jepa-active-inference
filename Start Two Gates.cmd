@echo off
setlocal
title Two Gates
cd /d "%~dp0"

if /I "%TWO_GATES_NO_BROWSER%"=="1" (
    "%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -ExecutionPolicy Bypass -File "%~dp0start-demo.ps1" -NoBrowser
) else (
    "%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -ExecutionPolicy Bypass -File "%~dp0start-demo.ps1"
)

set "TWO_GATES_EXIT=%ERRORLEVEL%"
if not "%TWO_GATES_EXIT%"=="0" (
    echo.
    echo Press any key to close this window.
    pause >nul
)

exit /b %TWO_GATES_EXIT%
