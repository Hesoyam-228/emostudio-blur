@echo off
cd /d "%~dp0"
if not exist "venv\Scripts\pythonw.exe" (
    call install.bat
    exit /b
)
start "" "venv\Scripts\pythonw.exe" "src\app.py"
