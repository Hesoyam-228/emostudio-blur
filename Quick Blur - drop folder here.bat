@echo off
set PYTHONIOENCODING=utf-8
set PYTHONWARNINGS=ignore
set HF_HUB_VERBOSITY=error
set HF_HUB_DISABLE_PROGRESS_BARS=1
cd /d "%~dp0"
if not exist "venv\Scripts\python.exe" call install.bat
venv\Scripts\python "src\autoblur.py" %*
pause
