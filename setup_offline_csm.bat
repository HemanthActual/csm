@echo off
echo CSM Offline Setup
echo ================
echo.
echo This script will set up CSM to run completely offline.
echo.
echo Step 1: Download required models (requires internet connection)
echo Step 2: Modify code to use local models
echo Step 3: Create offline launcher
echo.
echo Press Ctrl+C to cancel or any key to continue...
pause > nul

echo.
echo Activating virtual environment...
call .venv\Scripts\activate.bat

echo.
echo Downloading models (this may take some time)...
python download_models.py

echo.
echo Modifying code to use local models...
python modify_for_offline.py

echo.
echo Setup complete!
echo.
echo To run CSM offline, use: run_csm_offline.bat
echo.
pause
