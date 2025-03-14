@echo off
echo === CSM Speech-to-Speech Interface for Windows ===
echo.
echo This launcher configures PyTorch to work without Triton on Windows
echo.

REM Activate the virtual environment
call .venv\Scripts\activate.bat

REM Set environment variables to disable problematic features
set TORCH_COMPILE_BACKEND=eager
set CUDA_LAUNCH_BLOCKING=1

REM Run the patched script
python run_speech_interface.py %*

REM Deactivate the virtual environment
call deactivate
