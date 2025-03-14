@echo off
title CSM Windows Launcher

:: Set up environment variables
set TORCH_COMPILE_BACKEND=eager
set CUDA_LAUNCH_BLOCKING=1
set TORCH_DYNAMO_VERBOSE=1

:: Activate virtual environment
call .venv\Scripts\activate.bat

echo ===== CSM Speech-to-Speech Interface for Windows =====
echo.
echo 1. Start CSM with default settings
echo 2. Create a voice profile
echo 3. Start CSM with a specific voice
echo 4. Start CSM in CPU mode (if GPU is causing issues)
echo 5. Run diagnostics
echo 6. Exit
echo.

:menu
set /p choice=Enter your choice (1-6): 

if "%choice%"=="1" (
    echo.
    echo Starting CSM with default settings...
    python run_speech_interface.py
    goto end
)

if "%choice%"=="2" (
    echo.
    echo Creating a new voice profile...
    python create_voice_profile.py
    goto end
)

if "%choice%"=="3" (
    echo.
    echo Available voice profiles:
    python -c "from speech_interface import SpeechInterface; interface = SpeechInterface(); print('\n'.join([f'{i+1}. {v}' for i, v in enumerate(interface.list_voice_profiles())]))"
    echo.
    set /p voice=Enter voice name: 
    echo Starting CSM with voice: %voice%
    python run_speech_interface.py --voice %voice%
    goto end
)

if "%choice%"=="4" (
    echo.
    echo Starting CSM in CPU mode...
    python run_speech_interface.py --device cpu
    goto end
)

if "%choice%"=="5" (
    echo.
    echo Running diagnostics...
    python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}'); print(f'PyTorch Version: {torch.__version__}'); print(f'GPU Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
    echo.
    echo Press any key to return to menu...
    pause >nul
    cls
    echo ===== CSM Speech-to-Speech Interface for Windows =====
    echo.
    echo 1. Start CSM with default settings
    echo 2. Create a voice profile
    echo 3. Start CSM with a specific voice
    echo 4. Start CSM in CPU mode (if GPU is causing issues)
    echo 5. Run diagnostics
    echo 6. Exit
    echo.
    goto menu
)

if "%choice%"=="6" (
    echo.
    echo Exiting...
    goto end
)

echo Invalid choice. Please try again.
goto menu

:end
:: Deactivate virtual environment
call deactivate
