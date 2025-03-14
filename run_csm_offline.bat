@echo off
echo Starting CSM in offline mode...
call .venv\Scripts\activate.bat
python run_csm_offline.py %*
