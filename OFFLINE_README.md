# CSM Offline Mode

This guide explains how to set up CSM (Conversational Speech Model) to run completely offline without requiring an internet connection after the initial setup.

## Why Run Offline?

Running the model locally without internet dependency provides several benefits:

1. **Faster Performance**: Eliminates network latency, reducing response times
2. **Consistent Experience**: No variation due to internet connectivity issues
3. **Privacy**: All processing happens locally on your machine
4. **Reliability**: Works anytime, anywhere, regardless of internet availability

## System Requirements

- Windows with CUDA support
- NVIDIA GPU with at least 8GB VRAM (RTX 3060 Ti recommended)
- 32GB RAM recommended
- At least 15GB of free disk space
- Python 3.8 or newer

## Quick Setup (Windows)

For the easiest setup, simply run:

```
setup_offline_csm.bat
```

This batch file will:
1. Download all required models (requires internet connection)
2. Modify the code to use local models
3. Create an offline launcher

## Manual Setup

If you prefer to run the steps manually:

### Step 1: Download Models

```bash
# Activate your virtual environment
.\.venv\Scripts\activate

# Download all models
python download_models.py
```

This will download:
- CSM-1B model (speech synthesis)
- TinyLlama-1.1B-Chat-v1.0 (text generation)
- Whisper Base (speech recognition)
- Mimi tokenizer (audio tokenization)

The models will be saved in the `local_models` directory.

### Step 2: Modify Code for Offline Use

```bash
python modify_for_offline.py
```

This script modifies the necessary files to use local models instead of downloading from Hugging Face.

## Running CSM Offline

After setup is complete, you can run CSM completely offline:

```bash
# Using the batch file
run_csm_offline.bat

# OR using Python directly
python run_csm_offline.py
```

### Command Line Options

```bash
python run_csm_offline.py --voice your_voice_profile  # Use a specific voice profile
python run_csm_offline.py --debug  # Run in debug mode with more logging
```

## Technical Details

### What Changes Were Made

The offline modification:

1. Downloads all models once and saves them locally
2. Modifies the code to use local paths instead of downloading from Hugging Face
3. Creates a fallback mechanism (will still work if you run online)
4. Preserves all original functionality

### Backups

During modification, backup files are created with the `.bak` extension. If you encounter issues, you can restore the original files.

### Performance Optimization

The system is optimized for your RTX 3060 Ti:

- Uses 4-bit quantization for the LLM to reduce VRAM usage
- Selects appropriately sized models for your hardware
- Configures PyTorch for Windows compatibility

## Troubleshooting

If you encounter issues:

1. **"Local configuration not found"**: Make sure you ran `download_models.py` successfully.

2. **CUDA errors**: Try running with reduced precision or switch to CPU mode:
   ```
   python run_csm_offline.py --device cpu
   ```

3. **Voice profile issues**: If a voice profile is not found, the system will use the first available one.

4. **Other errors**: Run in debug mode for more detailed information:
   ```
   python run_csm_offline.py --debug
   ```

## Files Added

- `download_models.py` - Downloads required models
- `modify_for_offline.py` - Modifies code for local use
- `run_csm_offline.py/.bat` - Launchers for offline mode
- `local_config.py` - Configuration with model paths
- `setup_offline_csm.bat` - Easy setup script
- `OFFLINE_README.md` - This documentation

Enjoy using CSM in offline mode!
