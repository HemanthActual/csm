# CSM Speech-to-Speech Interface

This extension to the CSM (Conversational Speech Model) adds speech-to-speech interaction capabilities, turning CSM into a complete conversational voice interface.

## Overview

The CSM Speech-to-Speech Interface combines three key components:

1. **Speech Recognition (ASR)** - Converts your voice to text using Whisper
2. **Text Generation (LLM)** - Generates responses using a local LLM
3. **Speech Synthesis (CSM)** - Converts text responses back to speech using CSM

Together, these components create a complete speech-to-speech experience where you can talk naturally to the system and hear responses in a customized voice.

## Setup

### 1. Install Requirements

First, install the base CSM requirements:

```bash
pip install -r requirements.txt
```

Then install the additional requirements for the speech interface:

```bash
pip install -r requirements_speech_interface.txt
```

### 2. Download Models (Automatic)

The system will automatically download the necessary models on first run:

- CSM model (from Hugging Face: `sesame/csm-1b`)
- Whisper model (default: `openai/whisper-base`)
- LLM model (will try to find an available model or download Llama-3, Mistral, or TinyLlama)

### 3. Run the Interface

For the basic experience, simply run:

```bash
python run_speech_interface.py
```

## Voice Customization

The system supports creating custom voice profiles to personalize the speech output.

### Using the Voice Manager

The voice manager tool helps you create and manage voice profiles:

```bash
# List available voice profiles
python voice_manager.py --list

# Create a new voice profile
python voice_manager.py --create my_voice --speaker-id 0 --description "My custom voice"

# Record samples for a voice profile
python voice_manager.py --record my_voice

# Test a voice profile
python voice_manager.py --test my_voice --text "Hello, this is my custom voice."

# Import existing audio samples
python voice_manager.py --import my_voice --files sample1.wav sample2.wav --texts "Sample one text" "Sample two text"
```

### Speaker IDs

CSM supports different base voice characteristics via speaker IDs:
- Speaker ID 0: Default voice (typically masculine)
- Speaker ID 1: Alternative voice (typically feminine)

For the best results, use voice samples that match your desired voice characteristics.

## Advanced Usage

### Command-line Arguments

The interface supports various command-line arguments:

```bash
# Use specific models
python run_speech_interface.py --asr-model "openai/whisper-small" --llm-model "mistralai/Mistral-7B-Instruct-v0.2"

# Specify device for inference
python run_speech_interface.py --device cuda

# Use a specific voice profile
python run_speech_interface.py --voice my_voice

# Create a new voice profile with samples
python run_speech_interface.py --create-voice new_voice --speaker-id 1

# Enable debug logging
python run_speech_interface.py --debug
```

### System Requirements

- **Minimum**: 8GB RAM, any CUDA-compatible GPU with at least 4GB VRAM
- **Recommended**: 16GB RAM, NVIDIA GPU with 8GB+ VRAM (RTX 3060 or better)
- **Storage**: ~5GB for all models

## Troubleshooting

### Common Issues

1. **Speech Recognition Issues**
   - If you're experiencing poor transcription, try using a larger Whisper model:
     ```bash
     python run_speech_interface.py --asr-model "openai/whisper-medium"
     ```

2. **LLM Loading Issues**
   - If you have limited VRAM, try a smaller model or enable 4-bit quantization (default)
   - For very limited systems, try TinyLlama:
     ```bash
     python run_speech_interface.py --llm-model "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
     ```

3. **Audio Playback Issues**
   - If you can't hear audio, check your system's default audio output device
   - The interface saves audio to `response.wav` if direct playback fails

4. **"No module named X" Error**
   - Make sure you've installed all requirements:
     ```bash
     pip install -r requirements.txt
     pip install -r requirements_speech_interface.txt
     ```

### Checking Hardware

To verify your system's compatibility:

```bash
# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, Device count: {torch.cuda.device_count()}, Device name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# Check available memory
python -c "import torch; print(f'Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB') if torch.cuda.is_available() else print('CUDA not available')"
```

## Limitations

- **ASR Accuracy**: Speech recognition may not be perfect, especially in noisy environments
- **LLM Content**: The quality of responses depends on the LLM used
- **Voice Customization**: Voice adaptation is based on limited samples and may not perfectly match the target voice
- **Latency**: Processing full conversation turns takes time, especially on slower hardware

## Future Improvements

- Web interface using Gradio
- More advanced voice customization
- Real-time response streaming
- Integration with external APIs
- Multi-speaker conversation tracking
