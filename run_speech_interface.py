#!/usr/bin/env python
"""
# Import PyTorch and disable problematic features first
import os
import torch

# Configure PyTorch for Windows compatibility
os.environ["TORCH_COMPILE_BACKEND"] = "eager"  # Use eager mode instead of inductor

# Tell PyTorch to suppress errors and fall back to eager mode
try:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
except ImportError:
    pass

# Disable cuda graph capture which can cause issues
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
"""
"""
Launcher script for the CSM Speech-to-Speech Interface.
Provides a simple way to start the system with various options.
"""

import argparse
import logging
import os
import sys
from speech_interface import SpeechInterface

def main():
    """Parse arguments and start the speech interface."""
    parser = argparse.ArgumentParser(description="CSM Speech-to-Speech Interface")
    
    # Model selection
    parser.add_argument(
        "--asr-model", 
        type=str, 
        default="openai/whisper-base", 
        help="ASR model to use (default: openai/whisper-base)"
    )
    parser.add_argument(
        "--llm-model", 
        type=str, 
        default=None, 
        help="LLM model to use (default: auto-detect)"
    )
    parser.add_argument(
        "--csm-model", 
        type=str, 
        default=None, 
        help="Path to CSM model weights (default: download from HF)"
    )
    
    # Voice options
    parser.add_argument(
        "--voice", 
        type=str, 
        default=None, 
        help="Voice profile to use (default: first available)"
    )
    parser.add_argument(
        "--create-voice", 
        type=str, 
        default=None, 
        help="Create a new voice profile with the given name"
    )
    parser.add_argument(
        "--speaker-id", 
        type=int, 
        default=0, 
        help="Speaker ID for new voice profile (default: 0)"
    )
    
    # Hardware options
    parser.add_argument(
        "--device", 
        type=str, 
        default=None, 
        choices=["cuda", "cpu"], 
        help="Device to use (default: auto-detect)"
    )
    
    # Other options
    parser.add_argument(
        "--debug", 
        action="store_true", 
        help="Enable debug logging"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create interface
    interface = SpeechInterface(
        asr_model=args.asr_model,
        llm_model=args.llm_model,
        csm_model_path=args.csm_model,
        device=args.device
    )
    
    # Handle voice options
    if args.create_voice:
        try:
            print(f"Creating new voice profile: {args.create_voice}")
            interface.create_voice_profile(args.create_voice, args.speaker_id)
            interface.set_active_voice(args.create_voice)
            
            # Prompt for voice samples
            print("\nLet's record some samples for the new voice.")
            sample_texts = [
                "Hello, this is my voice for the conversational speech model.",
                "I'm recording a sample to create a custom voice profile.",
                "This will help the model generate speech that sounds more like me."
            ]
            
            for i, text in enumerate(sample_texts):
                print(f"\nSample {i+1}: '{text}'")
                print("Press Enter to start recording...")
                input()
                audio_data, rate = interface.record_user_speech()
                audio_tensor = torch.from_numpy(audio_data).float()
                interface.add_sample_to_voice(args.create_voice, text, audio_tensor, rate)
        except Exception as e:
            print(f"Error creating voice profile: {str(e)}")
            
    elif args.voice:
        try:
            interface.set_active_voice(args.voice)
        except ValueError:
            available_voices = interface.list_voice_profiles()
            print(f"Voice profile '{args.voice}' not found. Available voices: {', '.join(available_voices)}")
            if available_voices:
                interface.set_active_voice(available_voices[0])
    
    # Start conversation loop
    try:
        interface.conversation_loop()
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"Error in conversation loop: {str(e)}")
        if args.debug:
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
