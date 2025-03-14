#!/usr/bin/env python
"""
Offline launcher script for CSM speech interface.
This script runs the CSM system using locally downloaded models.
"""

import os
import sys
import torch
import argparse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("offline_launcher")

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

def main():
    parser = argparse.ArgumentParser(description="Run CSM in offline mode")
    parser.add_argument("--voice", type=str, default=None, 
                    help="Voice profile to use (default: first available)")
    parser.add_argument("--debug", action="store_true", 
                    help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Set logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Print system info
    logger.info("Starting CSM in offline mode")
    if torch.cuda.is_available():
        logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA version: {torch.version.cuda}")
    else:
        logger.info("CUDA not available, using CPU")
    
    # Check if local config exists
    try:
        import local_config
        logger.info("Local models configuration found")
    except ImportError:
        logger.error("Local configuration not found. Please run download_models.py first.")
        sys.exit(1)
    
    # Import speech interface
    from speech_interface import SpeechInterface
    
    # Create interface
    interface = SpeechInterface(
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Set voice if specified
    if args.voice:
        try:
            interface.set_active_voice(args.voice)
            logger.info(f"Using voice profile: {args.voice}")
        except ValueError:
            available_voices = interface.list_voice_profiles()
            logger.warning(f"Voice profile '{args.voice}' not found. Available voices: {', '.join(available_voices)}")
            if available_voices:
                interface.set_active_voice(available_voices[0])
                logger.info(f"Using default voice profile: {available_voices[0]}")
    
    # Start conversation loop
    try:
        interface.conversation_loop()
    except KeyboardInterrupt:
        logger.info("Exiting...")
    except Exception as e:
        logger.error(f"Error in conversation loop: {str(e)}")
        if args.debug:
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
