"""
Script to create a voice profile for the CSM speech interface.
This can be run to set up a voice profile before starting the main interface.
"""

import os
import torch
import argparse
from speech_interface import SpeechInterface

def main():
    parser = argparse.ArgumentParser(description="Create a voice profile for CSM")
    
    parser.add_argument(
        "--name", 
        type=str, 
        default="my_voice", 
        help="Name for the voice profile (default: my_voice)"
    )
    
    parser.add_argument(
        "--speaker-id", 
        type=int, 
        default=0, 
        help="Speaker ID to use (default: 0)"
    )
    
    parser.add_argument(
        "--description", 
        type=str, 
        default="My custom voice profile", 
        help="Description for the voice profile"
    )
    
    parser.add_argument(
        "--samples", 
        type=int, 
        default=3, 
        help="Number of voice samples to record (default: 3)"
    )
    
    parser.add_argument(
        "--device", 
        type=str, 
        default=None, 
        choices=["cuda", "cpu"], 
        help="Device to use (default: auto-detect)"
    )
    
    args = parser.parse_args()
    
    # Create interface
    print(f"Initializing CSM... (this may take a minute)")
    interface = SpeechInterface(device=args.device)
    
    # Check if profile already exists
    voice_profiles = interface.list_voice_profiles()
    if args.name in voice_profiles:
        overwrite = input(f"Voice profile '{args.name}' already exists. Overwrite? (y/n): ")
        if overwrite.lower() != 'y':
            print("Exiting without changes.")
            return
    
    # Create the profile
    print(f"\nCreating voice profile: {args.name}")
    interface.create_voice_profile(args.name, args.speaker_id, args.description)
    
    # Sample texts to record
    sample_texts = [
        "Hello, this is my voice for the conversational speech model.",
        "I'm recording a sample to create a custom voice profile.",
        "This will help the model generate speech that sounds more like me.",
        "I enjoy having natural conversations with AI assistants.",
        "The weather today is quite pleasant, don't you think?"
    ]
    
    # Limit to requested number of samples
    sample_texts = sample_texts[:args.samples]
    
    # Record samples
    print("\nLet's record some voice samples.")
    print("Speak clearly and naturally for best results.")
    
    for i, text in enumerate(sample_texts):
        print(f"\nSample {i+1}/{len(sample_texts)}: '{text}'")
        print("Press Enter when you're ready to start recording...")
        input()
        
        audio_data, rate = interface.record_user_speech()
        audio_tensor = torch.from_numpy(audio_data).float()
        
        interface.add_sample_to_voice(args.name, text, audio_tensor, rate)
        print(f"Sample {i+1} recorded successfully!")
    
    # Set as active voice
    interface.set_active_voice(args.name)
    print(f"\nVoice profile '{args.name}' created successfully with {len(sample_texts)} samples.")
    print(f"This voice is now set as the active voice.")

if __name__ == "__main__":
    main()
