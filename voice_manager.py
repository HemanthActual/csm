#!/usr/bin/env python
"""
Voice profile management tool for CSM Speech-to-Speech Interface.
Allows creating, editing, and testing voice profiles.
"""

import os
import torch
import torchaudio
import numpy as np
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Optional
from huggingface_hub import hf_hub_download
from generator import load_csm_1b, Segment
from asr import AudioRecorder, SpeechRecognizer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("voice_manager")

# Constants
VOICES_DIR = "voices"
DEFAULT_SPEAKER_ID = 0
SAMPLE_TEXTS = [
    "Hello, this is my voice sample for the conversational speech model.",
    "I'm creating a custom voice profile that captures my speaking style.",
    "The quick brown fox jumps over the lazy dog.",
    "Today is a beautiful day and I'm happy to be recording this sample.",
    "This voice profile will help the model generate speech that sounds more like me."
]


class VoiceManager:
    """Tool for managing voice profiles for CSM."""
    
    def __init__(self, voices_dir: str = VOICES_DIR, device: Optional[str] = None):
        """
        Initialize the voice manager.
        
        Args:
            voices_dir: Directory to store voice profiles
            device: Device to use ('cuda', 'cpu', or None for auto-detection)
        """
        self.voices_dir = Path(voices_dir)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create directories
        os.makedirs(self.voices_dir, exist_ok=True)
        
        # Load voice profiles
        self.voice_profiles = self._load_voice_profiles()
        
        # Initialize audio recorder
        self.audio_recorder = AudioRecorder()
        
        # Initialize CSM model for testing
        self.generator = None  # Lazy-loaded when needed
    
    def _load_voice_profiles(self) -> Dict[str, Dict]:
        """Load voice profiles from disk."""
        profiles = {}
        
        if not self.voices_dir.exists():
            return profiles
            
        for voice_name in os.listdir(self.voices_dir):
            voice_path = self.voices_dir / voice_name
            if voice_path.is_dir():
                # Look for profile.json
                profile_file = voice_path / "profile.json"
                if profile_file.exists():
                    try:
                        with open(profile_file, 'r') as f:
                            profile = json.load(f)
                            
                            # Add additional info
                            sample_count = len(list((voice_path / "samples").glob("*.wav"))) if (voice_path / "samples").exists() else 0
                            profile['sample_count'] = sample_count
                            
                            profiles[voice_name] = profile
                    except Exception as e:
                        logger.error(f"Error loading voice profile {voice_name}: {str(e)}")
        
        return profiles
    
    def _load_generator(self):
        """Lazy-load the CSM generator."""
        if self.generator is None:
            logger.info("Loading CSM model for testing...")
            model_path = hf_hub_download(repo_id="sesame/csm-1b", filename="ckpt.pt")
            self.generator = load_csm_1b(model_path, self.device)
            logger.info("CSM model loaded")
    
    def list_voices(self):
        """List available voice profiles."""
        if not self.voice_profiles:
            print("No voice profiles found.")
            return
        
        print("\n===== Available Voice Profiles =====")
        print(f"{'Name':<20} {'Speaker ID':<10} {'Samples':<10} {'Description'}")
        print("-" * 60)
        
        for name, profile in self.voice_profiles.items():
            print(f"{name:<20} {profile.get('speaker_id', 0):<10} {profile.get('sample_count', 0):<10} {profile.get('description', '')}")
    
    def create_voice(self, name: str, speaker_id: int = DEFAULT_SPEAKER_ID, description: str = ""):
        """Create a new voice profile."""
        if name in self.voice_profiles:
            print(f"Error: Voice profile '{name}' already exists.")
            return False
        
        profile = {
            'name': name,
            'speaker_id': speaker_id,
            'description': description,
            'sample_count': 0
        }
        
        # Create directory structure
        voice_path = self.voices_dir / name
        os.makedirs(voice_path, exist_ok=True)
        os.makedirs(voice_path / "samples", exist_ok=True)
        
        # Save profile
        with open(voice_path / "profile.json", 'w') as f:
            json.dump(profile, f, indent=2)
        
        self.voice_profiles[name] = profile
        logger.info(f"Created voice profile: {name}")
        print(f"Created voice profile: {name}")
        
        return True
    
    def delete_voice(self, name: str, confirm: bool = True):
        """Delete a voice profile."""
        if name not in self.voice_profiles:
            print(f"Error: Voice profile '{name}' does not exist.")
            return False
        
        if confirm:
            response = input(f"Are you sure you want to delete voice profile '{name}'? (y/N): ")
            if response.lower() != 'y':
                print("Deletion cancelled.")
                return False
        
        # Delete directory
        import shutil
        voice_path = self.voices_dir / name
        try:
            shutil.rmtree(voice_path)
            del self.voice_profiles[name]
            logger.info(f"Deleted voice profile: {name}")
            print(f"Deleted voice profile: {name}")
            return True
        except Exception as e:
            print(f"Error deleting voice profile: {str(e)}")
            return False
    
    def record_sample(self, voice_name: str, text: Optional[str] = None):
        """Record a voice sample for a profile."""
        if voice_name not in self.voice_profiles:
            print(f"Error: Voice profile '{voice_name}' does not exist.")
            return False
        
        # Get or select sample text
        if text is None:
            print("\nSelect a sample text to record:")
            for i, sample_text in enumerate(SAMPLE_TEXTS):
                print(f"{i+1}. {sample_text}")
            print("C. Custom text")
            
            choice = input("Enter choice (1-5, C): ")
            if choice.lower() == 'c':
                text = input("Enter custom text to record: ")
            elif choice.isdigit() and 1 <= int(choice) <= len(SAMPLE_TEXTS):
                text = SAMPLE_TEXTS[int(choice)-1]
            else:
                print("Invalid choice. Using default sample.")
                text = SAMPLE_TEXTS[0]
        
        # Record the sample
        print(f"\nSample text: '{text}'")
        print("Press Enter to start recording...")
        input()
        
        audio_data = self.audio_recorder.start_recording()
        sample_id = self.voice_profiles[voice_name].get('sample_count', 0)
        
        # Save audio file
        samples_dir = self.voices_dir / voice_name / "samples"
        os.makedirs(samples_dir, exist_ok=True)
        
        audio_path = samples_dir / f"sample_{sample_id}.wav"
        temp_file = self.audio_recorder.save_recording(audio_data, str(audio_path))
        
        # Save text
        text_path = samples_dir / f"sample_{sample_id}.txt"
        with open(text_path, 'w') as f:
            f.write(text)
        
        # Update profile
        self.voice_profiles[voice_name]['sample_count'] = sample_id + 1
        
        # Save profile
        with open(self.voices_dir / voice_name / "profile.json", 'w') as f:
            json.dump({
                'name': self.voice_profiles[voice_name]['name'],
                'speaker_id': self.voice_profiles[voice_name]['speaker_id'],
                'description': self.voice_profiles[voice_name].get('description', '')
            }, f, indent=2)
        
        print(f"Sample {sample_id} added to voice profile {voice_name}")
        
        # Offer to record another sample
        response = input("Record another sample? (y/N): ")
        if response.lower() == 'y':
            return self.record_sample(voice_name)
        
        return True
    
    def test_voice(self, voice_name: str, text: Optional[str] = None):
        """Test a voice profile with CSM."""
        if voice_name not in self.voice_profiles:
            print(f"Error: Voice profile '{voice_name}' does not exist.")
            return False
        
        # Check if there are samples
        sample_count = self.voice_profiles[voice_name].get('sample_count', 0)
        if sample_count == 0:
            print(f"Error: Voice profile '{voice_name}' has no samples.")
            return False
        
        # Lazy-load the generator
        self._load_generator()
        
        # Get test text
        if text is None:
            text = input("Enter text to synthesize (or press Enter for default): ")
            if not text:
                text = "This is a test of my custom voice profile with the CSM model."
        
        # Prepare context
        context = []
        speaker_id = self.voice_profiles[voice_name]['speaker_id']
        
        # Load samples
        samples_dir = self.voices_dir / voice_name / "samples"
        for i in range(sample_count):
            audio_path = samples_dir / f"sample_{i}.wav"
            text_path = samples_dir / f"sample_{i}.txt"
            
            if audio_path.exists() and text_path.exists():
                try:
                    # Load audio
                    waveform, sample_rate = torchaudio.load(str(audio_path))
                    # Load text
                    with open(text_path, 'r') as f:
                        sample_text = f.read().strip()
                    
                    # Resample if needed
                    if sample_rate != self.generator.sample_rate:
                        audio = torchaudio.functional.resample(
                            waveform.squeeze(), 
                            orig_freq=sample_rate, 
                            new_freq=self.generator.sample_rate
                        )
                    else:
                        audio = waveform.squeeze()
                    
                    # Create segment
                    segment = Segment(
                        text=sample_text,
                        speaker=speaker_id,
                        audio=audio
                    )
                    context.append(segment)
                except Exception as e:
                    logger.error(f"Error loading sample {i}: {str(e)}")
        
        print(f"Synthesizing speech with {len(context)} context samples...")
        
        # Generate audio
        audio = self.generator.generate(
            text=text,
            speaker=speaker_id,
            context=context,
            max_audio_length_ms=10000
        )
        
        # Save and play audio
        output_file = f"{voice_name}_test.wav"
        torchaudio.save(output_file, audio.unsqueeze(0).cpu(), self.generator.sample_rate)
        print(f"Audio saved to {output_file}")
        
        # Try to play audio
        try:
            import sounddevice as sd
            sd.play(audio.cpu().numpy(), self.generator.sample_rate)
            sd.wait()
        except ImportError:
            print("sounddevice not installed. Cannot play audio directly.")
        
        return True
    
    def import_samples(self, voice_name: str, audio_files: List[str], texts: Optional[List[str]] = None):
        """Import existing audio samples to a voice profile."""
        if voice_name not in self.voice_profiles:
            print(f"Error: Voice profile '{voice_name}' does not exist.")
            return False
        
        if texts is None:
            # Prompt for text for each audio file
            texts = []
            for audio_file in audio_files:
                text = input(f"Enter text for {audio_file}: ")
                texts.append(text)
        
        if len(texts) != len(audio_files):
            print("Error: Number of texts must match number of audio files.")
            return False
        
        # Add each sample
        samples_dir = self.voices_dir / voice_name / "samples"
        os.makedirs(samples_dir, exist_ok=True)
        
        sample_id = self.voice_profiles[voice_name].get('sample_count', 0)
        
        for audio_file, text in zip(audio_files, texts):
            try:
                # Load audio
                waveform, sample_rate = torchaudio.load(audio_file)
                
                # Save to profile
                audio_path = samples_dir / f"sample_{sample_id}.wav"
                torchaudio.save(str(audio_path), waveform, sample_rate)
                
                # Save text
                text_path = samples_dir / f"sample_{sample_id}.txt"
                with open(text_path, 'w') as f:
                    f.write(text)
                
                sample_id += 1
                print(f"Imported sample {sample_id-1} from {audio_file}")
                
            except Exception as e:
                print(f"Error importing {audio_file}: {str(e)}")
        
        # Update profile
        self.voice_profiles[voice_name]['sample_count'] = sample_id
        
        # Save profile
        with open(self.voices_dir / voice_name / "profile.json", 'w') as f:
            json.dump({
                'name': self.voice_profiles[voice_name]['name'],
                'speaker_id': self.voice_profiles[voice_name]['speaker_id'],
                'description': self.voice_profiles[voice_name].get('description', '')
            }, f, indent=2)
        
        print(f"Imported {sample_id - self.voice_profiles[voice_name].get('sample_count', 0)} samples to voice profile {voice_name}")
        return True


def main():
    """Parse arguments and run the voice manager."""
    parser = argparse.ArgumentParser(description="CSM Voice Profile Manager")
    
    # Main actions
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument("--list", action="store_true", help="List available voice profiles")
    action_group.add_argument("--create", type=str, metavar="NAME", help="Create a new voice profile")
    action_group.add_argument("--delete", type=str, metavar="NAME", help="Delete a voice profile")
    action_group.add_argument("--record", type=str, metavar="NAME", help="Record samples for a voice profile")
    action_group.add_argument("--test", type=str, metavar="NAME", help="Test a voice profile")
    action_group.add_argument("--import", dest="import_files", type=str, metavar="NAME", help="Import audio files to a voice profile")
    
    # Additional options
    parser.add_argument("--speaker-id", type=int, default=DEFAULT_SPEAKER_ID, help="Speaker ID for new voice profile")
    parser.add_argument("--description", type=str, default="", help="Description for new voice profile")
    parser.add_argument("--text", type=str, help="Text for recording or testing")
    parser.add_argument("--files", type=str, nargs="+", help="Audio files to import")
    parser.add_argument("--texts", type=str, nargs="+", help="Texts for imported audio files")
    
    # Hardware options
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], help="Device to use")
    
    args = parser.parse_args()
    
    # Create voice manager
    manager = VoiceManager(device=args.device)
    
    # Run requested action
    if args.list:
        manager.list_voices()
    elif args.create:
        manager.create_voice(args.create, args.speaker_id, args.description)
    elif args.delete:
        manager.delete_voice(args.delete)
    elif args.record:
        manager.record_sample(args.record, args.text)
    elif args.test:
        manager.test_voice(args.test, args.text)
    elif args.import_files:
        if not args.files:
            print("Error: --files is required with --import")
            return
        manager.import_samples(args.import_files, args.files, args.texts)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nOperation cancelled.")
    finally:
        # Clean up
        if "manager" in locals() and hasattr(manager, "audio_recorder"):
            manager.audio_recorder.close()
