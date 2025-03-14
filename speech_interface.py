"""
Main interface for the CSM Speech-to-Speech system.
Connects ASR, LLM, and CSM components to create a complete speech-to-speech experience.
"""

import os
import time
import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass
import threading
import logging
import json

# Import components
from huggingface_hub import hf_hub_download
from generator import load_csm_1b, Segment
from asr import AudioRecorder, SpeechRecognizer
from llm import LLMEngine, FallbackLLM

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("speech_interface")

# Configuration
DEFAULT_SPEAKER_ID = 0
MAX_AUDIO_LENGTH_MS = 10000
VOICES_DIR = "voices"

@dataclass
class ConversationTurn:
    """Represents a single turn in the conversation."""
    role: str  # "user" or "assistant"
    text: str
    audio: torch.Tensor
    speaker_id: int = 0
    
    def to_segment(self) -> Segment:
        """Convert to a Segment for CSM."""
        return Segment(
            text=self.text,
            speaker=self.speaker_id,
            audio=self.audio
        )


class SpeechInterface:
    """Main interface for the speech-to-speech system."""
    
    def __init__(self, 
                 asr_model: str = "openai/whisper-base",
                 llm_model: Optional[str] = None,
                 csm_model_path: Optional[str] = None,
                 device: Optional[str] = None,
                 voices_dir: str = VOICES_DIR):
        """
        Initialize the speech interface.
        
        Args:
            asr_model: Name of the ASR model to use
            llm_model: Name of the LLM model to use (or None for auto-selection)
            csm_model_path: Path to the CSM model weights (or None to download)
            device: Device to use ('cuda', 'cpu', or None for auto-detection)
            voices_dir: Directory to store voice profiles
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Create directories
        self.voices_dir = voices_dir
        os.makedirs(self.voices_dir, exist_ok=True)
        
        # Initialize ASR module
        logger.info("Initializing ASR module...")
        self.audio_recorder = AudioRecorder()
        self.speech_recognizer = SpeechRecognizer(model_name=asr_model)
        
        # Initialize LLM module
        logger.info("Initializing LLM module...")
        try:
            self.llm = LLMEngine(model_name=llm_model, device=self.device)
        except Exception as e:
            logger.warning(f"Failed to initialize LLM: {str(e)}")
            logger.warning("Falling back to rule-based responses")
            self.llm = FallbackLLM()
        
        # Initialize CSM module
        logger.info("Initializing CSM module...")
        if csm_model_path is None:
            csm_model_path = hf_hub_download(repo_id="sesame/csm-1b", filename="ckpt.pt")
        
        self.generator = load_csm_1b(csm_model_path, self.device)
        logger.info("CSM model loaded")
        
        # Voice management
        self.speaker_id = DEFAULT_SPEAKER_ID
        self.active_voice = None
        self.voice_profiles = self._load_voice_profiles()
        if not self.voice_profiles:
            # Create default profile
            self._create_default_voice_profile()
        
        # Conversation history
        self.conversation_history = []
    
    def _load_voice_profiles(self) -> Dict[str, Dict]:
        """Load voice profiles from disk."""
        profiles = {}
        
        if not os.path.exists(self.voices_dir):
            return profiles
            
        for voice_name in os.listdir(self.voices_dir):
            voice_path = Path(self.voices_dir) / voice_name
            if voice_path.is_dir():
                # Look for profile.json
                profile_file = voice_path / "profile.json"
                if profile_file.exists():
                    try:
                        with open(profile_file, 'r') as f:
                            profile = json.load(f)
                            profiles[voice_name] = profile
                            
                            # Load samples
                            profile['samples'] = []
                            samples_dir = voice_path / "samples"
                            if samples_dir.exists():
                                for sample_file in samples_dir.glob("*.wav"):
                                    text_file = sample_file.with_suffix('.txt')
                                    if text_file.exists():
                                        with open(text_file, 'r') as tf:
                                            text = tf.read().strip()
                                            
                                        waveform, sample_rate = torchaudio.load(str(sample_file))
                                        profile['samples'].append({
                                            'text': text,
                                            'audio': waveform.squeeze(),
                                            'sample_rate': sample_rate
                                        })
                                        
                            logger.info(f"Loaded voice profile: {voice_name} with {len(profile['samples'])} samples")
                    except Exception as e:
                        logger.error(f"Error loading voice profile {voice_name}: {str(e)}")
        
        return profiles
    
    def _create_default_voice_profile(self):
        """Create a default voice profile."""
        name = "default"
        speaker_id = DEFAULT_SPEAKER_ID
        
        profile = {
            'name': name,
            'speaker_id': speaker_id,
            'description': "Default voice profile",
            'samples': []
        }
        
        # Create directory structure
        voice_path = Path(self.voices_dir) / name
        os.makedirs(voice_path, exist_ok=True)
        os.makedirs(voice_path / "samples", exist_ok=True)
        
        # Save profile
        with open(voice_path / "profile.json", 'w') as f:
            json.dump(profile, f, indent=2)
        
        self.voice_profiles[name] = profile
        self.active_voice = name
        self.speaker_id = speaker_id
        
        logger.info(f"Created default voice profile: {name}")
        
        # Add a dummy sample to avoid empty context issues
        try:
            logger.info("Adding a dummy sample to the default voice profile")
            # Use a default text and generate a short audio clip
            dummy_text = "Hello, this is the default voice."
            dummy_audio = torch.zeros(int(self.generator.sample_rate * 1.5))  # 1.5 seconds of silence
            
            # Create sample directory
            samples_dir = voice_path / "samples"
            os.makedirs(samples_dir, exist_ok=True)
            
            # Save audio file
            audio_path = samples_dir / "sample_0.wav"
            torchaudio.save(str(audio_path), dummy_audio.unsqueeze(0).unsqueeze(0), self.generator.sample_rate)
            
            # Save text
            text_path = samples_dir / "sample_0.txt"
            with open(text_path, 'w') as f:
                f.write(dummy_text)
            
            # Add to profile
            profile['samples'].append({
                'text': dummy_text,
                'audio': dummy_audio,
                'sample_rate': self.generator.sample_rate
            })
            
            logger.info("Added dummy sample to default voice profile")
        except Exception as e:
            logger.warning(f"Failed to add dummy sample: {str(e)}")
            # This is not critical, so we can continue
    
    def create_voice_profile(self, name: str, speaker_id: int = 0, description: str = "") -> str:
        """
        Create a new voice profile.
        
        Args:
            name: Name of the profile
            speaker_id: Speaker ID to use with CSM
            description: Optional description
            
        Returns:
            Profile name
        """
        if name in self.voice_profiles:
            raise ValueError(f"Voice profile '{name}' already exists")
        
        profile = {
            'name': name,
            'speaker_id': speaker_id,
            'description': description,
            'samples': []
        }
        
        # Create directory structure
        voice_path = Path(self.voices_dir) / name
        os.makedirs(voice_path, exist_ok=True)
        os.makedirs(voice_path / "samples", exist_ok=True)
        
        # Save profile
        with open(voice_path / "profile.json", 'w') as f:
            json.dump(profile, f, indent=2)
        
        self.voice_profiles[name] = profile
        logger.info(f"Created voice profile: {name}")
        
        return name
    
    def add_sample_to_voice(self, voice_name: str, text: str, audio: torch.Tensor, sample_rate: int) -> int:
        """
        Add a sample to a voice profile.
        
        Args:
            voice_name: Name of the voice profile
            text: Text of the sample
            audio: Audio tensor
            sample_rate: Sample rate of the audio
            
        Returns:
            Sample ID
        """
        if voice_name not in self.voice_profiles:
            raise ValueError(f"Voice profile '{voice_name}' does not exist")
        
        profile = self.voice_profiles[voice_name]
        sample_id = len(profile['samples'])
        
        # Save audio file
        samples_dir = Path(self.voices_dir) / voice_name / "samples"
        os.makedirs(samples_dir, exist_ok=True)
        
        audio_path = samples_dir / f"sample_{sample_id}.wav"
        torchaudio.save(str(audio_path), audio.unsqueeze(0), sample_rate)
        
        # Save text
        text_path = samples_dir / f"sample_{sample_id}.txt"
        with open(text_path, 'w') as f:
            f.write(text)
        
        # Add to profile
        profile['samples'].append({
            'text': text,
            'audio': audio,
            'sample_rate': sample_rate
        })
        
        logger.info(f"Added sample {sample_id} to voice profile {voice_name}")
        return sample_id
    
    def set_active_voice(self, voice_name: str):
        """Set the active voice profile."""
        if voice_name not in self.voice_profiles:
            raise ValueError(f"Voice profile '{voice_name}' does not exist")
        
        self.active_voice = voice_name
        self.speaker_id = self.voice_profiles[voice_name]['speaker_id']
        logger.info(f"Set active voice to {voice_name} (speaker_id: {self.speaker_id})")
    
    def get_voice_context(self, voice_name: Optional[str] = None) -> List[Segment]:
        """
        Get voice context for the CSM model.
        
        Args:
            voice_name: Name of the voice profile, or None for active voice
            
        Returns:
            List of Segment objects
        """
        if voice_name is None:
            voice_name = self.active_voice
            
        if not voice_name or voice_name not in self.voice_profiles:
            # Use the default voice profile if it exists
            if "default" in self.voice_profiles:
                logger.info(f"Using 'default' voice profile instead of '{voice_name}'")
                voice_name = "default"
                self.active_voice = "default"
            else:
                logger.warning(f"No valid voice profile found, creating a default profile")
                self._create_default_voice_profile()
                voice_name = "default"
                self.active_voice = "default"
        
        profile = self.voice_profiles[voice_name]
        context = []
        
        # Create segments from samples
        for sample in profile['samples']:
            # Resample to CSM rate if needed
            if sample['sample_rate'] != self.generator.sample_rate:
                audio = torchaudio.functional.resample(
                    sample['audio'], 
                    orig_freq=sample['sample_rate'], 
                    new_freq=self.generator.sample_rate
                )
            else:
                audio = sample['audio']
            
            segment = Segment(
                text=sample['text'],
                speaker=profile['speaker_id'],
                audio=audio
            )
            context.append(segment)
        
        return context
    
    def get_conversation_context(self, max_turns: int = 5) -> List[Segment]:
        """
        Get conversation context from history.
        
        Args:
            max_turns: Maximum number of turns to include
            
        Returns:
            List of Segment objects
        """
        context = []
        
        # Get recent conversation turns
        recent_turns = self.conversation_history[-max_turns*2:] if max_turns > 0 else []
        
        for turn in recent_turns:
            context.append(turn.to_segment())
        
        return context
    
    def record_user_speech(self) -> Tuple[np.ndarray, int]:
        """
        Record user speech using the audio recorder.
        
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        logger.info("Recording user speech...")
        audio_data = self.audio_recorder.start_recording()
        
        # Save temporary file for ASR (which works better with WAV files)
        temp_file = self.audio_recorder.save_recording(audio_data, "temp_recording.wav")
        
        return audio_data, self.audio_recorder.rate
    
    def transcribe_speech(self, audio: Union[str, np.ndarray, Tuple[np.ndarray, int]]) -> str:
        """
        Transcribe speech to text.
        
        Args:
            audio: Audio input (file path, numpy array, or tuple)
            
        Returns:
            Transcribed text
        """
        logger.info("Transcribing speech...")
        text = self.speech_recognizer.transcribe_audio(audio)
        logger.info(f"Transcription: {text}")
        
        return text
    
    def generate_text_response(self, user_input: str) -> str:
        """
        Generate text response using the LLM.
        
        Args:
            user_input: User's text input
            
        Returns:
            Generated text response
        """
        logger.info("Generating text response...")
        response = self.llm.generate_response(user_input)
        logger.info(f"Generated response: {response}")
        
        return response
    
    def synthesize_speech(self, text: str) -> Tuple[torch.Tensor, int]:
        """
        Synthesize speech from text using CSM.
        
        Args:
            text: Text to synthesize
            
        Returns:
            Tuple of (audio_tensor, sample_rate)
        """
        logger.info("Synthesizing speech...")
        
        # Prepare context
        context = []
        
        # Add voice profile samples for voice characteristics
        voice_context = self.get_voice_context()
        if voice_context:
            logger.info(f"Using voice context with {len(voice_context)} samples")
            context.extend(voice_context)
        else:
            logger.warning("No voice context available")
        
        # Add conversation history for continuity
        conversation_context = self.get_conversation_context()
        if conversation_context:
            logger.info(f"Using conversation context with {len(conversation_context)} segments")
            context.extend(conversation_context)
        else:
            logger.info("No conversation context available yet")
        
        # Try with reduced context length if we have a lot of context
        max_audio_length = MAX_AUDIO_LENGTH_MS
        if len(context) > 5:
            # If we have a lot of context, we might need to reduce the max audio length
            # to avoid OOM errors
            max_audio_length = 5000
            logger.info(f"Reducing max audio length to {max_audio_length}ms due to large context")
            
        # Generate audio
        try:
            logger.info(f"Generating audio with {len(context)} context segments, speaker ID {self.speaker_id}")
            audio = self.generator.generate(
                text=text,
                speaker=self.speaker_id,
                context=context,
                max_audio_length_ms=max_audio_length
            )
            logger.info("Audio generation successful")
        except Exception as e:
            logger.error(f"Error generating audio: {str(e)}")
            # Try again with empty context as a fallback
            logger.info("Retrying with empty context")
            audio = self.generator.generate(
                text=text,
                speaker=self.speaker_id,
                context=[],
                max_audio_length_ms=max_audio_length
            )
        
        return audio, self.generator.sample_rate
    
    def save_audio(self, audio: torch.Tensor, sample_rate: int, filename: str = "response.wav") -> str:
        """
        Save audio to a file.
        
        Args:
            audio: Audio tensor
            sample_rate: Sample rate
            filename: Output filename
            
        Returns:
            Path to the saved file
        """
        torchaudio.save(filename, audio.unsqueeze(0).cpu(), sample_rate)
        return filename
    
    def play_audio(self, audio: Union[str, torch.Tensor, Tuple[torch.Tensor, int]]):
        """
        Play audio through the speakers.
        
        Args:
            audio: Audio to play (file path, tensor, or tuple)
        """
        logger.info("Playing audio...")
        
        try:
            import sounddevice as sd
            
            if isinstance(audio, str):
                # Load from file
                waveform, sample_rate = torchaudio.load(audio)
                audio_array = waveform.squeeze().numpy()
            elif isinstance(audio, torch.Tensor):
                # Assume CSM sample rate
                audio_array = audio.cpu().numpy()
                sample_rate = self.generator.sample_rate
            else:
                # Tuple of (tensor, sample_rate)
                audio_array = audio[0].cpu().numpy()
                sample_rate = audio[1]
            
            # Play audio
            sd.play(audio_array, sample_rate)
            sd.wait()
            
        except ImportError:
            logger.warning("sounddevice not installed. Cannot play audio directly.")
            if isinstance(audio, str):
                logger.info(f"Audio saved to {audio}")
            else:
                # Save to file
                filename = "response.wav"
                if isinstance(audio, torch.Tensor):
                    torchaudio.save(filename, audio.unsqueeze(0).cpu(), self.generator.sample_rate)
                else:
                    torchaudio.save(filename, audio[0].unsqueeze(0).cpu(), audio[1])
                logger.info(f"Audio saved to {filename}")
    
    def process_turn(self, audio_input: Optional[Union[str, np.ndarray, Tuple[np.ndarray, int]]] = None) -> Dict:
        """
        Process a complete conversation turn from speech input to speech output.
        
        Args:
            audio_input: Audio input (None to record from microphone)
            
        Returns:
            Dictionary with results
        """
        # Record audio if not provided
        if audio_input is None:
            user_audio, audio_rate = self.record_user_speech()
        else:
            if isinstance(audio_input, str):
                # Load from file
                waveform, audio_rate = torchaudio.load(audio_input)
                user_audio = waveform.squeeze().numpy()
            elif isinstance(audio_input, np.ndarray):
                # Assume default rate
                user_audio = audio_input
                audio_rate = self.audio_recorder.rate
            else:
                # Tuple
                user_audio, audio_rate = audio_input
        
        # Convert to tensor for history
        user_audio_tensor = torch.from_numpy(user_audio).float()
        
        # Transcribe speech to text
        user_text = self.transcribe_speech((user_audio, audio_rate))
        
        # Generate text response
        assistant_text = self.generate_text_response(user_text)
        
        # Synthesize speech
        assistant_audio, assistant_rate = self.synthesize_speech(assistant_text)
        
        # Save to conversation history
        user_turn = ConversationTurn(
            role="user",
            text=user_text,
            audio=user_audio_tensor,
            speaker_id=1  # Always use 1 for user to differentiate
        )
        
        assistant_turn = ConversationTurn(
            role="assistant",
            text=assistant_text,
            audio=assistant_audio,
            speaker_id=self.speaker_id
        )
        
        self.conversation_history.append(user_turn)
        self.conversation_history.append(assistant_turn)
        
        # Limit history size
        if len(self.conversation_history) > 20:  # Keep last 10 turns
            self.conversation_history = self.conversation_history[-20:]
        
        # Play the audio (can be disabled)
        audio_file = self.save_audio(assistant_audio, assistant_rate)
        self.play_audio(audio_file)
        
        return {
            "user_audio": (user_audio, audio_rate),
            "user_text": user_text,
            "assistant_text": assistant_text,
            "assistant_audio": (assistant_audio, assistant_rate),
            "audio_file": audio_file
        }
    
    def conversation_loop(self):
        """Run an interactive conversation loop."""
        print("\n===== CSM Speech-to-Speech Interface =====")
        print(f"Using voice profile: {self.active_voice}")
        print("Press Ctrl+C to exit")
        print("======================================\n")
        
        try:
            turn_number = 1
            while True:
                print(f"\n--- Turn {turn_number} ---")
                print("Press Enter to speak...")
                input()
                
                result = self.process_turn()
                
                print(f"You: {result['user_text']}")
                print(f"AI: {result['assistant_text']}")
                
                turn_number += 1
                
        except KeyboardInterrupt:
            print("\nExiting conversation...")
            
        finally:
            # Clean up recorder
            self.audio_recorder.close()
            
            # Clean up temporary files
            for file in ["temp_recording.wav", "response.wav"]:
                if os.path.exists(file):
                    try:
                        os.remove(file)
                    except Exception:
                        pass
    
    def list_voice_profiles(self) -> List[str]:
        """List available voice profiles."""
        return list(self.voice_profiles.keys())


def main():
    """Main function to run the speech interface."""
    # Create and run the interface
    interface = SpeechInterface()
    
    # List available voices
    voices = interface.list_voice_profiles()
    if len(voices) > 1:
        print("Available voice profiles:")
        for i, voice in enumerate(voices):
            print(f"{i+1}. {voice}")
        
        choice = input("Select a voice (number) or press Enter for default: ")
        if choice.strip() and choice.isdigit() and 0 < int(choice) <= len(voices):
            interface.set_active_voice(voices[int(choice)-1])
    
    # Start conversation
    interface.conversation_loop()


if __name__ == "__main__":
    main()
