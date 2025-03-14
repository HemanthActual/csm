"""
Speech recognition module for CSM Speech-to-Speech interface.
Uses Whisper for transcription.
"""

import os
import torch
import torchaudio
import numpy as np
import wave
import pyaudio
from dataclasses import dataclass
from typing import Tuple, Optional, Union
from transformers import pipeline

# Constants for audio recording
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000  # Whisper works best with 16kHz audio
THRESHOLD = 0.03  # For detecting silence
SILENCE_DURATION = 1.5  # Seconds of silence to stop recording


class AudioRecorder:
    """Records audio from microphone until stopped."""
    
    def __init__(self, 
                 rate: int = RATE,
                 chunk: int = CHUNK,
                 channels: int = CHANNELS,
                 format: int = FORMAT):
        self.rate = rate
        self.chunk = chunk
        self.channels = channels
        self.format = format
        self.recording = False
        self.audio_interface = pyaudio.PyAudio()
        
    def start_recording(self, silence_threshold: float = THRESHOLD, 
                        silence_duration: float = SILENCE_DURATION) -> np.ndarray:
        """
        Record audio until silence is detected.
        Returns the audio as a numpy array.
        """
        # Open stream
        stream = self.audio_interface.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk
        )
        
        print("Recording... (speak now)")
        frames = []
        self.recording = True
        silence_frames = 0
        silence_threshold_frames = int(silence_duration * self.rate / self.chunk)
        
        while self.recording:
            data = stream.read(self.chunk)
            frames.append(data)
            
            # Check for silence to auto-stop
            audio_data = np.frombuffer(data, dtype=np.int16)
            volume_norm = np.abs(audio_data).mean() / 32768.0
            
            if volume_norm < silence_threshold:
                silence_frames += 1
                if silence_frames >= silence_threshold_frames:
                    print("Silence detected, stopping recording.")
                    self.recording = False
            else:
                silence_frames = 0
                
        print("Recording stopped.")
        stream.stop_stream()
        stream.close()
        
        # Convert frames to numpy array
        audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
        audio_data = audio_data.astype(np.float32) / 32768.0  # Normalize to [-1.0, 1.0]
        
        return audio_data
    
    def save_recording(self, audio_data: np.ndarray, filename: str = "recording.wav") -> str:
        """Save recording to a WAV file and return the path."""
        # Convert back to int16 for WAV file
        audio_int16 = (audio_data * 32768.0).astype(np.int16)
        
        # Save as WAV
        wf = wave.open(filename, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(self.audio_interface.get_sample_size(self.format))
        wf.setframerate(self.rate)
        wf.writeframes(audio_int16.tobytes())
        wf.close()
        
        return filename
    
    def stop_recording(self):
        """Stop the current recording."""
        self.recording = False
    
    def close(self):
        """Close the PyAudio interface."""
        self.audio_interface.terminate()


class SpeechRecognizer:
    """Speech recognition using Whisper."""
    
    def __init__(self, model_name: str = "openai/whisper-base"):
        """
        Initialize the speech recognizer.
        
        Args:
            model_name: Name of the Whisper model variant to use.
                Options: tiny, base, small, medium, large
        """
        self.model_name = model_name
        print(f"Loading ASR model: {model_name}")
        
        # Initialize the ASR model
        self.asr = pipeline(
            "automatic-speech-recognition", 
            model=model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        print("ASR model loaded.")
    
    def transcribe_audio(self, audio: Union[str, np.ndarray, Tuple[np.ndarray, int]]) -> str:
        """
        Transcribe audio to text.
        
        Args:
            audio: Either a path to an audio file, a numpy array of audio data,
                  or a tuple of (audio_array, sample_rate)
                  
        Returns:
            The transcribed text
        """
        if isinstance(audio, str):
            # Load audio from file
            waveform, sample_rate = torchaudio.load(audio)
            audio_array = waveform.squeeze().numpy()
            input_features = {"array": audio_array, "sampling_rate": sample_rate}
        elif isinstance(audio, np.ndarray):
            # Assume 16kHz sample rate for raw numpy arrays
            input_features = {"array": audio, "sampling_rate": RATE}
        else:
            # Tuple of (audio_array, sample_rate)
            audio_array, sample_rate = audio
            input_features = {"array": audio_array, "sampling_rate": sample_rate}
        
        # Perform transcription
        result = self.asr(input_features)
        transcription = result["text"].strip()
        
        return transcription


def test_asr():
    """Test the ASR functionality."""
    print("Testing Audio Speech Recognition")
    
    recorder = AudioRecorder()
    recognizer = SpeechRecognizer()
    
    try:
        print("Press Enter to start recording...")
        input()
        
        # Record audio
        audio_data = recorder.start_recording()
        
        # Save to file
        audio_file = recorder.save_recording()
        
        # Transcribe
        print("Transcribing...")
        text = recognizer.transcribe_audio(audio_file)
        
        print(f"Transcription: {text}")
        
    finally:
        recorder.close()


if __name__ == "__main__":
    test_asr()
