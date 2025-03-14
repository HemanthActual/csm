import torch
import torchaudio
from huggingface_hub import hf_hub_download
from generator import load_csm_1b

# Print GPU information
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"CUDA Version: {torch.version.cuda}")
print(f"PyTorch Version: {torch.__version__}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
else:
    print("No GPU detected. The model will run on CPU which will be very slow.")
    print("Check your CUDA installation.")

# Simple test to ensure CUDA works
if torch.cuda.is_available():
    print("\nTesting CUDA with a simple matrix multiplication...")
    a = torch.randn(1000, 1000).cuda()
    b = torch.randn(1000, 1000).cuda()
    
    # Warmup
    torch.matmul(a, b)
    torch.cuda.synchronize()
    
    # Timing
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    c = torch.matmul(a, b)
    end.record()
    
    torch.cuda.synchronize()
    print(f"Matrix multiplication time: {start.elapsed_time(end):.2f} ms")
    print("CUDA is working properly!\n")

# Set device based on availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Try to download the model
print("Downloading the model (if not already downloaded)...")
try:
    model_path = hf_hub_download(repo_id="sesame/csm-1b", filename="ckpt.pt")
    print(f"Model downloaded successfully to: {model_path}")
except Exception as e:
    print(f"Error downloading model: {e}")
    exit(1)

# Try to load the model on the selected device
print(f"\nLoading model on {device}...")
try:
    generator = load_csm_1b(model_path, device)
    print("Model loaded successfully!")
    print(f"Model is on device: {next(generator._model.parameters()).device}")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Try running with device='cpu' if you continue to have issues.")
    exit(1)

# Try generating a short audio sample
print("\nGenerating a short audio sample...")
try:
    audio = generator.generate(
        text="Hello, this is a test of the CSM model on GPU.",
        speaker=0,
        context=[],
        max_audio_length_ms=3000,  # Keep it short for testing
    )
    
    output_file = "gpu_test_output.wav"
    torchaudio.save(output_file, audio.unsqueeze(0).cpu(), generator.sample_rate)
    print(f"Audio generation successful! Saved to: {output_file}")
    print("Your GPU setup is working correctly with CSM!")
except Exception as e:
    print(f"Error generating audio: {e}")
    print("\nTroubleshooting tips:")
    print("1. Check if your GPU has enough memory")
    print("2. Try with device='cpu' to see if it's a GPU-specific issue")
    print("3. Ensure all dependencies are correctly installed")
