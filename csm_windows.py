"""
Windows-compatible startup script for CSM.
This disables Triton/Inductor backends and configures PyTorch to use CPU fallbacks.
"""

import os
import torch
import sys

# Configure PyTorch to skip Triton/Inductor
print("Configuring PyTorch for Windows compatibility...")

# Disable inductor backend
os.environ["TORCH_COMPILE_BACKEND"] = "eager"  # Use eager mode instead of inductor

# Tell PyTorch to suppress errors and fall back to eager mode
try:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
    print("PyTorch dynamo errors suppressed, will fall back to eager mode")
except ImportError:
    print("Could not configure torch._dynamo, this might cause errors")

# Disable cuda graph capture which can cause issues
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Print CUDA information
if torch.cuda.is_available():
    print(f"CUDA is available: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
else:
    print("CUDA is not available")

# Import and run the speech interface
print("Starting CSM Speech Interface...")
import run_speech_interface

# Run the main function
if __name__ == "__main__":
    run_speech_interface.main()
