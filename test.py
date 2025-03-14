from huggingface_hub import hf_hub_download
from generator import load_csm_1b
import torchaudio
import torch

model_path = hf_hub_download(repo_id="sesame/csm-1b", filename="ckpt.pt")
generator = load_csm_1b(model_path, "gpu") # could use cpu
audio = generator.generate(
    text="the green fox jumps over the lazy dog",
    speaker=0,
    context=[],
    max_audio_length_ms=10_000,
)

torchaudio.save("audio.wav", audio.unsqueeze(0).cpu(), generator.sample_rate)

print(torch.cuda.is_available())  # Should print True
print(torch.version.cuda)  # Should show the CUDA version