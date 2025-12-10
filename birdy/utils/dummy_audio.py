import torchaudio
import torch
from pathlib import Path

sample_rate = 16000
duration = 1
waveform = torch.randn(1, sample_rate * duration)

# Save to data folder
data_dir = Path(__file__).parent.parent.parent / "data"
data_dir.mkdir(parents=True, exist_ok=True)
output_path = data_dir / "dummy.wav"
torchaudio.save(str(output_path), waveform, sample_rate)
