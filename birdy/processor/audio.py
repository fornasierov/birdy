import torch
import torch.nn as nn
import torchaudio.transforms as T
from PIL import Image
import numpy as np


class AudioToMel(nn.Module):
    """
    Docstring for AudioToMel
    """

    def __init__(
        self,
        n_fft: int,
        hop_length: int,
        n_mels: int,
        top_db: float,
        device: torch.device,
    ):
        super().__init__()
        self.device = device

        self.mel_spec = T.MelSpectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            normalized=True,
        ).to(device)

        self.amplitude_to_db = T.AmplitudeToDB(top_db=top_db).to(device)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Docstring for forward

        """

        spectogram = self.mel_spec(waveform)
        log_spectogram = self.amplitude_to_db(spectogram)
        return log_spectogram

    def mel_to_image(self, mel_tensor: torch.Tensor) -> Image.Image:
        """
        Convert a mel spectrogram tensor to a PIL Image.
        """
        if mel_tensor.dim() == 3:
            mel_tensor = mel_tensor.squeeze(0)

        mel_min = mel_tensor.min()
        mel_max = mel_tensor.max()
        if mel_max > mel_min:
            mel_normalized = (mel_tensor - mel_min) / (mel_max - mel_min) * 255
        else:
            mel_normalized = mel_tensor * 255

        mel_normalized = mel_normalized.to(torch.uint8)

        # Transpose to (time_steps, n_mels) for proper image orientation
        mel_normalized = mel_normalized.transpose(0, 1)

        # Move to CPU and convert to numpy only for PIL Image creation
        mel_np = mel_normalized.cpu().detach().numpy()

        # Create PIL Image
        image = Image.fromarray(mel_np, mode="L")

        return image
