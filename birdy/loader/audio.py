import torch
import torchaudio


def load_audio(
    file_path: str,
    desired_sample_rate: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Docstring for load_audio
    """
    waveform, incoming_sample_rate = torchaudio.load(file_path)

    waveform = waveform.to(device)

    if incoming_sample_rate != desired_sample_rate:
        resampler = torchaudio.transforms.Resample(
            incoming_sample_rate, desired_sample_rate
        ).to(waveform.device)
        waveform = resampler(waveform)

    return waveform
