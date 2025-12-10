import torch
from birdy.loader.audio import load_audio
from birdy.processor.audio import AudioToMel
from birdy.helper.device import get_best_device


def main():
    # For now config will be here, but I will be moving to a config class
    # later on
    SAMPLE_RATE = 3200
    N_FFT = 2048
    HOP_LENGTH = 512
    N_MELS = 128
    TOP_DB = 80
    FILE_PATH = ""
    DEVICE = get_best_device()

    # 1. Setup AudioToMel
    audio_processor = AudioToMel(
        n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS, top_db=TOP_DB, device=DEVICE
    )

    # 2. Load Audio
    audio_tensor = load_audio(
        file_path=FILE_PATH, desired_sample_rate=SAMPLE_RATE, device=DEVICE
    )

    # 3. Process audio
    mel_spectogram = audio_processor(audio_tensor)


if __name__ == "__main__":
    main()
