import torchaudio
import torch
import soundfile as sf
import numpy as np

from typing import Union, Optional, Tuple


def load_audio(
            audio: Union[str, np.ndarray, torch.Tensor],
            target_sr: int,
            sr: Optional[int] = None,
        ) -> Tuple[torch.Tensor, int]:
    """
    Load audio from file / array / tensor convert to mono andresample.
    """
    if isinstance(audio, str):
        wav, sr = sf.read(audio)
        wav = torch.from_numpy(wav).float()
    elif isinstance(audio, np.ndarray):
        wav = torch.from_numpy(audio).float()
    elif isinstance(audio, torch.Tensor):
        wav = audio.float()
    else:
        raise TypeError(f"Unsupported audio type: {type(audio)}")

    if sr is None:
        raise ValueError("Sample rate must be provided for non-file inputs.")

    if wav.ndim > 1:
        wav = wav.mean(dim=1)

    original_sr = sr

    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=target_sr)

    return wav.squeeze(0), original_sr


def classify_audio(
                    wav: torch.Tensor,
                    vad_segments: list,
                    ratio_threshold: float = 0.05,
                ) -> str:
    """Return speech or non_speech based on VAD coverage ratio."""
    duration = wav.numel()
    if duration == 0:
        return "non_speech"
    speech_duration = sum(seg["end"] - seg["start"] for seg in vad_segments)
    return "speech" if (speech_duration / duration) >= ratio_threshold else "non_speech"