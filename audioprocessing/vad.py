import torch
from .config import AudioConfig

class VADModel:
    """Uses Silero VAD."""

    def __init__(self, config: AudioConfig):
        self.sr = config.sample_rate
        torch.hub.set_dir(config.cache_dir)

        vad_model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=True,
        )
        self._get_timestamps = utils[0]
        self._model = vad_model

    def __call__(self, wav: torch.Tensor) -> list:
        """
        Run VAD and return list of segment dicts with 'start' / 'end' in samples.
        """
        return self._get_timestamps(wav, self._model, sampling_rate=self.sr)