import numpy as np
import torch
from typing import Optional

from faster_whisper import WhisperModel

from .config import AudioConfig

class Transcriber:
    def __init__(self, config: AudioConfig):
        self.config = config
        self.model = WhisperModel(
            config.whisper_model,
            device=config.whisper_device,
            compute_type=config.whisper_compute_type,
        )

    def transcribe_segment(self, 
                           audio: np.ndarray, language: str
                          ) -> str:
        """Transcribe an audio array. Returns text."""
        
        audio = np.squeeze(audio).astype(np.float32)
        if len(audio) == 0:
            return ""

        segments_gen, _ = self.model.transcribe(audio, 
                                                language=language, 
                                                condition_on_previous_text=False
                                               )
        return " ".join(seg.text for seg in segments_gen).strip() 

    def run(self,
            wav: torch.Tensor,
            segments: list,
            lang_mapper: callable,
            ) -> None:
        """
        Transcribe each segment.

        Adds a text key to every segment dict.
        lang_mapper converts the segment's language code to a whisper language code.
        """
        for seg in segments:
            s, e = int(seg["start"]), int(seg["end"])
            audio_slice = wav[s:e].cpu().numpy()

            whisper_lang = lang_mapper(seg["language"])
            seg["text"] = self.transcribe_segment(audio_slice, whisper_lang)    