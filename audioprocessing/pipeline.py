import torchaudio
import sys
import os

if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["soundfile"]
for _mod in ("transformer_engine", "modelopt", "torchao"):
    if _mod not in sys.modules:
        sys.modules[_mod] = None


import torch
import numpy as np
from typing import Union, Optional, Dict, Any, List

from .config import AudioConfig
from .utils import load_audio, classify_audio
from .segment_ops import (
    pre_merge,
    filter_short,
    initialise_metadata,
    merge_segments,
    format_output,
)

class AudioProcessor:
    """
    Audio processing pipeline.
    
    VAD + Pre-merge + Filter short segments -> Classification (Speech/Non-Speech) \
    (Speech) Diarization -> LID -> Re-merge using language + speaker -> transcription -> output
    (Non-speech) -> annotation -> output
    """
    def __init__(self, config: Optional[AudioConfig] = None):
        self.config = config or AudioConfig()
        if self.config.device == "cuda" and not torch.cuda.is_available():
            self.config.device = "cpu"
        
        # set cache directory
        os.makedirs(self.config.cache_dir, exist_ok=True)
        torch.hub.set_dir(self.config.cache_dir)

        # Initialise the models
        # ----- VAD -----
        self.vad = None
        if self.config.enable_vad:
            from .vad import VADModel
            self.vad = VADModel(self.config)

        # ----- Diarizater -----
        self.diarizer = None
        if self.config.enable_diarization:
            from .speaker import SpeakerDiarizer
            self.diarizer = SpeakerDiarizer(self.config)

        # ----- LID -----
        self.lid = None
        if self.config.enable_language_id:
            from .language import LanguageIdentifier
            self.lid = LanguageIdentifier(self.config)

        # ----- Transcriber -----
        self.transcriber = None
        if self.config.enable_transcription:
            from .transcription import Transcriber
            self.transcriber = Transcriber(self.config)

        # ----- Annotation -----
        self.annotator = None
        if self.config.enable_annotation:
            from .annotation import NonSpeechAnnotator
            self.annotator = NonSpeechAnnotator(self.config)
    
    def process(self,
                audio: Union[str, np.ndarray, torch.Tensor],
                sr: Optional[int] = None,
                **overrides,
            ) -> List[Dict[str, Any]]:
        
        cfg = self._apply_overrides(overrides)

        wav, original_sr = load_audio(audio, target_sr=cfg.sample_rate, sr=sr)

        # ----- VAD -----
        if self.vad and cfg.enable_vad:
            vad_segments = self.vad(wav)
        else:
            vad_segments = [{"start": 0, "end": wav.numel()}]
        
        # ----- Pre merge and filtering -----
        gap_samples = int(cfg.pre_merge_gap * cfg.sample_rate)
        segments = pre_merge(vad_segments, max_gap_samples=gap_samples)  

        min_samples = int(cfg.min_segment_duration * cfg.sample_rate)
        segments = filter_short(segments, min_samples=min_samples)  

        if not segments:
            return self._build_output(wav, cfg, "non_speech", 0)

        # ----- Classification -----
        audio_type = classify_audio(wav, segments, cfg.speech_ratio_threshold)

        if audio_type == "speech":
            initialise_metadata(segments, language=cfg.language)

            # ----- Diarization -----
            if self.diarizer and cfg.enable_diarization:
                self.diarizer.run(
                    wav, 
                    segments, 
                    sr=cfg.sample_rate, 
                    num_speakers=cfg.num_speakers,
                )

            # ----- Language identification -----
            if self.lid and cfg.enable_language_id:
                self.lid.run(wav, segments)
            
            #----- Re-merge -----
            merge_gap = int(cfg.merge_threshold_duration * cfg.sample_rate)
            merge_max = int(cfg.max_merge_duration * cfg.sample_rate)
            segments = merge_segments(
                segments,
                max_gap_samples=merge_gap,
                max_duration_samples=merge_max,
                keys=("speaker", "language"),
            )

            #----- Transcription -----
            if self.transcriber and cfg.enable_transcription:
                lang_mapper = self.lid.map_language if self.lid else lambda x: x
                self.transcriber.run(wav, segments, lang_mapper)

            speech_attr = format_output(segments, cfg.sample_rate)
            return self._build_output(
                wav, cfg, "speech", speech_attributes=speech_attr
            )

        non_speech_attr = None
        if self.annotator and cfg.enable_annotation:
            non_speech_attr = self.annotator.run(wav)

        return self._build_output(
            wav, cfg, "non_speech", non_speech_attributes=non_speech_attr
        )


    def _apply_overrides(self, overrides: dict) -> AudioConfig:
        """
        Return an AudioConfig to overcome overrides.
        """
        if not overrides:
            return self.config
        from dataclasses import asdict

        base = asdict(self.config)
        base.update({k: v for k, v in overrides.items() if k in base})
        return AudioConfig(**base)

    @staticmethod
    def _build_output(wav: torch.Tensor,
                    cfg: AudioConfig,
                    audio_type: str,
                    speech_attributes: Optional[list] = None,
                    non_speech_attributes: Optional[str] = None,
                    quality: Optional[int] = -1
                ) -> List[Dict[str, Any]]:
        
        return [{"audio_type": audio_type,
                "duration": wav.numel() / cfg.sample_rate,
                "sample_rate": cfg.sample_rate,
                "speech_attributes": speech_attributes,
                "non_speech_attributes": non_speech_attributes,
                "quality" : quality
                }]
