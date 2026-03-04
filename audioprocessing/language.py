import json
import numpy as np
import torch
from typing import Optional, Tuple, Dict

from transformers import Wav2Vec2ForSequenceClassification, AutoFeatureExtractor

from .config import AudioConfig


class LanguageIdentifier:

    def __init__(self, config: AudioConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.sr = config.sample_rate

        if config.language is None:
            self.processor = AutoFeatureExtractor.from_pretrained(config.lid_model_name)
            self.model = Wav2Vec2ForSequenceClassification.from_pretrained(
                config.lid_model_name
            ).to(self.device)
            self.model.eval()
        else:
            self.processor = None
            self.model = None

        # MMS language code to Whisper language code
        with open(config.lang_mapper_path, "r") as f:
            self.mapper: dict = json.load(f)
    
    def detect(self, audio: torch.Tensor) -> Tuple[str, float]:
        """
        Run LID on a waveform tensor. 
        Returns (lang_code, confidence).
        """
        inputs = self.processor(
            audio.cpu().numpy(), sampling_rate=self.sr, return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.model(**inputs).logits

        probs = torch.softmax(logits, dim=-1)
        confidence, lang_id = torch.max(probs, dim=-1)
        lang = self.model.config.id2label[lang_id.item()]
        return lang, confidence.item()

    def detect_per_speaker(self,
                           wav: torch.Tensor,
                           segments: list,
                          ) -> Dict[int, Tuple[str, float]]:
        """
        For each unique speaker label, stitch their segments up to
        lid_merge_duration seconds and run LID only once.

        Returns {speaker_id: (lang_code, confidence)}.
        """
        lid_samples = self.config.lid_merge_duration * self.sr
        speakers = sorted(set(seg["speaker"] for seg in segments))
        speaker_langs: Dict[int, Tuple[str, float]] = {}

        for spk in speakers:
            chunks = []
            total = 0
            for seg in segments:
                if seg["speaker"] != spk:
                    continue
                chunk = wav[int(seg["start"]) : int(seg["end"])]
                chunks.append(chunk)
                total += chunk.shape[0]
                if total >= lid_samples:
                    break

            if chunks:
                combined = torch.cat(chunks)
                lang, score = self.detect(combined)
                speaker_langs[spk] = (lang, score)
            else:
                speaker_langs[spk] = ("unknown", 0.0)

        return speaker_langs

    # ------------------------------------------------------------------

    def run(self, wav: torch.Tensor, segments: list) -> None:
        """
        Full LID pass — updates each segment's language and score

        If a known language was supplied via config, all segments are
        assigned that language with score 1.0 and the model is not called.
        """
        known = self.config.language
        if known is not None:
            for seg in segments:
                seg["language"] = known
                seg["score"] = 1.0
            return

        speaker_langs = self.detect_per_speaker(wav, segments)
        for seg in segments:
            lang, score = speaker_langs.get(seg["speaker"], ("unknown", 0.0))
            seg["language"] = lang
            seg["score"] = score

    def map_language(self, mms_code: str) -> str:
        return self.mapper.get(mms_code, mms_code)