"""
Non-speech audio annotation using Qwen3-Omni.

Lazy-loaded: the heavy model is only instantiated when `run()` is called.
"""

import torch
from typing import Optional

from .config import AudioConfig


NON_SPEECH_PROMPT = """
You are an expert audio annotator. Generate a concise, objective caption (1 to 3 sentences) describing the provided audio.

CRITICAL INSTRUCTIONS:
- STRICT GROUNDING: Describe ONLY sounds, environments, or instruments that are explicitly and unmistakably audible. Do not guess, infer, or hallucinate elements. If there is no music, do not mention music.
- NO FILLER: Begin the first sentence directly with a noun, article (A/An/The), or adjective. Never use introductory padding like "The audio features," "In this clip," "I can hear," or "This is a recording of."
- DENSITY: Focus strictly on the acoustic environment, discrete sound events, and distinct musical characteristics (if undeniably present).

EXAMPLES:
Good: "A bustling city street with distant sirens. Heavy traffic hums steadily in the background."
Good: "Upbeat electronic music featuring a driving bassline and a synthesized drum machine."
Bad: "In this clip, I can hear a quiet room. There might be a dog barking outside." (Contains filler and guessing).

Caption:
""".strip()


class NonSpeechAnnotator:
    """Lazy-loaded Qwen3-Omni wrapper for non-speech captioning."""

    def __init__(self, config: AudioConfig):
        self.config = config
        self._model = None
        self._processor = None

    def _load(self) -> None:
        """Load model on first use."""
        if self._model is not None:
            return

        from transformers import (
            Qwen3OmniMoeForConditionalGeneration,
            Qwen3OmniMoeProcessor,
        )

        path = self.config.annotation_model_path or self.config.cache_dir
        self._model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            path,
            dtype="auto",
            device_map="auto",
            attn_implementation="flash_attention_2",
        )
        self._processor = Qwen3OmniMoeProcessor.from_pretrained(path)

    def run(self, wav: torch.Tensor) -> str:
        """Generate a non-speech caption for the full waveform."""
        self._load()
        model, processor = self._model, self._processor

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": "in_memory_tensor"},
                    {"type": "text", "text": NON_SPEECH_PROMPT},
                ],
            },
        ]

        text_input = processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        audios = [wav.squeeze().cpu().numpy()]

        inputs = processor(
            text=text_input,
            audio=audios,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=False,
        ).to(model.device).to(model.dtype)

        text_ids, _ = model.generate(
            **inputs, thinker_return_dict_in_generate=True
        )

        generated = processor.batch_decode(
            text_ids.sequences[:, inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return generated