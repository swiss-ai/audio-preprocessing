from dataclasses import dataclass
from typing import Optional

@dataclass
class AudioConfig:
    # "cuda" or "cpu" resolved at runtime
    device: str = "cuda"     

    # ----- Paths -----
    cache_dir: str = "/capstor/store/cscs/swissai/infra01/MLLM/audioprocessing"
    # maps ISO code of LID and Transcriber model
    lang_mapper_path: str = "mms_to_whisper.json"

    # ----- VAD -----
    sample_rate: int = 16000                # in Hz
    # Fraction of audio to be identified as speech
    speech_ratio_threshold: float = 0.05
    # VAD filters for merging and dropping segments that are short
    pre_merge_gap: float = 0.2              # in seconds
    min_segment_duration: float = 0.5       # in seconds

    # ----- Speaker diarization -----
    speaker_model_name: str = "titanet_large"
    max_speakers: int = 10
    # set num_speakers if known to skip estimation
    num_speakers: Optional[int] = None 
    # min duration for considering VAD segment for speaker identification 
    valid_vad_segment_min_dur: float = 1.0           # in seconds
    embedding_batch_size: int = 32  
    smoothing_max_dur: float = 0.1                   # in seconds
    diarization_window: float = 1.5                  # in seconds 
    diarization_overlap: float = 0.75                # in seconds

    # ----- Language identification -----
    lid_model_name: str = "facebook/mms-lid-126"
    # set language if know to skip LID
    language: Optional[str] = None  
    lid_merge_duration: float = 30.0                # in seconds

    # ----- Transcription -----
    whisper_model: str = "large-v3"
    whisper_device: str = "cuda"
    whisper_compute_type: str = "float16"

    # ----- Re-merge -----
    # merge segments if separation between them is less than threshold
    merge_threshold_duration: float = 2.0           # in seconds
    # merge until max_merge_duration
    max_merge_duration: float = 10.0                # in seconds

    # ----- Annotation (Non-speech only) -----
    annotation_model_path: Optional[str] = None

    # ----- Conditions -----
    enable_vad: bool = True
    enable_diarization: bool = True
    enable_language_id: bool = True
    enable_transcription: bool = True
    enable_annotation: bool = True