# Audio Processing Pipeline

Speech processing pipeline for VAD, speaker diarization, language identification, and transcription. Built on Silero VAD, NeMo TitaNet MMS LID and faster-whisper.

## Setup

### Environment

On Clariden, use the environment:

```bash
--environment=ctranslate2-nemo-cudnn
```

### Installation

```bash
make
```

This creates the virtual environment and installs all dependencies.

### Activate

```bash
source .venv-audioprocessing/bin/activate
```

## Usage

### Single file

```bash
python main.py "path/to/audio.wav"
```

### Options

| Flag | Description |
|---|---|
| `--output`, `-o` | Output JSON path (default: `output.json`) |
| `--num-speakers` | Known speaker count (skips estimation) |
| `--language` | Known language code (skips LID) |
| `--no-diarization` | Disable speaker diarization |
| `--no-lid` | Disable language identification |
| `--no-transcription` | Disable transcription |
| `--no-annotation` | Disable non-speech annotation |

## Pipeline

```
load_audio → VAD → pre-merge → filter short → classify
                                                  │
                          ┌───────────────────────┴────────────────┐
                      speech                                  non-speech
                          │                                        │
                    diarization                              annotation
                     (windowed if < 12 segments)
                          │
                        LID
                          │
                   final merge (by speaker + language)
                          │
                    transcription
                          │
                       output
```

## Configuration

All parameters live in `audioprocessing/config.py`. Override via CLI flags or by passing a modified `AudioConfig` to `AudioProcessor`.

### VAD

| Parameter | Default | Description |
|---|---|---|
| `speech_ratio_threshold` | `0.05` | Min speech ratio to classify audio as speech |
| `pre_merge_gap` | `0.2s` | Merge VAD fragments closer than this |
| `min_segment_duration` | `0.5s` | Drop segments shorter than this |

### Speaker diarization

| Parameter | Default | Description |
|---|---|---|
| `speaker_model_name` | `titanet_large` | NeMo speaker embedding model |
| `max_speakers` | `10` | Upper bound for speaker estimation |
| `num_speakers` | `None` | Set if known (skips NME-SC estimation) |
| `valid_vad_segment_min_dur` | `1.0s` | Long/short segment split threshold |
| `embedding_batch_size` | `32` | Batch size for TitaNet inference |
| `diarization_window` | `1.5s` | Sliding window length for embeddings |
| `diarization_overlap` | `0.75s` | Overlap between consecutive windows |
| `smoothing_max_dur` | `1.0s` | Smooth isolated segments shorter than this |

When VAD produces fewer than 12 segments, long segments are automatically split into 1.5s overlapping windows before embedding extraction. This ensures NME-SC has enough per-speaker embeddings for reliable clustering. Window-level labels are mapped back to original segments via majority vote.

### Language identification

| Parameter | Default | Description |
|---|---|---|
| `lid_model_name` | `facebook/mms-lid-126` | MMS language ID model |
| `language` | `None` | Set if known (skips LID) |
| `lid_merge_duration` | `30.0s` | Audio stitched per speaker for LID |

### Transcription

| Parameter | Default | Description |
|---|---|---|
| `whisper_model` | `large-v3` | Whisper model via CTranslate2 |
| `whisper_compute_type` | `float16` | Compute precision |

### Final merge

| Parameter | Default | Description |
|---|---|---|
| `merge_threshold_duration` | `1.0s` | Max gap to merge across |
| `max_merge_duration` | `10.0s` | Max combined segment length |

Final merge only combines consecutive segments with the same speaker and language.

## Project structure

```
preprocessor/
├── main.py                     # CLI 
├── mms_to_whisper.json         # ISO code conversion for LID to whisper
├── Makefile
├── audioprocessing/
    ├── __init__.py
    ├── config.py               # AudioConfig dataclass
    ├── pipeline.py             # AudioProcessor 
    ├── audio_utils.py          # Load, resample, classification
    ├── vad.py                  # Silero VAD wrapper
    ├── speaker.py              # TitaNet embeddings + NeMo NME-SC clustering
    ├── language.py             # MMS language identification
    ├── transcription.py        # Whisper transcription
    ├── segment_ops.py          # Merge, filter format operations
    └── annotation.py           # Non-speech annotation
└── sample/
    ├── multispeaker.mp3
    └── output.json
```