"""
Each SLURM task (1 per GPU) receives a shard of audio files,
processes them through the pipeline, and writes results to .arrow
files incrementally — one batch per input file
"""

import argparse
import os
import glob
import time
import struct

import numpy as np
import pyarrow as pa

from audioprocessing.config import AudioConfig
from audioprocessing.pipeline import AudioProcessor


ARROW_SCHEMA = pa.schema([
    ("id", pa.int64()),
    ("filename", pa.string()),
    ("audio", pa.binary()),
    ("duration", pa.float64()),
    ("start", pa.float64()),
    ("end", pa.float64()),
    ("text", pa.string()),
    ("speaker", pa.int64()),
    ("language", pa.string()),
])


def find_audio_files(input_dir: str) -> list[str]:
    """
    Find all audio files in input_dir.
    """
    extensions = ("*.mp3", "*.wav", "*.flac", "*.ogg", "*.m4a", "*.opus")
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(input_dir, "**", ext), recursive=True))
    files.sort()  
    return files


def build_file_id_map(files: list[str]) -> dict[str, int]:
    """
    Assign a sequential ID to each file based on its sorted position.
    """
    return {fpath: idx for idx, fpath in enumerate(files)}


def shard_files(files: list[str], rank: int, world_size: int) -> list[str]:
    """Split files across ranks"""
    return files[rank::world_size]


def process_and_write(
                    processor: AudioProcessor,
                    files: list[str],
                    file_id_map: dict[str, int],
                    output_path: str,
                    rank: int,
                    sr: int,
                    ):
    """
    Process files one by one and stream results to a single .arrow file.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    total = len(files)
    total_rows = 0

    with pa.OSFile(output_path, "wb") as sink:
        writer = pa.ipc.new_file(sink, ARROW_SCHEMA)

        for i, fpath in enumerate(files):
            basename = os.path.basename(fpath)
            file_id = file_id_map[fpath]
            t0 = time.time()

            try:
                result = processor.process(fpath)
            except Exception as e:
                print(f"[Rank {rank}] ERROR {basename}: {e}", flush=True)
                continue

            elapsed = time.time() - t0
            speech_attrs = result[0].get("speech_attributes")
            file_duration = result[0].get("duration", 0.0)

            batch_rows = {
                "id": [],
                "filename": [],
                "audio": [],
                "duration": [],
                "start": [],
                "end": [],
                "text": [],
                "speaker": [],
                "language": [],
            }

            if speech_attrs:
                from audioprocessing.utils import load_audio
                wav, _ = load_audio(fpath, target_sr=sr)
                wav_np = wav.numpy()

                for seg in speech_attrs:
                    start_sample = int(seg["start"] * sr)
                    end_sample = int(seg["end"] * sr)
                    seg_audio = wav_np[start_sample:end_sample]
                    audio_bytes = seg_audio.astype(np.float32).tobytes()

                    batch_rows["id"].append(file_id)
                    batch_rows["filename"].append(basename)
                    batch_rows["audio"].append(audio_bytes)
                    batch_rows["duration"].append(seg.get("duration", 0.0))
                    batch_rows["start"].append(seg["start"])
                    batch_rows["end"].append(seg["end"])
                    batch_rows["text"].append(seg.get("text", ""))
                    batch_rows["speaker"].append(seg.get("speaker", -1))
                    batch_rows["language"].append(seg.get("language", ""))
            else:
                batch_rows["id"].append(file_id)
                batch_rows["filename"].append(basename)
                batch_rows["audio"].append(b"")
                batch_rows["duration"].append(file_duration)
                batch_rows["start"].append(0.0)
                batch_rows["end"].append(0.0)
                batch_rows["text"].append("")
                batch_rows["speaker"].append(-1)
                batch_rows["language"].append("")
                

            batch = pa.record_batch(batch_rows, schema=ARROW_SCHEMA)
            writer.write_batch(batch)
            n_segs = len(batch_rows["id"])
            total_rows += n_segs

            print(
                f"[Rank {rank}] ({i + 1}/{total}) {basename} — "
                f"{file_duration:.1f}s audio, {n_segs} segments, "
                f"{elapsed:.1f}s",
                flush=True,
            )

        writer.close()

    print(
        f"[Rank {rank}] Done — wrote {total_rows} rows to {output_path}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description="audio processing worker")
    parser.add_argument("--input-dir", required=True, help="Root directory with audio folders")
    parser.add_argument("--output-dir", required=True, help="Output directory for .arrow files")
    parser.add_argument("--language", type=str, default=None, help="Known language (skip LID)")
    parser.add_argument("--num-speakers", type=int, default=None)
    parser.add_argument("--no-diarization", action="store_true")
    args = parser.parse_args()

    rank = int(os.environ.get("SLURM_PROCID", 0))
    world_size = int(os.environ.get("SLURM_NTASKS", 1))
    local_rank = int(os.environ.get("SLURM_LOCALID", 0))

    os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)

    print(
        f"[Rank {rank}] Starting — world_size={world_size}, "
        f"local_rank={local_rank}, GPU={local_rank}",
        flush=True,
    )

    all_files = find_audio_files(args.input_dir)
    file_id_map = build_file_id_map(all_files)

    my_files = shard_files(all_files, rank, world_size)
    print(
        f"[Rank {rank}] {len(all_files)} total files, {len(my_files)} assigned",
        flush=True,
    )

    if not my_files:
        print(f"[Rank {rank}] No files to process, exiting.", flush=True)
        return

    config = AudioConfig(
        language=args.language,
        num_speakers=args.num_speakers,
        enable_diarization=not args.no_diarization,
        enable_language_id=(args.language is None),
        enable_transcription=True,
        enable_annotation=False,
    )

    t0 = time.time()
    processor = AudioProcessor(config)
    print(f"[Rank {rank}] Pipeline initialised in {time.time() - t0:.1f}s", flush=True)

    output_path = os.path.join(args.output_dir, f"shard_{rank:05d}.arrow")
    process_and_write(processor, my_files, file_id_map, output_path, rank, config.sample_rate)


if __name__ == "__main__":
    main()