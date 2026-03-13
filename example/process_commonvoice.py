import argparse
import gzip
import json
import logging
import os
import subprocess
import shutil
import time
import threading
from collections import deque
from pathlib import Path
import queue
import concurrent.futures

os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("SLURM_LOCALID", "0")

import torch
import numpy as np

from faster_whisper import WhisperModel, BatchedInferencePipeline
from whisper_normalizer.basic import BasicTextNormalizer

from lhotse import CutSet

import jiwer

_RANK = os.environ.get("SLURM_PROCID", "0")

logging.basicConfig(
    format=f"%(asctime)s [Rank {_RANK}] %(levelname)s: %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

logging.getLogger("faster_whisper").setLevel(logging.WARNING)


LANG_MAP = {
    "nb-NO": "no",
    "zh-HK": "zh",
    "hy-AM": "hy",
    "zh-CN": "zh",
    "zh-TW": "zh",
    "sv-SE": "sv",
}

WHISPER_UNSUPPORTED = {"rm-sursilv", "eo", "ga-IE", "nn-NO"}


def parse_language(subdir_name: str) -> str:
    return subdir_name.rsplit("_", 1)[0]


def get_whisper_language(lang_code: str) -> str | None:
    if lang_code in WHISPER_UNSUPPORTED:
        return None
    return LANG_MAP.get(lang_code, lang_code)

def is_subdir_completed(output_dir: Path, subdir_name: str) -> bool:
    return (output_dir / subdir_name / ".done").exists()

def mark_subdir_completed(output_dir: Path, subdir_name: str, stats: dict):
    with open(output_dir / subdir_name / ".done", "w") as f:
        json.dump(stats, f, indent=2)


def is_shard_completed(output_dir: Path, subdir_name: str, shard_name: str) -> bool:
    return (output_dir / subdir_name / f".shard_done_{shard_name}").exists()


def mark_shard_completed(output_dir: Path, subdir_name: str, shard_name: str, shard_stats: dict):
    cp_path = output_dir / subdir_name / f".shard_done_{shard_name}"
    cp_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cp_path, "w") as f:
        json.dump(shard_stats, f, indent=2)


def try_finalize_subdir(output_dir: Path, input_dir: Path, subdir_name: str, logger: logging.Logger):
    if is_subdir_completed(output_dir, subdir_name):
        return

    cut_shards = sorted((input_dir / subdir_name).glob("cuts.*.jsonl.gz"))
    for cs in cut_shards:
        if not is_shard_completed(output_dir, subdir_name, cs.name):
            return

    lang_code = parse_language(subdir_name)
    total_cuts = 0
    total_failed = 0
    total_wer = 0.0
    total_cer = 0.0

    for f in sorted((output_dir / subdir_name).glob(".shard_done_*")):
        with open(f) as fh:
            s = json.load(fh)
            total_cuts += s["cuts"]
            total_failed += s["failed"]
            total_wer += s["sum_wer"]
            total_cer += s["sum_cer"]

    stats = {
        "subdir": subdir_name,
        "language": lang_code,
        "total_cuts": total_cuts,
        "failed": total_failed,
        "avg_wer": round(total_wer / max(total_cuts, 1), 6),
        "avg_cer": round(total_cer / max(total_cuts, 1), 6),
    }
    mark_subdir_completed(output_dir, subdir_name, stats)
    logger.info(
        f"All shards done for {subdir_name}: {total_cuts} cuts, "
        f"avg WER={stats['avg_wer']:.4f}, avg CER={stats['avg_cer']:.4f}"
    )

def build_work_list(input_dir: Path, output_dir: Path, logger: logging.Logger) -> list[dict]:
    work = []
    skipped_subdirs = 0
    skipped_shards = 0
    skipped_langs = 0

    for d in sorted(input_dir.iterdir()):
        if not d.is_dir():
            continue

        lang_code = parse_language(d.name)
        if get_whisper_language(lang_code) is None:
            skipped_langs += 1
            continue

        if is_subdir_completed(output_dir, d.name):
            skipped_subdirs += 1
            continue

        cut_shards = sorted(d.glob("cuts.*.jsonl.gz"))
        rec_shards = sorted(d.glob("recording.*.tar"))

        if len(cut_shards) == 0:
            continue

        total_shards = len(cut_shards)

        for idx, (cs, rs) in enumerate(zip(cut_shards, rec_shards)):
            if is_shard_completed(output_dir, d.name, cs.name):
                skipped_shards += 1
                continue

            work.append({
                "subdir": d.name,
                "shard_idx": idx,
                "total_shards": total_shards,
                "cut_shard": str(cs),
                "rec_shard": str(rs),
            })

    logger.info(
        f"Work list: {len(work)} shards to process, "
        f"{skipped_subdirs} subdirs completed, {skipped_shards} shards completed, "
        f"{skipped_langs} langs unsupported"
    )
    return work



class AudioPrefetcher:
    """
    Reads cuts from SHAR, loads and resamples audio.
    """
    def __init__(self, cuts, num_workers: int = 4, prefetch_size: int = 32):
        self.cuts = cuts
        self.num_workers = num_workers
        self.prefetch_size = prefetch_size
        self.queue = queue.Queue(maxsize=prefetch_size)
        self.done = threading.Event()
        self.error = None

    def _load_audio(self, cut):
        """Load and resample a single cut's audio (CPU work)."""
        try:
            if not cut.supervisions or not cut.supervisions[0].text:
                return (cut, None)  # No audio needed

            cut_16k = cut.resample(16000)
            audio_array = cut_16k.load_audio()
            if audio_array.ndim > 1:
                audio_array = audio_array[0]
            if isinstance(audio_array, torch.Tensor):
                audio_array = audio_array.numpy()
            return (cut, audio_array)
        except Exception as e:
            return (cut, e)  # Pass exception through

    def _producer(self):
        """Background thread that loads audio into the buffer."""
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                futures = set()
                cut_iter = iter(self.cuts)
                exhausted = False

                while not exhausted or futures:
                    while len(futures) < self.prefetch_size and not exhausted:
                        try:
                            cut = next(cut_iter)
                            futures.add(executor.submit(self._load_audio, cut))
                        except StopIteration:
                            exhausted = True
                            break

                    if not futures:
                        break

                    done, _ = concurrent.futures.wait(
                        futures, return_when=concurrent.futures.FIRST_COMPLETED
                    )

                    for fut in done:
                        futures.remove(fut)
                        result = fut.result()
                        self.queue.put(result)

        except Exception as e:
            self.error = e
        finally:
            self.done.set()

    def __iter__(self):
        thread = threading.Thread(target=self._producer, daemon=True)
        thread.start()

        while True:
            try:
                # Timeout allows us to check self.done periodically
                yield self.queue.get(timeout=0.1)
            except queue.Empty:
                if self.done.is_set():
                    # Drain any remaining items
                    while not self.queue.empty():
                        yield self.queue.get()
                    break

        if self.error:
            raise self.error


def process_shard(
        shard_info: dict,
        output_dir: Path,
        model_pipeline: BatchedInferencePipeline,
        normalizer: BasicTextNormalizer,
        batch_size: int,
        num_prefetch_workers: int,
        logger: logging.Logger,
    ):
    subdir_name = shard_info["subdir"]
    cut_shard = Path(shard_info["cut_shard"])
    rec_shard = Path(shard_info["rec_shard"])
    shard_idx = shard_info["shard_idx"]
    total_shards = shard_info["total_shards"]

    lang_code = parse_language(subdir_name)
    whisper_lang = get_whisper_language(lang_code)

    out_subdir = output_dir / subdir_name
    out_subdir.mkdir(parents=True, exist_ok=True)
    
    out_path = out_subdir / cut_shard.name

    base_tmp_dir = Path("/iopsstor/scratch/cscs/arsaikia/commonvoice/temp_processing")
    base_tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = base_tmp_dir / f"shard_{os.environ.get('SLURM_PROCID', 0)}_{cut_shard.name.replace('.gz', '')}.tmp"
    
    logger.info(
        f"Processing {subdir_name} shard {shard_idx+1}/{total_shards}, "
        f"lang={whisper_lang or 'auto'}"
    )

    shard_cuts = 0
    shard_failed = 0
    shard_wer = 0.0
    shard_cer = 0.0
    skip_count = 0

    # --- AUTO-RESUME ---
    if tmp_path.exists():
        logger.info(f"  Found existing temp text file for {cut_shard.name}. Calculating resume point...")
        try:
            with open(tmp_path, "r") as f:
                for line in f:
                    skip_count += 1
                    c = json.loads(line)
                    if c.get("custom", {}).get("status") == 1:
                        shard_cuts += 1
                        shard_wer += c["custom"].get("wer", 0.0)
                        shard_cer += c["custom"].get("cer", 0.0)
                    else:
                        shard_failed += 1
            logger.info(
                f"  Resuming from cut {skip_count} "
                f"(Recovered {shard_cuts} successful, {shard_failed} failed)"
            )
        except Exception as e:
            logger.warning(f"  Failed to read temp text file cleanly, starting over. Error: {e}")
            skip_count = 0
            shard_cuts = 0
            shard_failed = 0
            shard_wer = 0.0
            shard_cer = 0.0

    fields = {
        "cuts": [cut_shard],
        "recording": [rec_shard],
    }
    cuts = CutSet.from_shar(fields=fields, shuffle_shards=False)
    
    cut_iter = iter(cuts)
    for _ in range(skip_count):
        try:
            next(cut_iter)
        except StopIteration:
            break
    
    prefetcher = AudioPrefetcher(
        cut_iter, 
        num_workers=num_prefetch_workers,
        prefetch_size=16,
    )

    file_mode = "a" if skip_count > 0 else "w"
    
    with open(tmp_path, file_mode) as f:
        for cut, audio_or_error in prefetcher:
            try:
                if audio_or_error is None:
                    if cut.custom is None: cut.custom = {}
                    cut.custom.update({"transcription": None, "norm_transcription": None, "norm_text": None, "wer": None, "cer": None, "status": -1})
                    f.write(json.dumps(cut.to_dict(), default=str) + "\n")
                    shard_failed += 1
                    continue

                if isinstance(audio_or_error, Exception):
                    raise audio_or_error

                audio_array = audio_or_error
                raw_ground_truth = cut.supervisions[0].text

                segments, _ = model_pipeline.transcribe(audio_array, language=whisper_lang, batch_size=batch_size)
                
                del audio_array 
                
                segments_list = list(segments)
                raw_hypothesis = " ".join([seg.text for seg in segments_list]).strip()

                norm_ground_truth = normalizer(raw_ground_truth)
                norm_hypothesis = normalizer(raw_hypothesis)

                if not norm_ground_truth.strip():
                    if cut.custom is None: cut.custom = {}
                    cut.custom.update({"transcription": raw_hypothesis, "norm_transcription": norm_hypothesis, "norm_text": "", "wer": None, "cer": None, "status": 0})
                    f.write(json.dumps(cut.to_dict(), default=str) + "\n")
                    shard_failed += 1
                    continue

                wer_val = jiwer.wer(norm_ground_truth, norm_hypothesis) if norm_hypothesis.strip() else 1.0
                cer_val = jiwer.cer(norm_ground_truth, norm_hypothesis) if norm_hypothesis.strip() else 1.0

                if cut.custom is None: cut.custom = {}
                cut.custom.update({"transcription": raw_hypothesis, "norm_transcription": norm_hypothesis, "norm_text": norm_ground_truth, "wer": round(float(wer_val), 6), "cer": round(float(cer_val), 6), "status": 1})

                f.write(json.dumps(cut.to_dict(), default=str) + "\n")
                shard_cuts += 1
                shard_wer += wer_val
                shard_cer += cer_val

                total_processed = shard_cuts + shard_failed
   
                if total_processed % 500 == 0:
                    logger.info(f"  {subdir_name} shard {shard_idx+1}: {total_processed + skip_count} cuts processed, avg WER={shard_wer/max(shard_cuts, 1):.4f}")
            except Exception as e:
                logger.error(f"  Error processing cut {getattr(cut, 'id', '?')}: {e}")
                if cut.custom is None: cut.custom = {}
                cut.custom["transcription"] = None
                cut.custom["status"] = f"Error: {str(e)[:200]}"
                f.write(json.dumps(cut.to_dict(), default=str) + "\n")
                shard_failed += 1

    local_gz_path = tmp_path.with_suffix(".tmp.gz")
    
    logger.info(f"  Compressing {tmp_path.name} locally using system compression...")
    try:
        # Use multi-core pigz if available, otherwise fallback to fast gzip
        if shutil.which("pigz"):
            compress_cmd = ["pigz", "-c", "-1", "-p", str(num_prefetch_workers), str(tmp_path)]
            logger.info("  Using multi-core 'pigz' for lightning-fast compression.")
        else:
            compress_cmd = ["gzip", "-c", "-1", str(tmp_path)]
            logger.info("  'pigz' not found. Falling back to single-core 'gzip'.")

        with open(local_gz_path, "wb") as f_out:
            subprocess.run(compress_cmd, stdout=f_out, check=True)
        
        logger.info(f"  Transferring compressed file to Capstor: {out_path.name}")
        for attempt in range(5):
            try:
                shutil.copy2(local_gz_path, out_path)
                break
            except OSError as e:
                logger.warning(f"  Copy attempt {attempt+1}/5 failed: {e}")
                time.sleep(5 * (attempt + 1))
        else:
            logger.error("  Failed to copy shard to capstor after 5 attempts")
            raise OSError("Failed to write shard to capstor")
            
    finally:
        tmp_path.unlink(missing_ok=True)
        local_gz_path.unlink(missing_ok=True)

    shard_stats = {
        "cuts": shard_cuts,
        "failed": shard_failed,
        "sum_wer": round(shard_wer, 6),
        "sum_cer": round(shard_cer, 6),
    }
    mark_shard_completed(output_dir, subdir_name, cut_shard.name, shard_stats)

    avg_wer = shard_wer / max(shard_cuts, 1)
    avg_cer = shard_cer / max(shard_cuts, 1)
    logger.info(
        f"  Done {subdir_name} shard {shard_idx+1}/{total_shards}: "
        f"{shard_cuts} cuts, {shard_failed} failed, "
        f"avg WER={avg_wer:.4f}, avg CER={avg_cer:.4f}"
    )

def main():
    parser = argparse.ArgumentParser(description="CommonVoice quality pipeline")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/capstor/store/cscs/swissai/infra01/audio-datasets/SHAR/stage_2/commonvoice",
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--whisper_model", type=str, default="turbo")
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--compute_type", type=str, default="float16")
    parser.add_argument("--prefetch_workers", type=int, default=4,
                        help="Number of CPU threads for audio prefetching")
    args = parser.parse_args()

    rank = int(os.environ.get("SLURM_PROCID", 0))
    world_size = int(os.environ.get("SLURM_NTASKS", 1))

    logger = logging.getLogger(f"rank_{rank}")
    logger.info(f"Starting: rank={rank}, world_size={world_size}")

    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        device = "cuda"
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        logger.warning("No GPU available, using CPU")

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    work = build_work_list(input_dir, output_dir, logger)

    if not work:
        logger.info("Nothing to do, exiting.")
        return

    my_shards = [s for i, s in enumerate(work) if i % world_size == rank]

    subdirs_in_assignment = set(s["subdir"] for s in my_shards)
    logger.info(
        f"Assigned {len(my_shards)} shards across {len(subdirs_in_assignment)} languages"
    )

    if not my_shards:
        logger.info("No shards assigned to this rank, exiting.")
        return

    logger.info(f"Loading Whisper model: {args.whisper_model} ({args.compute_type})")
    t0 = time.time()
    base_model = WhisperModel(
        args.whisper_model,
        device=device,
        compute_type=args.compute_type,
    )
    model_pipeline = BatchedInferencePipeline(model=base_model)
    logger.info(f"Model loaded in {time.time() - t0:.1f}s")

    normalizer = BasicTextNormalizer()

    for i, shard_info in enumerate(my_shards):
        logger.info(f"{'='*60}")
        logger.info(f"Shard {i+1}/{len(my_shards)}")
        t1 = time.time()

        try:
            process_shard(
                shard_info=shard_info,
                output_dir=output_dir,
                model_pipeline=model_pipeline,
                normalizer=normalizer,
                batch_size=args.batch_size,
                num_prefetch_workers=args.prefetch_workers,
                logger=logger,
            )

            elapsed = time.time() - t1
            logger.info(f"Shard completed in {elapsed:.1f}s")

            try_finalize_subdir(output_dir, input_dir, shard_info["subdir"], logger)

        except Exception as e:
            logger.error(
                f"FAILED on {shard_info['subdir']}/{Path(shard_info['cut_shard']).name}: {e}",
                exc_info=True,
            )
            continue

    logger.info("All assigned shards processed. Done.")


if __name__ == "__main__":
    main()