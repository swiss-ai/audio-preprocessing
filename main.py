import argparse
import json
import time

from audioprocessing.config import AudioConfig
from audioprocessing.pipeline import AudioProcessor


def main():
    parser = argparse.ArgumentParser(description="Audio Processing Pipeline")
    parser.add_argument("input", help="Path to audio file")
    parser.add_argument("--output", "-o", default="output.json", help="Output JSON path")
    parser.add_argument("--num-speakers", type=int, default=None)
    parser.add_argument("--language", type=str, default=None, help="Known language code (skip LID)")
    parser.add_argument("--no-diarization", action="store_true")
    parser.add_argument("--no-transcription", action="store_true")
    parser.add_argument("--no-lid", action="store_true")
    parser.add_argument("--no-annotation", action="store_true")
    args = parser.parse_args()

    config = AudioConfig(
        num_speakers=args.num_speakers,
        language=args.language,
        enable_diarization=not args.no_diarization,
        enable_transcription=not args.no_transcription,
        enable_language_id=not args.no_lid,
        enable_annotation=not args.no_annotation,
    )

    t0 = time.time()
    processor = AudioProcessor(config)
    print(f"Initialise: {time.time() - t0:.2f}s")

    t0 = time.time()
    result = processor.process(args.input)
    print(f"Process:    {time.time() - t0:.2f}s")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

    print(f"Output written to {args.output}")


if __name__ == "__main__":
    main()