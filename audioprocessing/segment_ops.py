from typing import List, Dict, Any, Optional, Tuple
from copy import deepcopy


def pre_merge(
    segments: list,
    max_gap_samples: int,
) -> list:
    """
    merge VAD segments that are within
    max_gap_samples of each other.
    """
    if not segments:
        return []

    merged = []
    cur = deepcopy(segments[0])

    for nxt in segments[1:]:
        gap = nxt["start"] - cur["end"]
        if gap <= max_gap_samples:
            cur["end"] = nxt["end"]
        else:
            merged.append(cur)
            cur = deepcopy(nxt)

    merged.append(cur)
    return merged


def filter_short(segments: list, min_samples: int) -> list:
    """Drop segments shorter than min_samples."""
    return [s for s in segments if (s["end"] - s["start"]) >= min_samples]


def initialise_metadata(
    segments: list,
    language: Optional[str] = None,
) -> None:
    """Set default speaker / language / score on every segment (in-place)."""
    for seg in segments:
        seg["speaker"] = -1
        seg["language"] = language if language else "unknown"
        seg["score"] = 1.0 if language else 0.0


def merge_segments(
                    segments: list,
                    max_gap_samples: int,
                    max_duration_samples: int,
                    keys: Tuple[str, ...] = ("speaker", "language"),
                ) -> list:
    """
    Merge consecutive segments that share the same values for all keys,
    are separated by at most max_gap_samples, and whose combined
    duration stays <= max_duration_samples.
    """
    if not segments:
        return []

    merged = []
    cur = deepcopy(segments[0])

    for nxt in segments[1:]:
        gap = nxt["start"] - cur["end"]
        combined_dur = nxt["end"] - cur["start"]
        same = all(cur.get(k) == nxt.get(k) for k in keys)

        if gap <= max_gap_samples and same and combined_dur <= max_duration_samples:
            cur["end"] = nxt["end"]
        else:
            merged.append(cur)
            cur = deepcopy(nxt)

    merged.append(cur)
    return merged

def format_output(
                    segments: list,
                    sr: int,
                ) -> List[Dict[str, Any]]:
    """Convert sample-based segments to seconds-based output dicts."""
    output = []
    for seg in segments:
        output.append(
            {
                "start": seg["start"] / sr,
                "end": seg["end"] / sr,
                "duration": (seg["end"] - seg["start"]) / sr,
                "speaker": seg['speaker'],
                "language": seg.get("language", "unknown"),
                "score": seg.get("score", 0.0),
                "text": seg.get("text", ""),
            }
        )
    return output