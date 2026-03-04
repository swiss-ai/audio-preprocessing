import numpy as np
import torch
from typing import Optional, List
from collections import Counter

from sklearn.metrics.pairwise import cosine_similarity
from nemo.collections.asr.parts.utils.offline_clustering import SpeakerClustering
import nemo.collections.asr as nemo_asr

from .config import AudioConfig


class SpeakerDiarizer:
    """Assigns speaker labels to VAD segments."""

    # Minimum number of embeddings needed for reliable NME-SC clustering.
    # Below this, long VAD segments are windowed to produce more embeddings.
    MIN_EMBEDDINGS_FOR_CLUSTERING = 12

    def __init__(self, config: AudioConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")

        self.encoder = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
            model_name=config.speaker_model_name
        ).to(self.device)
        self.encoder.eval()

        use_cuda = torch.cuda.is_available() and config.device == "cuda"
        self.clusterer = SpeakerClustering(
            min_samples_for_nmesc=6,
            nme_mat_size=512,
            sparse_search=True,
            cuda=use_cuda,
        )

    # ------------------------------------------------------------------
    # Batched embedding extraction
    # ------------------------------------------------------------------

    def extract_embeddings(self, 
                           wav: torch.Tensor, 
                           segments: list, 
                           batch_size: Optional[int] = None
                          ) -> np.ndarray:
        """
        Extract TitaNet embeddings in batches.

        Segments are padded to the longest in each batch, with lengths
        passed explicitly so the model ignores padding.
        """
        batch_size = batch_size or self.config.embedding_batch_size
        all_embeddings: List[np.ndarray] = []

        slices = []
        for seg in segments:
            s, e = int(seg["start"]), int(seg["end"])
            slices.append(wav[s:e])

        for batch_start in range(0, len(slices), batch_size):
            batch = slices[batch_start : batch_start + batch_size]
            lengths = [s.shape[0] for s in batch]
            max_len = max(lengths)

            padded = torch.zeros(len(batch), max_len)
            for i, s in enumerate(batch):
                padded[i, : s.shape[0]] = s

            padded = padded.to(self.device)
            length_tensor = torch.tensor(lengths, dtype=torch.long, device=self.device)

            with torch.no_grad():
                embs, _ = self.encoder.forward(
                    input_signal=padded, input_signal_length=length_tensor
                )
            all_embeddings.append(embs.cpu().numpy())

        return np.concatenate(all_embeddings, axis=0)

    def _create_windows(self, segments: list, sr: int) -> list:
        """
        Split VAD segments into overlapping windows for embedding.

        diarization_window as window size with diarization_overlap as shift.

        Returns list of dicts with 'start', 'end' (in samples), 'parent_idx'.
        """
        window_samples = int(self.config.diarization_window * sr)
        hop_samples = int(
            (self.config.diarization_window - self.config.diarization_overlap) * sr
        )

        windows = []
        for idx, seg in enumerate(segments):
            seg_start = seg["start"]
            seg_end = seg["end"]
            seg_len = seg_end - seg_start

            if seg_len <= window_samples:
                windows.append({
                    "start": seg_start,
                    "end": seg_end,
                    "parent_idx": idx,
                })
            else:
                pos = seg_start
                while pos + window_samples <= seg_end:
                    windows.append({
                        "start": pos,
                        "end": pos + window_samples,
                        "parent_idx": idx,
                    })
                    pos += hop_samples
                if pos < seg_end and (seg_end - pos) > hop_samples // 2:
                    windows.append({
                        "start": seg_end - window_samples,
                        "end": seg_end,
                        "parent_idx": idx,
                    })

        return windows

    def _needs_windowing(self, segments: list) -> bool:
        """Check if we have too few segments for reliable clustering."""
        return len(segments) < self.MIN_EMBEDDINGS_FOR_CLUSTERING

    def _run_nmesc(self,
                   embeddings: np.ndarray,
                   timestamps_sec: list,
                   num_speakers: Optional[int] = None,
                  ) -> np.ndarray:
        n_segments = len(embeddings)

        embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)
        timestamps_np = np.array(timestamps_sec, dtype=np.float32)
        timestamps_tensor = torch.tensor(timestamps_np, dtype=torch.float32)

        multiscale_segment_counts = torch.tensor([n_segments], dtype=torch.long)

        multiscale_weights = torch.tensor([[1.0]])

        oracle = num_speakers if num_speakers is not None else -1

        labels = self.clusterer.forward_infer(
            embeddings_in_scales=embeddings_tensor,
            timestamps_in_scales=timestamps_tensor,
            multiscale_segment_counts=multiscale_segment_counts,
            multiscale_weights=multiscale_weights,
            oracle_num_speakers=oracle,
            max_num_speakers=self.config.max_speakers,
            max_rp_threshold=0.25,
            sparse_search_volume=30,
        )

        return labels.cpu().numpy()

    def assign_short_segments(self,
                              long_embeddings: np.ndarray,
                              long_labels: np.ndarray,
                              short_embeddings: np.ndarray,
                             ) -> np.ndarray:
        """Assign short segments to the nearest speaker centroid."""
        unique_speakers = np.unique(long_labels)
        centroids = np.array(
            [long_embeddings[long_labels == spk].mean(axis=0) for spk in unique_speakers]
        )
        sim = cosine_similarity(short_embeddings, centroids)
        best_idx = np.argmax(sim, axis=1)
        return unique_speakers[best_idx]

    def smooth_speakers(self, segments: list, sr: int) -> None:
        """
        Flip isolated short segments to match their neighbours.
        """
        max_samples = self.config.smoothing_max_dur * sr
        for i in range(1, len(segments) - 1):
            seg = segments[i]
            dur = seg["end"] - seg["start"]
            if dur > max_samples:
                continue
            prev_spk = segments[i - 1]["speaker"]
            next_spk = segments[i + 1]["speaker"]
            if prev_spk == next_spk and seg["speaker"] != prev_spk:
                seg["speaker"] = prev_spk

    def run(self,
            wav: torch.Tensor,
            segments: list,
            sr: int,
            num_speakers: Optional[int] = None,
            ) -> None:
        """
        embed → cluster → assign → smooth.

        If VAD produces too few segments (< MIN_EMBEDDINGS_FOR_CLUSTERING),
        long segments are split into overlapping windows before embedding.
        Window-level labels are then mapped back to original segments via
        majority vote.
        """
        if len(segments) == 0:
            return

        use_windowing = self._needs_windowing(segments)

        if use_windowing:
            windows = self._create_windows(segments, sr)
            if not windows:
                return

            embeddings = self.extract_embeddings(wav, windows)
            if len(embeddings) == 0:
                return

            timestamps = [
                (w["start"] / sr, w["end"] / sr) for w in windows
            ]

            labels = self._run_nmesc(embeddings, timestamps, num_speakers)

            for i, label in enumerate(labels):
                windows[i]["speaker"] = int(label)

            # Map back to original segments via majority vote
            for idx, seg in enumerate(segments):
                window_labels = [
                    w["speaker"]
                    for w in windows
                    if w["parent_idx"] == idx and "speaker" in w
                ]
                if window_labels:
                    seg["speaker"] = Counter(window_labels).most_common(1)[0][0]
                else:
                    seg["speaker"] = 0

        else:
            embeddings = self.extract_embeddings(wav, segments)
            if len(embeddings) == 0:
                return

            # Split long / short
            min_samples = self.config.valid_vad_segment_min_dur * sr
            long_idx, short_idx = [], []
            for i, seg in enumerate(segments):
                (
                    long_idx
                    if (seg["end"] - seg["start"]) >= min_samples
                    else short_idx
                ).append(i)

            if not long_idx:
                long_idx, short_idx = short_idx, []

            long_embs = embeddings[long_idx]

            timestamps = [
                (segments[i]["start"] / sr, segments[i]["end"] / sr)
                for i in long_idx
            ]

            labels = self._run_nmesc(long_embs, timestamps, num_speakers)

            for i, label in zip(long_idx, labels):
                segments[i]["speaker"] = int(label)

            # Assign short segments to nearest centroid
            if short_idx:
                short_embs = embeddings[short_idx]
                short_labels = self.assign_short_segments(
                    long_embs, labels, short_embs
                )
                for i, label in zip(short_idx, short_labels):
                    segments[i]["speaker"] = int(label)

        # Temporal smoothening
        self.smooth_speakers(segments, sr)