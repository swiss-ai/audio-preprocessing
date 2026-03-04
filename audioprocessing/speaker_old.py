"""
The speaker-count estimation and clustering use the NME-SC algorithm from:
"Auto-Tuning Spectral Clustering for Speaker Diarization Using Normalized 
Maximum Eigengap", IEEE SPL 2019. https://arxiv.org/abs/2003.02405

From the NVIDIA NeMo implementation:
  https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/asr/parts/utils/nmesc_clustering.py
"""

import numpy as np
import torch
from typing import Optional, List

from sklearn.cluster._kmeans import k_means
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

from .config import AudioConfig


def _get_cosine_affinity(embeddings: np.ndarray) -> np.ndarray:
    """Cosine similarity scaled to [0, 1] via MinMaxScaler (per NeMo)."""
    sim = cosine_similarity(embeddings)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(sim)
    return scaler.transform(sim)


class NMESC:
    """
    Normalized Maximum Eigengap Spectral Clustering.

    Sweeps p-values (number of neighbors to keep in binarized affinity graph),
    computes g_p = (p/N) / (max_eigengap / max_eigenvalue) for each p,
    and picks the p with smallest g_p. This auto-tunes the threshold
    without needing a dev set.

    Parameters
    ----------
    affinity : np.ndarray
        N x N cosine similarity matrix (scaled to [0,1]).
    max_num_speaker : int
        Upper bound on number of speakers.
    max_rp_threshold : float
        Maximum fraction of N to search for p. Default 0.25.
    sparse_search : bool
        If True, search a sparse grid of p values (faster).
    sparse_search_volume : int
        Number of p values to try when sparse_search=True.
    nme_mat_size : int
        Subsample affinity to this size for speed.
    use_cuda : bool
        Use GPU for eigen decomposition.
    """

    def __init__(self,
                 affinity: np.ndarray,
                 max_num_speaker: int = 10,
                 max_rp_threshold: float = 0.25,
                 sparse_search: bool = True,
                 sparse_search_volume: int = 30,
                 nme_mat_size: int = 512,
                 use_cuda: bool = False,
                ):
        self.orig_affinity = affinity
        self.mat = affinity.copy()
        self.max_num_speaker = max_num_speaker
        self.max_rp_threshold = max_rp_threshold
        self.sparse_search = sparse_search
        self.sparse_search_volume = sparse_search_volume
        self.nme_mat_size = nme_mat_size
        self.use_cuda = use_cuda
        self.eps = 1e-10
        self.max_N = None

        # Set after analyze()
        self._best_p_hat = None

    def analyze(self):
        """
        Run NME analysis to find optimal p-value and speaker count.

        Returns
        -------
        est_num_speakers : int
            Estimated number of speakers.
        p_hat : int
            Estimated p-value (scaled back to original matrix size).
        """
        subsample_ratio = self._subsample()
        p_list = self._get_p_list()

        if not p_list:
            self._best_p_hat = 1
            return 1, 1

        # Scan all p values
        eig_ratio_list = []
        est_spk_dict = {}
        for p in p_list:
            est_n, g_p = self._get_eig_ratio(p)
            est_spk_dict[p] = est_n
            eig_ratio_list.append(g_p)

        # Pick p with minimum g_p
        best_idx = np.argmin(eig_ratio_list)
        best_p = p_list[best_idx]

        # Ensure full connectivity
        affinity_graph = self._get_affinity_graph(self.mat, best_p)
        if not self._is_fully_connected(affinity_graph):
            _, best_p = self._get_minimum_connection(self.mat, self.max_N, p_list)

        p_hat = int(subsample_ratio * best_p)
        self._best_p_hat = p_hat
        est_num_speakers = est_spk_dict.get(best_p, 1)

        return est_num_speakers, p_hat

    def cluster(self, n_clusters: int) -> np.ndarray:
        """
        Spectral clustering on the original affinity matrix.

        Uses the p_hat found by analyze().

        Steps:
          1. Binarize original affinity at p_hat
          2. Ensure full connectivity
          3. Compute Laplacian eigenvectors
          4. k-means on top-k eigenvectors

        Returns:
        labels : np.ndarray of shape (N,)
        """
        n = self.orig_affinity.shape[0]
        if n_clusters >= n:
            return np.arange(n)

        if self._best_p_hat is None:
            self.analyze()

        p = max(1, min(self._best_p_hat, n - 1))
        affinity_graph = self._get_affinity_graph(self.orig_affinity, p)

        if not self._is_fully_connected(affinity_graph):
            max_N = max(1, int(n * self.max_rp_threshold))
            p_list = list(np.linspace(1, max_N, 30, endpoint=True).astype(int))
            affinity_graph, _ = self._get_minimum_connection(
                self.orig_affinity, max_N, p_list
            )

        spectral_emb = self._get_spectral_embeddings(affinity_graph, n_clusters)
        _, labels, _ = k_means(spectral_emb, n_clusters, random_state=0, n_init=10)
        return labels

    @staticmethod
    def _get_kneighbors_connections(affinity: np.ndarray, p: int) -> np.ndarray:
        """Binarize: keep top-p values per row, set rest to 0."""
        binarized = np.zeros_like(affinity)
        for i, row in enumerate(affinity):
            top_idx = np.argsort(row)[::-1][:p]
            binarized[top_idx, i] = 1
        return binarized

    @staticmethod
    def _get_affinity_graph(affinity: np.ndarray, p: int) -> np.ndarray:
        """Binarize and symmetrize."""
        X = NMESC._get_kneighbors_connections(affinity, p)
        return 0.5 * (X + X.T)

    @staticmethod
    def _is_fully_connected(affinity: np.ndarray) -> bool:
        """BFS check for graph connectivity."""
        n = affinity.shape[0]
        visited = np.zeros(n, dtype=bool)
        queue = np.zeros(n, dtype=bool)
        queue[0] = True
        for _ in range(n):
            prev = visited.sum()
            np.logical_or(visited, queue, out=visited)
            if visited.sum() == prev:
                break
            indices = np.where(queue)[0]
            queue.fill(False)
            for i in indices:
                np.logical_or(queue, affinity[i].astype(bool), out=queue)
        return visited.sum() == n

    @staticmethod
    def _get_minimum_connection(affinity: np.ndarray, max_n: int, p_list: list):
        """Increase p until graph is fully connected."""
        graph = NMESC._get_affinity_graph(affinity, 1)
        for p in p_list:
            if NMESC._is_fully_connected(graph) or p > max_n:
                break
            graph = NMESC._get_affinity_graph(affinity, p)
        return graph, p

    @staticmethod
    def _get_laplacian(X: np.ndarray) -> np.ndarray:
        """Unnormalized Laplacian: L = D - A."""
        A = X.copy()
        np.fill_diagonal(A, 0)
        D = np.diag(np.sum(np.abs(A), axis=1))
        return D - A

    @staticmethod
    def _eig_decompose(laplacian: np.ndarray, use_cuda: bool = False):
        """Eigen decomposition, optionally on GPU."""
        try:
            if use_cuda and torch.cuda.is_available():
                L_t = torch.from_numpy(laplacian).float().cuda()
                lambdas, diffusion_map = torch.linalg.eigh(L_t)
                return lambdas.cpu().numpy(), diffusion_map.cpu().numpy()
        except Exception:
            pass
        from scipy.linalg import eigh
        return eigh(laplacian)

    def _estimate_num_speakers_from_affinity(self, affinity: np.ndarray):
        """Eigengap-based speaker count from a binarized affinity matrix."""
        laplacian = self._get_laplacian(affinity)
        lambdas, diffusion_map = self._eig_decompose(laplacian, self.use_cuda)
        lambdas = np.sort(np.real(lambdas))
        gaps = list(lambdas[1:] - lambdas[:-1])
        n_spk = np.argmax(gaps[:min(self.max_num_speaker, len(gaps))]) + 1
        return n_spk, lambdas, gaps

    def _get_spectral_embeddings(self, affinity: np.ndarray, n_spks: int) -> np.ndarray:
        """Top-k eigenvectors from the Laplacian for k-means."""
        laplacian = self._get_laplacian(affinity)
        _, diffusion_map = self._eig_decompose(laplacian, self.use_cuda)
        return diffusion_map[:, :n_spks]

    def _subsample(self) -> int:
        """Subsample affinity matrix."""
        ratio = int(max(1, self.mat.shape[0] / self.nme_mat_size))
        self.mat = self.mat[::ratio, ::ratio]
        return ratio

    def _get_p_list(self) -> list:
        """Generate list of p-values to search."""
        self.max_N = max(1, int(self.mat.shape[0] * self.max_rp_threshold))
        if self.sparse_search:
            N = min(self.max_N, self.sparse_search_volume)
            return list(np.linspace(1, self.max_N, N, endpoint=True).astype(int))
        else:
            return list(range(1, self.max_N + 1))

    def _get_eig_ratio(self, p: int):
        """
        Compute g_p = (p/N) / (max_eigengap / max_eigenvalue + eps).

        Returns (est_num_speakers, g_p).
        """
        affinity_graph = self._get_affinity_graph(self.mat, p)
        est_n, lambdas, gap_list = self._estimate_num_speakers_from_affinity(
            affinity_graph
        )
        max_gap_idx = np.argmax(gap_list[:self.max_num_speaker])
        max_eig_gap = gap_list[max_gap_idx] / (max(lambdas) + self.eps)
        g_p = (p / self.mat.shape[0]) / (max_eig_gap + self.eps)
        return est_n, g_p


class SpeakerDiarizer:
    """Identifies speaker labels for each VAD segment."""

    def __init__(self, config: AudioConfig):
        import nemo.collections.asr as nemo_asr

        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")

        self.encoder = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
            model_name=config.speaker_model_name
        ).to(self.device)
        self.encoder.eval()

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

    def estimate_num_speakers(self, 
                              embeddings: np.ndarray, 
                              max_speakers: Optional[int] = None
                             ) -> int:
        """
        Estimate speaker count using NME-SC.

        Sweeps binarization thresholds (p-values), computes the normalized
        maximum eigengap ratio g_p for each, and picks the threshold with
        the smallest g_p.
        """
        max_speakers = max_speakers or self.config.max_speakers
        n = len(embeddings)

        if n <= 2:
            return 1

        max_k = min(max_speakers, n - 1)
        if max_k < 2:
            return 1

        affinity = _get_cosine_affinity(embeddings)
        use_cuda = torch.cuda.is_available() and self.config.device == "cuda"

        nmesc = NMESC(
            affinity=affinity,
            max_num_speaker=max_k,
            max_rp_threshold=0.25,
            sparse_search=True,
            sparse_search_volume=30,
            nme_mat_size=300,
            use_cuda=use_cuda,
        )

        est_num_speakers, _ = nmesc.analyze()
        return max(1, min(est_num_speakers, max_k))


    def cluster(self, embeddings: np.ndarray, n_clusters: int) -> np.ndarray:
        """
        NME-SC spectral clustering:
          1. Build cosine affinity
          2. NME analysis → optimal p-value
          3. Binarize → Laplacian eigenvectors → k-means
        """
        n = len(embeddings)
        if n_clusters >= n:
            return np.arange(n)

        affinity = _get_cosine_affinity(embeddings)
        use_cuda = torch.cuda.is_available() and self.config.device == "cuda"

        nmesc = NMESC(
            affinity=affinity,
            max_num_speaker=n_clusters,
            max_rp_threshold=0.25,
            sparse_search=True,
            sparse_search_volume=30,
            nme_mat_size=300,
            use_cuda=use_cuda,
        )
        nmesc.analyze()
        return nmesc.cluster(n_clusters)


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
        Operates in-place.
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
        embedding → estimate number of speakers → clustering → assignment → smoothening.
        """
        embeddings = self.extract_embeddings(wav, segments)
        if len(embeddings) == 0:
            return

        min_samples = self.config.valid_vad_segment_min_dur * sr
        long_idx, short_idx = [], []
        for i, seg in enumerate(segments):
            (long_idx if (seg["end"] - seg["start"]) >= min_samples else short_idx).append(i)

        if not long_idx:
            long_idx, short_idx = short_idx, []

        long_embs = embeddings[long_idx]

        n_clusters = num_speakers or self.estimate_num_speakers(long_embs)
        n_clusters = min(n_clusters, len(long_idx))

        labels = self.cluster(long_embs, n_clusters)
        for i, label in zip(long_idx, labels):
            segments[i]["speaker"] = int(label)

        if short_idx:
            short_embs = embeddings[short_idx]
            short_labels = self.assign_short_segments(long_embs, labels, short_embs)
            for i, label in zip(short_idx, short_labels):
                segments[i]["speaker"] = int(label)

        self.smooth_speakers(segments, sr)