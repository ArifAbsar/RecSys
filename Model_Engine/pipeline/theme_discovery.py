"""
pipeline/theme_discovery.py
============================
AutoThemeDiscovery — data-driven replacement for SemanticRuleEngine + YAML rules.
Works on any domain with zero manual configuration.

Extracted verbatim from run_stage2.py (lines 142-285).
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

from .config import PipelineConfig


def _auto_select_k(embeddings, k_min: int, k_max: int, sample_size: int) -> int:
    """Pick the number of clusters that maximises sampled silhouette score."""
    n = len(embeddings)
    k_max = max(k_min, min(k_max, n // 5))
    if k_min >= k_max:
        return k_min

    idx = np.random.choice(n, min(sample_size, n), replace=False)
    sample = embeddings[idx]

    best_k, best_score = k_min, -1.0
    step = max(1, (k_max - k_min) // 8)          # at most ~8 candidates
    for k in range(k_min, k_max + 1, step):
        km = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=3)
        labels = km.fit_predict(sample)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(sample, labels, sample_size=min(500, len(sample)))
        if score > best_score:
            best_score, best_k = score, k
    return best_k


class AutoThemeDiscovery:
    """
    Data-driven replacement for SemanticRuleEngine + YAML rules.

    Pipeline:
      1. Encode all item descriptions with SentenceTransformer.
      2. Cluster embeddings with MiniBatchKMeans (k auto-selected via silhouette).
      3. Label each cluster from its most centroid-proximal item.
      4. Strategic score = cosine similarity of each item to its own cluster centroid
         (normalised globally to [0, 1]).

    No hand-written rules, no domain knowledge required.
    """

    def __init__(
        self,
        model_name: str = 'all-MiniLM-L6-v2',
        n_clusters='auto',
        k_min: int = 5,
        k_max: int = 50,
        secondary_cluster_threshold: float = 0.85,
        cfg: PipelineConfig = None,
    ):
        self.cfg = cfg or PipelineConfig()
        print(f"[THEME] Initialising AutoThemeDiscovery ({model_name})...")
        self.encoder = SentenceTransformer(model_name)
        self.n_clusters_cfg = n_clusters
        self.k_min = k_min
        self.k_max = k_max
        self.secondary_cluster_threshold = secondary_cluster_threshold
        self.item_cluster_ids: np.ndarray | None = None
        self.cluster_centroids: np.ndarray | None = None
        self.theme_labels: dict[int, str] = {}
        self.strategic_scores: np.ndarray | None = None
        self._embeddings: np.ndarray | None = None

    def fit(self, descriptions: list[str]) -> 'AutoThemeDiscovery':
        print("[THEME] Encoding item descriptions...")
        emb = self.encoder.encode(
            descriptions, normalize_embeddings=True, show_progress_bar=True
        )
        self._embeddings = emb   # cached for soft theme assignment
        n = len(descriptions)

        if self.n_clusters_cfg == 'auto':
            if n < 20:
                k = max(2, n // 3)
            elif n < 100:
                k = max(3, int(np.sqrt(n)))
            else:
                k = _auto_select_k(emb, self.k_min, min(self.k_max, n // 5), sample_size=self.cfg.silhouette_sample_size)
        else:
            k = int(self.n_clusters_cfg)
        k = max(2, min(k, n - 1))

        print(f"[THEME] Clustering {n} items into {k} themes...")
        km = MiniBatchKMeans(
            n_clusters=k,
            random_state=self.cfg.clustering_random_state,
            n_init=self.cfg.clustering_n_init
        )
        self.item_cluster_ids = km.fit_predict(emb)
        self.cluster_centroids = km.cluster_centers_   # (k, dim)

        self.theme_labels = {}
        for c in range(k):
            members = np.where(self.item_cluster_ids == c)[0]
            if len(members) == 0:
                self.theme_labels[c] = f"Theme_{c}"
                continue
            sims = cosine_similarity(
                emb[members], self.cluster_centroids[c].reshape(1, -1)
            ).flatten()
            rep_item = members[np.argmax(sims)]
            words = descriptions[rep_item].split()[:4]
            label = "_".join(w.strip('.,;:') for w in words if w.strip('.,;:'))
            self.theme_labels[c] = label or f"Theme_{c}"

        all_sims  = cosine_similarity(emb, self.cluster_centroids)
        own_c     = self.item_cluster_ids
        own_sim   = all_sims[np.arange(n), own_c]

        mask_val = all_sims.copy()
        mask_val[np.arange(n), own_c] = -1.0
        best_other_sim = mask_val.max(axis=1)

        raw = own_sim - best_other_sim
        raw = np.clip(raw, 0, None)
        s_max = raw.max()
        self.strategic_scores = raw / s_max if s_max > 0 else raw

        sample_labels = list(self.theme_labels.values())[:5]
        print(f"[THEME] Done. {k} themes discovered. Sample: {sample_labels}")
        return self

    def get_item_themes(self, item_idx: int) -> list[str]:
        primary_c = int(self.item_cluster_ids[item_idx])
        themes = [self.theme_labels[primary_c]]
        if self._embeddings is not None:
            emb = self._embeddings[item_idx].reshape(1, -1)
            sims = cosine_similarity(emb, self.cluster_centroids).flatten()
            primary_sim = float(sims[primary_c])
            sims[primary_c] = -1.0
            secondary_c = int(np.argmax(sims))
            if primary_sim > 0 and float(sims[secondary_c]) >= self.secondary_cluster_threshold * primary_sim:
                themes.append(self.theme_labels[secondary_c])
        return themes

    def build_theme_map(self, n_items: int) -> list[list[str]]:
        return [self.get_item_themes(i) for i in range(n_items)]

    def build_strategic_scores(self, n_items: int) -> np.ndarray:
        return self.strategic_scores[:n_items].copy()
