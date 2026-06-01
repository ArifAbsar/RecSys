"""
pipeline/user_profiler.py
==========================
UserInterestProfiler — builds a per-user interest profile from interaction history.

Optimization: interaction index built once in __init__ (O(I) upfront)
instead of scanning inter_feat per user (was O(U × I) total).
"""

import numpy as np

from .config import PipelineConfig


class UserInterestProfiler:
    def __init__(
        self,
        dataset,
        item_id_to_desc: dict,
        item_themes_map: list,
        cfg: PipelineConfig = None,
    ):
        self.cfg = cfg or PipelineConfig()
        self.dataset = dataset
        self.item_id_to_desc = item_id_to_desc
        self.item_themes_map = item_themes_map
        self.user_profiles: dict = {}

        uid_tensor = dataset.inter_feat[dataset.uid_field].numpy()
        iid_tensor = dataset.inter_feat[dataset.iid_field].numpy()
        self._user_items: dict[int, list[int]] = {}
        for uid, iid in zip(uid_tensor, iid_tensor):
            self._user_items.setdefault(int(uid), []).append(int(iid))

    def get_profile(self, user_idx: int) -> dict:
        if user_idx in self.user_profiles:
            return self.user_profiles[user_idx]

        iid_f = self.dataset.iid_field
        item_indices = np.array(self._user_items.get(user_idx, []), dtype=np.int64)

        theme_counts: dict[str, float] = {}
        past_names: list[str] = []
        # Exclude PAD (idx=0) from the denominator so decay weights are accurate
        n_items = int(np.sum(item_indices != 0))

        for rank, idx in enumerate(item_indices):
            if idx == 0:
                continue
            past_names.append(
                self.item_id_to_desc.get(
                    self.dataset.id2token(iid_f, idx), "Unknown"
                )
            )
            weight = float(np.exp(-self.cfg.recency_decay * (n_items - rank - 1)))
            for theme in self.item_themes_map[idx]:
                theme_counts[theme] = theme_counts.get(theme, 0.0) + weight

        ranked = sorted(theme_counts.items(), key=lambda x: x[1], reverse=True)
        n_inter = max(len([x for x in item_indices if x != 0]), 1)
        max_cnt = max(theme_counts.values()) if theme_counts else 1.0
        profile = {
            "top_themes": [t for t, _ in ranked[:self.cfg.top_themes]],
            "theme_weights": {t: cnt / max_cnt for t, cnt in ranked},
            "past_items": past_names[-self.cfg.past_items_window:],
            "past_item_indices": set(int(x) for x in item_indices if x != 0),
            "n_interactions": n_inter,
        }
        self.user_profiles[user_idx] = profile
        return profile

    def get_personalization_match(self, user_idx: int, item_themes: list[str]) -> float:
        profile = self.get_profile(user_idx)
        if not item_themes:
            return 0.0
        match = max([profile['theme_weights'].get(t, 0.0) for t in item_themes])
        return min(match * self.cfg.perso_match_multiplier, 1.0)
