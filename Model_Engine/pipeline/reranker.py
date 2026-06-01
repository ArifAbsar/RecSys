"""
pipeline/reranker.py
=====================
ThompsonReranker — Multi-Armed Bandit diversity via Beta distribution sampling.

Replaces the hardcoded adaptive exposure cap (clip(10/sqrt(C), 0.04, 0.30))
with a mathematically pure Thompson Sampling approach. No arbitrary heuristic
bounds are used. The Beta distribution naturally suppresses over-exposed items
and boosts cold-start items through statistical probability.

How it works:
  - Each item is treated as a slot machine ("arm").
  - Alpha (α): Encodes how good the item is (AI score + popularity).
  - Beta  (β): Encodes batch fatigue — grows each time the item is selected.
  - A score multiplier is sampled from Beta(α, β) for every item.
  - As β grows (item gets recommended more), the curve collapses near 0,
    naturally crushing that item's final score without any hard cap.
  - Cold-start items (β = 1, wide uncertain curve) occasionally roll a high
    multiplier by chance, guaranteeing organic exploration.

This makes the system stochastic (non-deterministic). Set
np.random.seed(42) at the top of run_stage2.py for reproducible testing.
"""

import numpy as np

from .config import PipelineConfig


# Backward-compatible alias so existing imports don't break.
DiversityReranker = None  # overwritten below after class definition


class ThompsonReranker:
    """
    Multi-Armed Bandit reranker using Beta distribution sampling.

    Completely replaces the hardcoded exposure cap.
    Zero heuristic bounds — the math self-regulates.
    """

    def __init__(self, cfg: PipelineConfig = None):
        self.cfg = cfg or PipelineConfig()
        # Tracks how many times each item_id has been selected in this batch.
        self.global_counts: dict = {}
        print("[RERANKER] Thompson Sampling active — no hardcoded exposure cap.")

    def sample_multipliers(
        self,
        item_ids: list,
        ai_scores: np.ndarray,
        pop_scores: np.ndarray,
    ) -> np.ndarray:
        """
        Compute a Thompson Sampling multiplier for every item in one vectorized pass.

        α = 1 + scale * (ai_score + pop_score)
            → High-confidence items have a large α → curve skews right → high multipliers.

        β = 1 + global_impressions
            → Every past selection increments β → curve collapses toward 0 as item
              is over-recommended. Cold-start items stay at β=1 (wide, exploratory curve).

        Returns an array of floats in [0, 1] — multiply against base_score in scoring.py.
        """
        scale = self.cfg.thompson_alpha_scale
        alpha = 1.0 + scale * (ai_scores + pop_scores)  # vectorized
        beta = np.array(
            [1.0 + self.global_counts.get(iid, 0) for iid in item_ids],
            dtype=np.float64,
        )
        # np.random.beta requires α,β > 0 — guaranteed by the +1 offsets above.
        multipliers = np.random.beta(alpha, beta)
        return multipliers

    def is_over_exposed(self, item_id, total_users: int, is_campaign: bool = False) -> bool:
        """
        Thompson Sampling self-regulates exposure organically, so this hard gate
        is kept only as a last-resort safety net for campaign items.
        Campaign items still get a configurable ceiling to prevent full catalog
        monopolization during heavy promotional periods.
        """
        if not is_campaign:
            return False  # pure Thompson — no hard cap for regular items
        if total_users < self.cfg.cold_start_min_users:
            return False
        count = self.global_counts.get(item_id, 0)
        return (count / total_users) > self.cfg.campaign_exposure_cap

    def record_selection(self, item_id) -> None:
        """Increment the impression counter for the selected item."""
        self.global_counts[item_id] = self.global_counts.get(item_id, 0) + 1


# Alias for backward compatibility with any code that still imports DiversityReranker.
DiversityReranker = ThompsonReranker
