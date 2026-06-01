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

    def __init__(self, cfg: PipelineConfig = None, item_limit: int = 0):
        self.cfg = cfg or PipelineConfig()
        # Pre-allocated numpy array indexed by integer item index.
        # Replaces the dict to make sample_multipliers fully vectorized.
        self._counts = np.zeros(item_limit, dtype=np.float64)
        print("[RERANKER] Thompson Sampling active — no hardcoded exposure cap.")

    def sample_multipliers(
        self,
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
        beta  = 1.0 + self._counts                       # fully vectorized — no Python loop
        # np.random.beta requires α,β > 0 — guaranteed by the +1 offsets above.
        multipliers = np.random.beta(alpha, beta)
        return multipliers

    def is_over_exposed(self, item_idx: int, total_users: int, is_campaign: bool = False) -> bool:
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
        count = self._counts[item_idx] if item_idx < len(self._counts) else 0
        return (count / total_users) > self.cfg.campaign_exposure_cap

    def record_selection(self, item_idx: int) -> None:
        """Increment the impression counter for the selected item."""
        if item_idx < len(self._counts):
            self._counts[item_idx] += 1


# Alias for backward compatibility with any code that still imports DiversityReranker.
DiversityReranker = ThompsonReranker
