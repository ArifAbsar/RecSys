"""
pipeline/scoring.py
====================
Scoring utilities:
  - compute_adaptive_weights : dataset-level tier selection (cold/moderate/dense)
  - compute_per_user_scores  : per-user score aggregation extracted from the main loop

Extracted verbatim from run_stage2.py (lines 410-731).
No logic has been changed — code is identical, just wrapped in functions.
"""

import numpy as np

from .config import PipelineConfig


def compute_adaptive_weights(
    dataset, cfg: PipelineConfig = None
) -> tuple[float, float, float, float]:
    """
    Weights shift based on how data-rich the dataset is (thresholds & tiers
    all come from PipelineConfig so operators can tune without touching code).

    Returns (w_curation, w_strategic, w_personalization, w_popularity).
    """
    cfg = cfg or PipelineConfig()
    n_users = max(dataset.user_num - 1, 1)
    avg_inter = len(dataset.inter_feat) / n_users

    if avg_inter < cfg.cold_start_cutoff:
        w = cfg.cold_weights
        tier = "cold"
    elif avg_inter < cfg.moderate_cutoff:
        w = cfg.moderate_weights
        tier = "moderate"
    else:
        w = cfg.dense_weights
        tier = "dense"

    print(
        f"[WEIGHTS] avg_interactions={avg_inter:.1f} tier={tier} -> "
        f"w_ai={w[0]}, w_strat={w[1]}, w_perso={w[2]}, w_pop={w[3]}"
    )
    return w   # (w_curation, w_strat, w_perso, w_pop)


def compute_per_user_scores(
    ai_scores,
    user_idx: int,
    item_limit: int,
    cfg: PipelineConfig,
    global_strategic_scores: np.ndarray,
    popularity_scores: np.ndarray,
    is_campaign_item: np.ndarray,
    item_ids: list,
    item_themes_map: list,
    profiler,
    reranker,
    total_users: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute final scores for all items for a single user.

    Extracted verbatim from run_stage2.py (lines 655-731).
    Returns (final_scores, ai_norm, eligible_indices, personalization_scores).
    """

    retrieved_indices = np.where(ai_scores > -900.0)[0]
    retrieved_scores = ai_scores[retrieved_indices]

    ai_norm = np.zeros_like(ai_scores)
    if len(retrieved_scores) > 0:
        score_min = np.min(retrieved_scores)
        score_max = np.max(retrieved_scores)
        score_range = score_max - score_min
        ai_norm[retrieved_indices] = (
            (retrieved_scores - score_min) / score_range
            if score_range > 0
            else 1.0
        )

    if len(retrieved_indices) > 0:
        threshold = np.percentile(ai_norm[retrieved_indices], cfg.relevance_percentile)
        eligible_indices = retrieved_indices[ai_norm[retrieved_indices] >= threshold]
        if cfg.candidate_limit is not None:
            top_k = np.argsort(ai_norm[eligible_indices])[::-1][:cfg.candidate_limit]
            eligible_indices = eligible_indices[top_k]
    else:
        eligible_indices = np.array([], dtype=int)

    user_profile = profiler.get_profile(user_idx)

    seen = user_profile.get("past_item_indices", set())
    if seen and len(eligible_indices) > 0:
        keep = np.array([int(idx) not in seen for idx in eligible_indices], dtype=bool)
        eligible_indices = eligible_indices[keep]

    theme_weights   = user_profile.get('theme_weights', {})
    perso_mult      = cfg.perso_match_multiplier
    personalization_scores = np.zeros(item_limit)
    if len(eligible_indices) > 0:
        for idx in eligible_indices:
            themes = item_themes_map[idx]
            if themes:
                match = max(theme_weights.get(t, 0.0) for t in themes)
                personalization_scores[idx] = min(match * perso_mult, 1.0)

    if len(retrieved_indices) > 0 and np.var(ai_norm[retrieved_indices]) < 0.001:
        ai_norm[retrieved_indices] += np.random.normal(0, 0.001, size=len(retrieved_indices))

    mask = np.full(item_limit, -100.0)
    mask[eligible_indices] = 0.0

    jitter = np.zeros(item_limit)
    perso_max = float(np.max(personalization_scores[eligible_indices])) if len(eligible_indices) > 0 else 0.0
    if len(eligible_indices) > 0 and perso_max < cfg.discovery_jitter_threshold:
        jitter[eligible_indices] = np.random.uniform(
            0, cfg.discovery_jitter_threshold, size=len(eligible_indices)
        )

    avg_inter = len(user_profile.get("past_item_indices", []))
    if avg_inter < cfg.cold_start_cutoff:
        weights = cfg.cold_weights
    elif avg_inter < cfg.moderate_cutoff:
        weights = cfg.moderate_weights
    else:
        weights = cfg.dense_weights

    w_curation, w_strategic, w_personalization, w_popularity = weights

    campaign_scores = np.zeros(item_limit)
    campaign_scores[is_campaign_item] = 1.0

    #Base score (weighted sum of all signals)
    base_scores = (
        w_curation          * ai_norm
        + w_strategic       * global_strategic_scores
        + w_personalization * personalization_scores
        + w_popularity      * popularity_scores
        + 0.40              * campaign_scores
        + jitter
    )

    thompson_multipliers = reranker.sample_multipliers(
        item_ids=item_ids,
        ai_scores=ai_norm,
        pop_scores=popularity_scores,
    )

    final_scores = base_scores * thompson_multipliers + mask
    final_scores = np.clip(final_scores, -100.0, 2.0)

    return final_scores, ai_norm, eligible_indices, personalization_scores
