"""
pipeline/selector.py
=====================
select_recommendations — quota-based greedy recommendation selector.

Extracted verbatim from run_stage2.py (lines 738-858).
No logic has been changed — code is identical, just wrapped in a function.
"""

import numpy as np

from .config import PipelineConfig


def select_recommendations(
    eligible_sorted: np.ndarray,
    final_scores: np.ndarray,
    ai_norm: np.ndarray,
    personalization_scores: np.ndarray,
    global_strategic_scores: np.ndarray,
    popularity_scores: np.ndarray,
    item_ids: list,
    item_themes_map: list,
    promoted_ids: set,
    clearance_ids: set,
    is_campaign_item: np.ndarray,
    item_id_to_desc: dict,
    user_profile: dict,
    reranker,
    cfg: PipelineConfig,
    total_users: int,
) -> list[dict]:
    """
    Greedy quota-based selection of top-N recommendations for a single user.

    Hierarchy: Promotion > Personalization > Curation.
    Falls back gracefully when a quota type is exhausted.

    Returns a list of recommendation dicts ready for JSON output.
    """
    user_theme_weights = user_profile.get('theme_weights', {})

    top_recs: list[dict] = []
    theme_usage: dict[str, int] = {}

    quotas = {
        "personalization": cfg.perso_quota,
        "curation":        cfg.curation_quota,
        "promotion":       cfg.promo_quota,
    }
    counts = {value: 0 for value in quotas}

    dynamic_threshold = max(
        cfg.perso_min_threshold,
        np.percentile(personalization_scores[eligible_sorted], cfg.perso_label_percentile)
        if len(eligible_sorted) > 0 else cfg.perso_min_threshold
    )
    curation_threshold = (
        np.percentile(ai_norm[eligible_sorted], cfg.curation_discovery_percentile)
        if len(eligible_sorted) > 0 else 0.0
    )

    for idx in eligible_sorted:
        if len(top_recs) >= cfg.n_recs:
            break

        # Check exposure cap
        if reranker.is_over_exposed(
            item_ids[idx], total_users, bool(is_campaign_item[idx])
        ):
            continue

        themes = item_themes_map[idx]
        primary_theme = themes[0] if themes else "General"

        is_admin_push = (
            str(item_ids[idx]) in promoted_ids
            or str(item_ids[idx]) in clearance_ids
        )
        is_promo = cfg.enable_promotion and is_admin_push

        is_perso = (
            not is_promo
            and (personalization_scores[idx] >= dynamic_threshold
                 or any(t in user_profile.get('top_themes', []) for t in themes))
        )

        is_curation = (
            not is_promo
            and not is_perso
            and ai_norm[idx] >= curation_threshold
        )

        rec_type = (
            "promotion"       if is_promo else
            "personalization" if is_perso else
            "curation"        if is_curation else
            "curation"
        )

        other_types_exhausted = all(
            counts[k] >= quotas[k] for k in quotas if k != rec_type
        )
        if counts[rec_type] < quotas[rec_type] or other_types_exhausted:
            if theme_usage.get(primary_theme, 0) >= cfg.per_theme_cap:
                continue

            business_boosted = bool(is_campaign_item[idx]) and cfg.enable_promotion

            campaign_type = "none"
            if str(item_ids[idx]) in promoted_ids:
                campaign_type = "promoted"
            elif str(item_ids[idx]) in clearance_ids:
                campaign_type = "clearance"

            matched_themes = [t for t in themes if t in user_theme_weights]
            if is_promo:
                reason = f"{campaign_type.capitalize()}: Featured brand selected for you"
            elif matched_themes:
                reason = f"Personalized: Matches your interest in {', '.join(matched_themes[:2])}"
            elif is_perso:
                reason = "Matches your browsing patterns"
            else:
                reason = "Discovery: A new item the AI thinks you will love"

            top_recs.append({
                "item_id":             str(item_ids[idx]),
                "description":         item_id_to_desc.get(item_ids[idx], "No Description"),
                "recommendation_type": rec_type,
                "business_boosted":    business_boosted,
                "campaign_type":       campaign_type,
                "scores": {
                    "final_score":           round(float(final_scores[idx]),            4),
                    "ai_relevance":          round(float(ai_norm[idx]),                 4),
                    "strategic_boost":       round(float(global_strategic_scores[idx]), 4),
                    "personalization_match": round(float(personalization_scores[idx]),  4),
                    "popularity_score":      round(float(popularity_scores[idx]),       4),
                },
                "explanation": {
                    "matched_user_interests": matched_themes,
                    "reason": reason,
                },
            })
            counts[rec_type] += 1
            reranker.record_selection(item_ids[idx])
            theme_usage[primary_theme] = theme_usage.get(primary_theme, 0) + 1

    return top_recs
