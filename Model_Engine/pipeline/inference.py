"""
pipeline/inference.py
======================
run_stage2_semantic_inference — the main orchestrator.

This function is now thin: it calls helpers from the other pipeline modules
in sequence. All logic is identical to the original run_stage2.py —
only the structure has changed.
"""

import os
import json
import argparse

import torch
import numpy as np
from recbole.data.interaction import Interaction
from scipy.sparse import dok_matrix

if not hasattr(dok_matrix, '_update'):
    dok_matrix._update = dok_matrix.update

from .config import PipelineConfig
from .theme_discovery import AutoThemeDiscovery
from .user_profiler import UserInterestProfiler
from .reranker import DiversityReranker
from .scoring import compute_adaptive_weights, compute_per_user_scores
from .selector import select_recommendations
from .data_utils import (
    resolve_paths,
    load_intelligence_map,
    load_recbole_model,
    extract_item_metadata,
    compute_popularity_scores,
    build_campaign_flags,
)


def _apply_overrides(cfg: PipelineConfig, **overrides) -> None:
    """Apply non-None control panel values on top of PipelineConfig defaults."""
    mapping = {
        "batch_size":                    "batch_size",
        "relevance_percentile":          "relevance_percentile",
        "candidate_limit":               "candidate_limit",
        "cold_weights":                  "cold_weights",
        "moderate_weights":              "moderate_weights",
        "dense_weights":                 "dense_weights",
        "cold_start_cutoff":             "cold_start_cutoff",
        "moderate_cutoff":               "moderate_cutoff",
        "perso_label_percentile":        "perso_label_percentile",
        "curation_discovery_percentile": "curation_discovery_percentile",
        "perso_min_threshold":           "perso_min_threshold",
        "perso_match_multiplier":        "perso_match_multiplier",
        "past_items_window":             "past_items_window",
        "top_themes":                    "top_themes",
        "recency_decay":                 "recency_decay",
        "penalty_strength":              None,  # removed — Thompson Sampling replaces this
        "diversity_penalty_weight":      None,  # removed — Thompson Sampling replaces this
        "thompson_alpha_scale":          "thompson_alpha_scale",
        "exposure_cap_min":              None,  # removed — no hardcoded cap anymore
        "exposure_cap_max":              None,  # removed — no hardcoded cap anymore
        "campaign_exposure_cap":         "campaign_exposure_cap",
        "cold_start_min_users":          "cold_start_min_users",
        "per_theme_cap":                 "per_theme_cap",
        "discovery_jitter_threshold":    "discovery_jitter_threshold",
        "enable_promotion":              "enable_promotion",
        "promo_threshold_percentile":    "promo_threshold_percentile",
        "clustering_model":              "clustering_model",
        "k_min":                         "k_min",
        "k_max":                         "k_max",
        "secondary_cluster_threshold":   "secondary_cluster_threshold",
        "silhouette_sample_size":        "silhouette_sample_size",
    }
    for kwarg_key, cfg_attr in mapping.items():
        val = overrides.get(kwarg_key)
        if val is not None and cfg_attr is not None:
            setattr(cfg, cfg_attr, val)


def _apply_business_goals(cfg: PipelineConfig, intelligence_map: dict,
                           override_n_recs, override_perso_pct,
                           override_curation_pct, override_promo_pct) -> None:
    """
    Parse CLI args + rec_config.json business_goals, then mutate cfg.
    Extracted verbatim from run_stage2.py (lines 467-513).
    """
    parser = argparse.ArgumentParser(description='Adaptive Semantic Inference Engine')
    parser.add_argument('--n',       type=int, help='Number of recommendations')
    parser.add_argument('--perso',   type=int, help='Personalization percentage (0-100)')
    parser.add_argument('--curation',type=int, help='Curation percentage (0-100)')
    parser.add_argument('--promo',   type=int, help='Promotion percentage (0-100)')
    args, unknown = parser.parse_known_args()

    goals = intelligence_map.get("business_goals", {})

    if override_n_recs is not None:
        cfg.n_recs = override_n_recs
    elif args.n is not None:
        cfg.n_recs = args.n
    else:
        cfg.n_recs = goals.get("total_recommendations", 10)

    p_pct = (override_perso_pct    if override_perso_pct    is not None else (args.perso    if args.perso    is not None else goals.get("personalization_pct", 70))) / 100.0
    c_pct = (override_curation_pct if override_curation_pct is not None else (args.curation if args.curation is not None else goals.get("curation_pct",       20))) / 100.0
    r_pct = (override_promo_pct    if override_promo_pct    is not None else (args.promo    if args.promo    is not None else goals.get("promotion_pct",       10))) / 100.0

    cfg.promo_quota    = max(0, int(round(r_pct * cfg.n_recs)))
    cfg.curation_quota = int(round(c_pct * cfg.n_recs))
    cfg.perso_quota    = cfg.n_recs - cfg.promo_quota - cfg.curation_quota

    if cfg.perso_quota < 0:
        cfg.perso_quota = 0

    cfg.enable_promotion              = True
    cfg.perso_label_percentile        = 92
    cfg.curation_discovery_percentile = 70

    pop_w   = 0.05
    strat_w = 0.10   
    rem     = 1.0 - pop_w - strat_w  
    cfg.dense_weights    = (c_pct * rem,            strat_w, p_pct * rem,            pop_w)
    cfg.moderate_weights = (max(c_pct, 0.35) * rem, strat_w, min(p_pct, 0.50) * rem, pop_w + 0.05)
    cfg.cold_weights     = (0.60,                   strat_w, 0.10,                   0.10)

    print(f"[BUSINESS] Admin Strategy: {cfg.n_recs} Recs | {p_pct*100:.0f}/{c_pct*100:.0f}/{r_pct*100:.0f}")
    print(f"           Dynamic Quotas -> Perso: {cfg.perso_quota}, Curation: {cfg.curation_quota}, Promo: {cfg.promo_quota}")
    print(f"           Dense Weights  -> ai={cfg.dense_weights[0]:.2f} strat={cfg.dense_weights[1]:.2f} perso={cfg.dense_weights[2]:.2f} pop={cfg.dense_weights[3]:.2f}")


def _run_batch_inference(retrieval_model, ranking_model, batch_tokens, dataset, item_rows, item_limit, config, cfg):
    """
    Two-Stage Architecture:
    1. Retrieval: LightGCN predicts scores for all items, selects Top K.
    2. Ranking: DeepFM predicts precise scores only for the Top K candidates.
    Returns (all_user_indices, batch_ai_scores).
    """
    all_user_indices = [
        dataset.token2id(dataset.uid_field, t) for t in batch_tokens
    ]

    # --- 1. RETRIEVAL (LightGCN) ---
    inter_retrieval = Interaction({
        config['USER_ID_FIELD']: torch.tensor(all_user_indices).to(retrieval_model.device)
    })
    
    with torch.no_grad():
        scores_retrieval = retrieval_model.full_sort_predict(inter_retrieval)
        scores_retrieval = scores_retrieval.view(len(all_user_indices), -1)
        k = min(cfg.retrieval_top_k, item_limit) if cfg.retrieval_top_k is not None else item_limit
        _, topk_idx = torch.topk(scores_retrieval, k=k)

    # --- 2. RANKING (DeepFM) ---
    batch_ai_scores = np.full((len(all_user_indices), item_limit), -999.0, dtype=np.float32)
    flat_item_indices = topk_idx.flatten().cpu().numpy()

    batch_tensors: dict = {}
    expected_fields = set(ranking_model.token_field_names)
    user_feats = dataset.get_user_feature()[all_user_indices]

    for field in user_feats.interaction.keys():
        if field in expected_fields:
            f_idx = ranking_model.token_field_names.index(field)
            limit = ranking_model.token_field_dims[f_idx]
            batch_tensors[field] = torch.clamp(
                user_feats[field].repeat_interleave(k), 0, limit - 1
            ).to(config['device'])

    item_subset_rows = item_rows[flat_item_indices]
    for field in item_subset_rows.interaction.keys():
        if field in expected_fields:
            f_idx = ranking_model.token_field_names.index(field)
            limit = ranking_model.token_field_dims[f_idx]
            batch_tensors[field] = torch.clamp(
                item_subset_rows[field], 0, limit - 1
            ).to(config['device'])

    inter = Interaction(batch_tensors)
    with torch.no_grad():
        rank_scores = ranking_model.predict(inter).cpu().numpy().reshape(len(all_user_indices), k)

    # Scatter back to the padded array
    topk_idx_np = topk_idx.cpu().numpy()
    for i in range(len(all_user_indices)):
        batch_ai_scores[i, topk_idx_np[i]] = rank_scores[i]

    return all_user_indices, batch_ai_scores


def run_stage2_semantic_inference(

    override_n_recs: int        = None,
    override_perso_pct: int     = None,
    override_curation_pct: int  = None,
    override_promo_pct: int     = None,
    **overrides,
) -> None:
    """
    Main entry point for the Adaptive Semantic Inference Pipeline.
    All arguments are optional — omit any to use the PipelineConfig default.
    """

    cfg = PipelineConfig()

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    paths    = resolve_paths(cfg, base_dir)

    intelligence_map = load_intelligence_map(paths["mapping_path"])
    _apply_business_goals(
        cfg, intelligence_map,
        override_n_recs, override_perso_pct, override_curation_pct, override_promo_pct,
    )
    _apply_overrides(cfg, **overrides)

    print("\n[INTELLIGENCE] Starting Adaptive Semantic Inference...")
    
    # 1. Load Retrieval Model
    retrieval_config_path = os.path.join(paths["base_dir"], "Model_Engine", cfg.retrieval_config_filename)
    ret_config, _, retrieval_model = load_recbole_model(
        retrieval_config_path, paths["cp_dir"], cfg.retrieval_model_name
    )

    # 2. Load Ranking Model
    config, dataset, ranking_model = load_recbole_model(
        paths["config_path"], paths["cp_dir"], cfg.base_model_name
    )

    item_limit       = dataset.item_num
    item_rows        = dataset.get_item_feature()[:item_limit]
    all_users        = list(dataset.field2token_id[dataset.uid_field])
    users_to_process = all_users[1:]
    np.random.shuffle(users_to_process)
    production_output: dict = {}

    item_ids, descriptions, item_id_to_desc = extract_item_metadata(dataset, intelligence_map)
    print(f"[ENGINE] Processing {len(users_to_process)} users...")

    theme_engine = AutoThemeDiscovery(
        model_name=cfg.clustering_model,
        n_clusters='auto',
        k_min=cfg.k_min,
        k_max=min(cfg.k_max, item_limit // 5),
        secondary_cluster_threshold=cfg.secondary_cluster_threshold,
        cfg=cfg,
    )
    theme_engine.fit(descriptions)
    item_themes_map         = theme_engine.build_theme_map(item_limit)
    global_strategic_scores = theme_engine.build_strategic_scores(item_limit)

    promoted_ids, clearance_ids, is_campaign_item = build_campaign_flags(
        item_ids, intelligence_map, item_limit
    )
    profiler = UserInterestProfiler(dataset, item_id_to_desc, item_themes_map, cfg=cfg)
    reranker = DiversityReranker(cfg=cfg)  # ThompsonReranker — no catalog_size needed
    compute_adaptive_weights(dataset, cfg=cfg)

    popularity_scores = compute_popularity_scores(dataset, item_limit)

    total_users     = len(users_to_process)
    processed_count = 0
    n_batches       = -(-total_users // cfg.batch_size)

    for b_idx in range(0, len(users_to_process), cfg.batch_size):
        batch_tokens = users_to_process[b_idx: b_idx + cfg.batch_size]
        print(f" > Batch {b_idx // cfg.batch_size + 1}/{n_batches}  ({processed_count} done)")

        all_user_indices, batch_ai_scores = _run_batch_inference(
            retrieval_model, ranking_model, batch_tokens, dataset, item_rows, item_limit, config, cfg
        )

        for i, user_token in enumerate(batch_tokens):
            user_idx  = all_user_indices[i]
            ai_scores = batch_ai_scores[i]

            final_scores, ai_norm, eligible_indices, personalization_scores = compute_per_user_scores(
                ai_scores=ai_scores,
                user_idx=user_idx,
                item_limit=item_limit,
                cfg=cfg,
                global_strategic_scores=global_strategic_scores,
                popularity_scores=popularity_scores,
                is_campaign_item=is_campaign_item,
                item_ids=item_ids,
                item_themes_map=item_themes_map,
                profiler=profiler,
                reranker=reranker,
                total_users=total_users,
            )

            user_profile   = profiler.get_profile(user_idx)
            eligible_sorted = eligible_indices[
                np.argsort(final_scores[eligible_indices])[::-1]
            ]

            top_recs = select_recommendations(
                eligible_sorted=eligible_sorted,
                final_scores=final_scores,
                ai_norm=ai_norm,
                personalization_scores=personalization_scores,
                global_strategic_scores=global_strategic_scores,
                popularity_scores=popularity_scores,
                item_ids=item_ids,
                item_themes_map=item_themes_map,
                promoted_ids=promoted_ids,
                clearance_ids=clearance_ids,
                is_campaign_item=is_campaign_item,
                item_id_to_desc=item_id_to_desc,
                user_profile=user_profile,
                reranker=reranker,
                cfg=cfg,
                total_users=total_users,
            )

            production_output[user_token] = top_recs
            processed_count += 1

    with open(paths["output_path"], 'w') as outputs_file:
        json.dump(production_output, outputs_file, indent=4)

    print(f"\n[SUCCESS] Saved {len(production_output)} user records -> {paths['output_path']}")
    print("-" * 80)
    first_user  = users_to_process[0]
    sample_recs = production_output.get(first_user, [])
    print(f"Sample - User {first_user} ({len(sample_recs)} recs):")
    for rank, rec in enumerate(sample_recs, 1):
        print(
            f"  {rank:>2}. {rec['item_id']:<12}"
            f"  {rec['description'][:30]:<32}"
            f"  [{rec['recommendation_type']:<15}]  "
            f"{rec['explanation']['reason']}"
        )
