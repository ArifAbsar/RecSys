"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 HOW TO TUNE: Edit the USER CONTROL PANEL below, then re-run.
 Set any value to None to use the PipelineConfig default.
 You never need to open any other file.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""



OVERRIDE_N_RECS         = None   
OVERRIDE_PERSO_PCT      = None   
OVERRIDE_CURATION_PCT   = None  
OVERRIDE_PROMO_PCT      = None  
BATCH_SIZE              = 64    

RELEVANCE_PERCENTILE    = 10    # drop bottom X%; keep top (100-X)%
CANDIDATE_LIMIT         = None  # None = evaluate entire eligible catalog after percentile gate

COLD_WEIGHTS            = (0.65, 0.10, 0.15, 0.10)
MODERATE_WEIGHTS        = (0.30, 0.10, 0.50, 0.10)
DENSE_WEIGHTS           = (0.15, 0.10, 0.70, 0.05)
COLD_START_CUTOFF       = 5.0    
MODERATE_CUTOFF         = 20.0   

PERSO_LABEL_PERCENTILE          = 80    
CURATION_DISCOVERY_PERCENTILE   = 60    
PERSO_MIN_THRESHOLD             = 0.20 
PERSO_MATCH_MULTIPLIER          = 1.5  
PAST_ITEMS_WINDOW               = 10   
TOP_THEMES                      = 5    
RECENCY_DECAY                   = 0.1   

THOMPSON_ALPHA_SCALE    = 50.0  # control exploration vs exploitation trade-off
CAMPAIGN_EXPOSURE_CAP   = 0.60  
COLD_START_MIN_USERS    = 50    
PER_THEME_CAP           = 2     

DISCOVERY_JITTER_THRESHOLD = 0.05 

ENABLE_PROMOTION            = True  
PROMO_THRESHOLD_PERCENTILE  = 93    

CLUSTERING_MODEL            = 'all-MiniLM-L6-v2'
K_MIN                       = 5    
K_MAX                       = 50   
SECONDARY_CLUSTER_THRESHOLD = 0.85 
SILHOUETTE_SAMPLE_SIZE      = 2000  



import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

from pipeline.inference import run_stage2_semantic_inference

if __name__ == "__main__":
    run_stage2_semantic_inference(
        override_n_recs=OVERRIDE_N_RECS,
        override_perso_pct=OVERRIDE_PERSO_PCT,
        override_curation_pct=OVERRIDE_CURATION_PCT,
        override_promo_pct=OVERRIDE_PROMO_PCT,
        batch_size=BATCH_SIZE,
        relevance_percentile=RELEVANCE_PERCENTILE,
        candidate_limit=CANDIDATE_LIMIT,
        cold_weights=COLD_WEIGHTS,
        moderate_weights=MODERATE_WEIGHTS,
        dense_weights=DENSE_WEIGHTS,
        cold_start_cutoff=COLD_START_CUTOFF,
        moderate_cutoff=MODERATE_CUTOFF,
        perso_label_percentile=PERSO_LABEL_PERCENTILE,
        curation_discovery_percentile=CURATION_DISCOVERY_PERCENTILE,
        perso_min_threshold=PERSO_MIN_THRESHOLD,
        perso_match_multiplier=PERSO_MATCH_MULTIPLIER,
        past_items_window=PAST_ITEMS_WINDOW,
        top_themes=TOP_THEMES,
        recency_decay=RECENCY_DECAY,
        thompson_alpha_scale=THOMPSON_ALPHA_SCALE,
        campaign_exposure_cap=CAMPAIGN_EXPOSURE_CAP,
        cold_start_min_users=COLD_START_MIN_USERS,
        per_theme_cap=PER_THEME_CAP,
        discovery_jitter_threshold=DISCOVERY_JITTER_THRESHOLD,
        enable_promotion=ENABLE_PROMOTION,
        promo_threshold_percentile=PROMO_THRESHOLD_PERCENTILE,
        clustering_model=CLUSTERING_MODEL,
        k_min=K_MIN,
        k_max=K_MAX,
        secondary_cluster_threshold=SECONDARY_CLUSTER_THRESHOLD,
        silhouette_sample_size=SILHOUETTE_SAMPLE_SIZE,
    )