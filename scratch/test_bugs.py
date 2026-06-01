import numpy as np

# Mocking the pipeline config
class MockConfig:
    relevance_percentile = 10
    candidate_limit = None
    perso_match_multiplier = 1.5
    discovery_jitter_threshold = 0.05
    cold_start_cutoff = 5.0
    moderate_cutoff = 20.0
    cold_weights = (0.65, 0.10, 0.15, 0.10)
    moderate_weights = (0.30, 0.10, 0.50, 0.10)
    dense_weights = (0.15, 0.10, 0.70, 0.05)
    thompson_alpha_scale = 50.0

cfg = MockConfig()
item_limit = 10  # 10 items in total

# Suppose only items 2, 4, 6 were retrieved by LightGCN
# Their DeepFM scores are:
# Item 2: 0.8 (highly relevant)
# Item 4: 0.5 (moderately relevant)
# Item 6: 0.1 (low relevance)
# All other items have -999.0
ai_scores = np.full(item_limit, -999.0, dtype=np.float32)
ai_scores[2] = 0.8
ai_scores[4] = 0.5
ai_scores[6] = 0.1

print("--- ORIGINAL BUGGY LOGIC ---")
# 1. Compute AI Norm
score_range = np.max(ai_scores) - np.min(ai_scores)
ai_norm = (
    (ai_scores - np.min(ai_scores)) / score_range
    if score_range > 0
    else np.zeros_like(ai_scores)
)
print("Normalized AI scores:")
for idx, val in enumerate(ai_norm):
    print(f"  Item {idx}: {val:.6f} (raw: {ai_scores[idx]})")

# 2. Thresholding & Eligibility
threshold = np.percentile(ai_norm, cfg.relevance_percentile)
eligible_indices = np.where(ai_norm >= threshold)[0]
print(f"\nPercentile Threshold (10th): {threshold:.6f}")
print(f"Eligible indices: {eligible_indices.tolist()} (All items became eligible!)")

# 3. Mask definition
mask = np.full(item_limit, -100.0)
mask[eligible_indices] = 0.0

# Mock other features
global_strategic_scores = np.zeros(item_limit)
personalization_scores = np.zeros(item_limit)
popularity_scores = np.zeros(item_limit)
campaign_scores = np.zeros(item_limit)
jitter = np.zeros(item_limit)

# Curation-heavy weights
w_curation, w_strategic, w_personalization, w_popularity = cfg.cold_weights

base_scores = (
    w_curation          * ai_norm
    + w_strategic       * global_strategic_scores
    + w_personalization * personalization_scores
    + w_popularity      * popularity_scores
    + 0.40              * campaign_scores
    + jitter
    + mask
)

# Suppose Item 0 (non-retrieved) and Item 2 (retrieved) get Thompson multipliers:
# Retrieved item 2 gets a multiplier of 0.8
# Non-retrieved item 0 gets a multiplier of 0.01 (crushed because of impressions or chance)
thompson_multipliers = np.ones(item_limit)
thompson_multipliers[0] = 0.01
thompson_multipliers[2] = 0.8

final_scores = base_scores * thompson_multipliers
clipped_scores = np.clip(final_scores, -100.0, 2.0)

print("\nFinal scores (Buggy):")
print(f"  Item 0 (Non-retrieved): Base={base_scores[0]:.2f}, Mult={thompson_multipliers[0]:.2f}, Final={clipped_scores[0]:.4f}")
print(f"  Item 2 (Retrieved, High relevance): Base={base_scores[2]:.2f}, Mult={thompson_multipliers[2]:.2f}, Final={clipped_scores[2]:.4f}")
print(f"Is Item 0 (Non-retrieved) ranked higher than Item 2? {clipped_scores[0] > clipped_scores[2]}")

print("\n--- FIXED LOGIC PROPOSAL ---")
# Only retrieve indices with score > -900
retrieved_indices = np.where(ai_scores > -900.0)[0]
retrieved_scores = ai_scores[retrieved_indices]

fixed_ai_norm = np.zeros_like(ai_scores)
if len(retrieved_scores) > 0:
    score_min = np.min(retrieved_scores)
    score_max = np.max(retrieved_scores)
    score_range = score_max - score_min
    fixed_ai_norm[retrieved_indices] = (
        (retrieved_scores - score_min) / score_range
        if score_range > 0
        else 1.0
    )

print("Fixed Normalized AI scores:")
for idx in range(item_limit):
    print(f"  Item {idx}: {fixed_ai_norm[idx]:.6f} (raw: {ai_scores[idx]})")

if len(retrieved_indices) > 0:
    fixed_threshold = np.percentile(fixed_ai_norm[retrieved_indices], cfg.relevance_percentile)
    fixed_eligible_indices = retrieved_indices[fixed_ai_norm[retrieved_indices] >= fixed_threshold]
else:
    fixed_eligible_indices = np.array([], dtype=int)

print(f"\nFixed Eligible indices: {fixed_eligible_indices.tolist()} (Only retrieved items are eligible)")

fixed_mask = np.full(item_limit, -100.0)
fixed_mask[fixed_eligible_indices] = 0.0

fixed_base_scores = (
    w_curation          * fixed_ai_norm
    + w_strategic       * global_strategic_scores
    + w_personalization * personalization_scores
    + w_popularity      * popularity_scores
    + 0.40              * campaign_scores
    + jitter
)

# Apply mask AFTER multiplication
fixed_final_scores = fixed_base_scores * thompson_multipliers + fixed_mask
fixed_clipped_scores = np.clip(fixed_final_scores, -100.0, 2.0)

print("\nFinal scores (Fixed):")
print(f"  Item 0 (Non-retrieved): Base={fixed_base_scores[0]:.2f}, Mult={thompson_multipliers[0]:.2f}, Final={fixed_clipped_scores[0]:.4f}")
print(f"  Item 2 (Retrieved, High relevance): Base={fixed_base_scores[2]:.2f}, Mult={thompson_multipliers[2]:.2f}, Final={fixed_clipped_scores[2]:.4f}")
print(f"Is Item 0 (Non-retrieved) ranked higher than Item 2? {fixed_clipped_scores[0] > fixed_clipped_scores[2]}")
