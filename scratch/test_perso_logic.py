
import numpy as np

def simulate_personalization_v2(n_inter, theme_distribution, item_themes, multiplier=1.5, decay=0.1):
    theme_counts = {}
    for theme_idx, count in enumerate(theme_distribution):
        theme_label = f"Theme_{theme_idx}"
        avg_weight = np.mean([np.exp(-decay * i) for i in range(count)]) if count > 0 else 0
        theme_counts[theme_label] = count * avg_weight
        
    if not theme_counts:
        return {}, 0.0
        
    # NEW: Max-normalization to prevent dilution for heavy users
    max_cnt = max(theme_counts.values())
    theme_weights = {t: c / max_cnt for t, c in theme_counts.items()}
    
    # NEW: Max-matching to prevent multi-theme penalty
    match = max([theme_weights.get(t, 0.0) for t in item_themes])
    score = min(match * multiplier, 1.0)
    
    return theme_weights, score

# Case A: New user, 5 interactions, all Theme 0
print("--- Case A: New user, 5 items in Theme 0 ---")
weights, score = simulate_personalization_v2(5, [5], ["Theme_0"])
print(f"Weight Theme_0: {weights['Theme_0']:.4f}, Score: {score:.4f}")

# Case B: Heavy user, 50/100 items in Theme 0
print("\n--- Case B: Heavy user, 50/100 items in Theme 0 ---")
weights, score = simulate_personalization_v2(100, [50, 50], ["Theme_0"])
print(f"Weight Theme_0: {weights['Theme_0']:.4f}, Score: {score:.4f}")

# Case C: Heavy user, 100 interactions, 5 items in Theme 0 (occasional interest)
print("\n--- Case C: Heavy user, 5/100 items in Theme 0 ---")
weights, score = simulate_personalization_v2(100, [5, 95], ["Theme_0"])
print(f"Weight Theme_0: {weights['Theme_0']:.4f}, Score: {score:.4f}")

# Case D: Item with 2 themes (one matched, one not)
print("\n--- Case D: Item with 2 themes (Matched T0, Unmatched T1) ---")
weights, score = simulate_personalization_v2(5, [5], ["Theme_0", "Theme_1"])
print(f"Score: {score:.4f}")
