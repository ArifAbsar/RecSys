import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set plotting style for academic papers
plt.style.use('ggplot')
sns.set_context("paper", font_scale=1.5)
sns.set_style("whitegrid")

OUTPUT_DIR = "output"
PLOTS_DIR = os.path.join(OUTPUT_DIR, "paper_plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

def load_data():
    with open(os.path.join(OUTPUT_DIR, "final_recommendations.json"), 'r') as f:
        data = json.load(f)
    return data

def calculate_gini(array):
    """Calculate the Gini coefficient of a numpy array."""
    array = array.flatten().astype(np.float64)
    if np.amin(array) < 0:
        array -= np.amin(array)
    array += 0.0000001
    array = np.sort(array)
    index = np.arange(1, array.shape[0] + 1)
    n = array.shape[0]
    return ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array)))

def plot_catalog_coverage_and_gini(data):
    print("Generating Catalog Coverage & Gini Plot...")
    item_counts = {}
    total_recs = 0
    
    for user_id, recs in data.items():
        for rec in recs:
            item_id = rec['item_id']
            item_counts[item_id] = item_counts.get(item_id, 0) + 1
            total_recs += 1
            
    counts_array = np.array(list(item_counts.values()))
    gini = calculate_gini(counts_array)
    
    # Sort for long-tail plot
    sorted_counts = np.sort(counts_array)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(len(sorted_counts)), sorted_counts, color='b', linewidth=2)
    plt.fill_between(np.arange(len(sorted_counts)), sorted_counts, color='b', alpha=0.3)
    plt.title(f"Item Exposure Distribution (Thompson Sampling)\nGini Coefficient: {gini:.3f}", fontweight='bold')
    plt.xlabel("Items (Ranked by Frequency)")
    plt.ylabel("Number of Impressions")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "1_catalog_coverage_gini.png"), dpi=300)
    plt.close()

def plot_quota_adherence(data):
    print("Generating Quota Adherence Plot...")
    type_counts = {"personalization": 0, "curation": 0, "promotion": 0}
    
    for user_id, recs in data.items():
        for rec in recs:
            r_type = rec.get('recommendation_type', 'curation')
            if r_type in type_counts:
                type_counts[r_type] += 1
                
    total = sum(type_counts.values())
    if total == 0: return
    
    labels = list(type_counts.keys())
    sizes = [count / total * 100 for count in type_counts.values()]
    colors = ['#ff9999','#66b3ff','#99ff99']
    explode = (0.05, 0.05, 0.05) 
    
    plt.figure(figsize=(8, 8))
    plt.pie(sizes, explode=explode, labels=[l.capitalize() for l in labels], colors=colors, 
            autopct='%1.1f%%', shadow=True, startangle=140, textprops={'fontsize': 14, 'weight': 'bold'})
    plt.title("Business Quota Adherence (Greedy Selector)", fontweight='bold')
    plt.savefig(os.path.join(PLOTS_DIR, "2_quota_adherence.png"), dpi=300)
    plt.close()

def plot_signal_contribution(data):
    print("Generating Signal Contribution Plot...")
    
    ai_scores = []
    strat_scores = []
    perso_scores = []
    pop_scores = []
    
    for user_id, recs in data.items():
        for rec in recs:
            s = rec.get('scores', {})
            ai_scores.append(s.get('ai_relevance', 0))
            strat_scores.append(s.get('strategic_boost', 0))
            perso_scores.append(s.get('personalization_match', 0))
            pop_scores.append(s.get('popularity_score', 0))
            
    means = [np.mean(ai_scores), np.mean(strat_scores), np.mean(perso_scores), np.mean(pop_scores)]
    labels = ['AI Relevance', 'Strategic Boost', 'Personalization', 'Popularity']
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=labels, y=means, palette="viridis")
    plt.title("Average Signal Contribution to Final Score", fontweight='bold')
    plt.ylabel("Normalized Score Value")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "3_signal_contributions.png"), dpi=300)
    plt.close()

def plot_intra_list_diversity(data):
    print("Generating Intra-List Diversity Plot...")
    
    unique_themes_per_user = []
    for user_id, recs in data.items():
        themes_in_list = set()
        for rec in recs:
            # Extract themes if available
            themes = rec.get('explanation', {}).get('matched_user_interests', [])
            for t in themes:
                themes_in_list.add(t)
        unique_themes_per_user.append(len(themes_in_list))
        
    # Filter out 0s if some users had no themes matched (pure discovery)
    unique_themes_per_user = [x for x in unique_themes_per_user if x > 0]
    
    if unique_themes_per_user:
        plt.figure(figsize=(10, 6))
        sns.histplot(unique_themes_per_user, bins=10, kde=True, color='purple')
        plt.axvline(np.mean(unique_themes_per_user), color='r', linestyle='dashed', linewidth=2, label=f'Mean: {np.mean(unique_themes_per_user):.1f}')
        plt.title("Intra-List Thematic Diversity per User", fontweight='bold')
        plt.xlabel("Number of Unique Themes in Top-10 List")
        plt.ylabel("Number of Users")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "4_intra_list_diversity.png"), dpi=300)
        plt.close()

if __name__ == "__main__":
    print("Loading recommendation data...")
    try:
        data = load_data()
        print(f"Loaded data for {len(data)} users.")
        
        plot_catalog_coverage_and_gini(data)
        plot_quota_adherence(data)
        plot_signal_contribution(data)
        plot_intra_list_diversity(data)
        
        print(f"\\nSUCCESS! All plots saved to: {PLOTS_DIR}")
        print("You can insert these images directly into your research paper.")
    except Exception as e:
        print(f"Error: {e}")
