import json
import csv
import os

# Paths
input_file = r'e:\docker-crash-course\RecSys\output\final_recommendations.json'
output_file = r'e:\docker-crash-course\RecSys\output\final_recommendations.csv'

def flatten_recommendations():
    if not os.path.exists(input_file):
        print(f"Error: Could not find {input_file}")
        return

    print(f"Reading {input_file}...")
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Updated headers to match run_stage2.py actual output structure
    headers = [
        'user_id', 
        'item_id', 
        'description', 
        'recommendation_type',
        'business_boosted',
        'campaign_type',
        'final_score', 
        'ai_relevance', 
        'strategic_boost', 
        'personalization_match',
        'popularity_score',
        'reason',
        'matched_interests'
    ]

    print(f"Flattening data for Looker Studio...")
    count = 0
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        
        for user_id, recs in data.items():
            for rec in recs:
                scores = rec.get('scores', {})
                explanation = rec.get('explanation', {})
                
                # Correctly mapping from the nested structure
                row = {
                    'user_id': user_id,
                    'item_id': rec.get('item_id'),
                    'description': rec.get('description'),
                    'recommendation_type': rec.get('recommendation_type'),  # Moved to root
                    'business_boosted': rec.get('business_boosted'),
                    'campaign_type': rec.get('campaign_type', 'none'),
                    'final_score': scores.get('final_score'),
                    'ai_relevance': scores.get('ai_relevance'),
                    'strategic_boost': scores.get('strategic_boost'),
                    'personalization_match': scores.get('personalization_match'),
                    'popularity_score': scores.get('popularity_score'),
                    'reason': explanation.get('reason'),
                    'matched_interests': ", ".join(explanation.get('matched_user_interests', []))
                }
                writer.writerow(row)
                count += 1

    print(f"Success! Flattened {count} recommendations.")
    print(f"Output saved to: {output_file}")

if __name__ == "__main__":
    flatten_recommendations()
