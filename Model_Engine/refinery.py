import os
import json
import pandas as pd
import numpy as np
from pathlib import Path

def run_refinery(data_path: str, config_path: str, output_dir: str):
    """
    Step 4.1: The Profiler & Atomic Refinery (Autonomous Version)
    No hardcoded column names. Everything derived from BML Config.
    """
    print(f"\n[REFINERY] Loading BML Config: {config_path}")
    with open(config_path, 'r') as f:
        config = json.load(f)

    provenance = config.get("mapping_provenance", {})

    user_col = config.get("identity", [])[0] if config.get("identity") else None
    
    item_col = config.get("filters", [])[0] if config.get("filters") else None
    
    signals = config.get("interaction_signals", {})
    if signals:
        rating_col = max(signals, key=lambda k: signals[k].get("weight", 0) if isinstance(signals[k], dict) else signals[k])
    else:
        rating_col = None

    group_col = None
    for src_col, meta in provenance.items():
        field_target = meta.get("field", "").lower()
        if "transaction_id" in field_target or "invoiceno" in src_col.lower():
            group_col = src_col
            break

    timestamp_col = None
    for src_col, meta in provenance.items():
        field_target = meta.get("field", "").lower()
        if "date" in field_target or "time" in field_target or "date" in src_col.lower() or "time" in src_col.lower():
            timestamp_col = src_col
            break

    print(f"[REFINERY] Roles Identified (Autonomous):")
    print(f"  • User ID:      {user_col}")
    print(f"  • Item ID:      {item_col}")
    print(f"  • Rating Col:   {rating_col}")
    print(f"  • Grouping Col: {group_col}")
    print(f"  • Time Col:     {timestamp_col}")

    if not user_col or not item_col:
        raise ValueError("Critical mapping failure: User ID or Item ID not identified in config.")

    print(f"[REFINERY] Loading Data: {data_path}")
    df = pd.read_csv(data_path, encoding='ISO-8859-1')
    initial_len = len(df)
    if rating_col and pd.api.types.is_numeric_dtype(df[rating_col]):
        df = df[df[rating_col] > 0]
    df = df.dropna(subset=[user_col, item_col])
    
    if group_col:
        df = df[~df[group_col].astype(str).str.startswith('C', na=False)]

    print(f"[REFINERY] Cleaned Data: {len(df)} rows (Dropped {initial_len - len(df)})")
    print("\n[PROFILER] Analyzing Data Patterns...")
    
    if group_col:
        items_per_group = df.groupby(group_col).size().mean()
        print(f"  • Average Items per Group: {items_per_group:.2f}")
    else:
        items_per_group = 1.0

    if timestamp_col:
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        events_per_user = df.groupby(user_col)[timestamp_col].nunique().mean()
        print(f"  • Average Events per User: {events_per_user:.2f}")
    else:
        events_per_user = 1.0

    if items_per_group > 3.0:
        recommendation = "LightGCN (Basket-based)"
        suggested_model = "LightGCN"
    elif events_per_user > 2.0:
        recommendation = "SASRec (Sequence-based)"
        suggested_model = "SASRec"
    else:
        recommendation = "BPR (Simple Collaborative)"
        suggested_model = "BPR"

    print(f"\n[DECISION] Profiler recommends: {recommendation}")

    print(f"\n[REFINERY] Exporting Atomic Files to: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    inter_cols = {
        'user_id:token': df[user_col].astype(str),
        'item_id:token': df[item_col].astype(str),
    }
    if rating_col:
        inter_cols['rating:float'] = df[rating_col].astype(float)
    if timestamp_col:
        inter_cols['timestamp:float'] = df[timestamp_col].astype('int64') // 10**9
    
    pd.DataFrame(inter_cols).to_csv(os.path.join(output_dir, 'rec.inter'), index=False, sep='\t')

    user_feats = {}
    for col in config.get("interaction_signals", {}).keys():
        if col != rating_col and col != group_col:
            user_feats[f"{col}:token"] = df.groupby(user_col)[col].first().values
    
    if user_feats:
        user_feat_df = pd.DataFrame({'user_id:token': df.groupby(user_col).groups.keys()})
        for feat_name, feat_vals in user_feats.items():
            user_feat_df[feat_name] = feat_vals
        user_feat_df.to_csv(os.path.join(output_dir, 'rec.user'), index=False, sep='\t')

    content_features = config.get("content_features", [])
    if content_features:
        item_feat_df = pd.DataFrame({'item_id:token': df.groupby(item_col).groups.keys()})
        for feat in content_features:
            item_feat_df[f"{feat}:token"] = df.groupby(item_col)[feat].first().values
        item_feat_df.to_csv(os.path.join(output_dir, 'rec.item'), index=False, sep='\t')

    print(f"[REFINERY] Success. Generated rec.inter and metadata files.")
    
    with open(os.path.join(os.path.dirname(output_dir), 'model_choice.json'), 'w') as f:
        json.dump({"suggested_model": suggested_model}, f)

if __name__ == "__main__":
    base_dir = r"e:\docker-crash-course\RecSys"
    run_refinery(
        data_path=os.path.join(base_dir, "data.csv"),
        config_path=os.path.join(base_dir, "output", "rec_config.json"),
        output_dir=os.path.join(base_dir, "Model_Engine", "rec")
    )
