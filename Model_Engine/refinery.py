import os
import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Role detection helpers
# ---------------------------------------------------------------------------

_TIMESTAMP_KEYWORDS = {"date", "time", "timestamp", "ts", "created", "updated", "at"}
_ITEM_ENTITY_TYPES  = {"PRODUCT", "ITEM", "CONTENT", "GAME", "ARTICLE", "SKU"}
_USER_ENTITY_TYPES  = {"CUSTOMER", "USER", "ACCOUNT", "MEMBER", "PLAYER"}
_TRANSACTION_FIELD_KEYWORDS = {"transaction_id", "invoice", "order_id", "session_id", "basket_id"}


def _detect_roles(config: dict) -> dict:
    """
    Derive every column role purely from rec_config.json.

    Priority order for each role:
      user_col  → provenance entry whose entity is in USER_ENTITY_TYPES
      item_col  → provenance entry whose entity is in ITEM_ENTITY_TYPES,
                  else the identity field that is NOT the user
      timestamp → provenance field whose name or mapped-field contains a time keyword
      rating    → highest-weighted key in interaction_signals
      group_col → provenance field whose mapped-field target is a transaction/order keyword
                  (optional — only used for basket-style datasets)
    """
    provenance = config.get("mapping_provenance", {})
    identity   = config.get("identity", [])
    signals    = config.get("interaction_signals", {})

    # --- user ---
    user_col = None
    for src_col, meta in provenance.items():
        if meta.get("entity", "").upper() in _USER_ENTITY_TYPES:
            user_col = src_col
            break
    # fallback: scan identity list for anything that sounds like a user
    if user_col is None:
        for col in identity:
            col_l = col.lower()
            if any(kw in col_l for kw in ("user", "customer", "member", "player", "account")):
                user_col = col
                break
    if user_col is None and identity:
        # last resort: second identity field (index 1) if it exists, else first
        user_col = identity[1] if len(identity) > 1 else identity[0]

    # --- item ---
    item_col = None
    for src_col, meta in provenance.items():
        if meta.get("entity", "").upper() in _ITEM_ENTITY_TYPES:
            item_col = src_col
            break
    # fallback: the identity field that is NOT the user
    if item_col is None:
        for col in identity:
            if col != user_col:
                item_col = col
                break
    # last resort: first identity field if it's different from user
    if item_col is None and identity:
        item_col = identity[0] if identity[0] != user_col else (identity[1] if len(identity) > 1 else None)

    # --- rating ---
    rating_col = None
    if signals:
        rating_col = max(
            signals,
            key=lambda k: signals[k].get("weight", 0) if isinstance(signals[k], dict) else signals[k]
        )

    # --- timestamp ---
    timestamp_col = None
    for src_col, meta in provenance.items():
        field_target = meta.get("field", "").lower()
        src_l = src_col.lower()
        if any(kw in field_target or kw in src_l for kw in _TIMESTAMP_KEYWORDS):
            timestamp_col = src_col
            break

    # --- group / basket (optional, truly generic) ---
    group_col = None
    for src_col, meta in provenance.items():
        field_target = meta.get("field", "").lower()
        src_l = src_col.lower()
        if any(kw in field_target or kw in src_l for kw in _TRANSACTION_FIELD_KEYWORDS):
            group_col = src_col
            break

    return {
        "user_col":      user_col,
        "item_col":      item_col,
        "rating_col":    rating_col,
        "timestamp_col": timestamp_col,
        "group_col":     group_col,
    }


def _load_csv(data_path: str) -> pd.DataFrame:
    """Try UTF-8 first, fall back to ISO-8859-1 (handles both game and retail CSVs)."""
    for enc in ("utf-8", "ISO-8859-1"):
        try:
            return pd.read_csv(data_path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Cannot decode {data_path} with utf-8 or ISO-8859-1.")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_refinery(data_path: str, config_path: str, output_dir: str):
    """
    Step 4.1: The Profiler & Atomic Refinery — fully data-driven.
    All column roles are resolved from rec_config.json via mapping_provenance.
    No hardcoded column names, dataset-specific keywords, or encoding assumptions.
    """
    print(f"\n[REFINERY] Loading BML Config: {config_path}")
    with open(config_path, 'r') as f:
        config = json.load(f)

    roles = _detect_roles(config)
    user_col      = roles["user_col"]
    item_col      = roles["item_col"]
    rating_col    = roles["rating_col"]
    timestamp_col = roles["timestamp_col"]
    group_col     = roles["group_col"]

    print(f"[REFINERY] Roles Identified (from provenance):")
    print(f"  * User ID:      {user_col}")
    print(f"  * Item ID:      {item_col}")
    print(f"  * Rating Col:   {rating_col}")
    print(f"  * Timestamp:    {timestamp_col}")
    print(f"  * Group/Basket: {group_col}  (None = no basket structure)")

    if not user_col or not item_col:
        raise ValueError(
            f"Critical: could not identify User or Item column from rec_config.json.\n"
            f"  Detected user_col={user_col!r}, item_col={item_col!r}\n"
            f"  Check that mapping_provenance contains a CUSTOMER/USER entity entry."
        )

 
    print(f"\n[REFINERY] Loading Data: {data_path}")
    df = _load_csv(data_path)
    initial_len = len(df)

    # Drop rows missing the core identifiers
    df = df.dropna(subset=[user_col, item_col])

    if rating_col and rating_col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[rating_col]):
            numeric_cast = pd.to_numeric(df[rating_col], errors='coerce')
            if numeric_cast.notna().sum() > 0.8 * len(df):
                df[rating_col] = numeric_cast
            
            rules_list = config.get("interaction_signal_rules", [])

            codes, uniques = pd.factorize(df[rating_col])
            mapping = {val: float(idx + 1) for idx, val in enumerate(uniques)}
            
            for val in uniques:
                val_lower = str(val).lower()
                matched = False
                
                for r in rules_list:
                    weight = float(r.get("weight", 0.5))
                    keywords = [str(k).lower() for k in r.get("keywords", [])]
                    if any(kw in val_lower for kw in keywords):
                        mapping[val] = weight
                        matched = True
                        break
                
                # 2. Dynamic baseline fallback for safety if no match in BML rules
                if not matched:
                    heuristics_path = os.path.join(os.path.dirname(__file__), "default_heuristics.yaml")
                    heuristics = {}
                    if os.path.exists(heuristics_path):
                        try:
                            import yaml
                            with open(heuristics_path, "r", encoding="utf-8") as hf:
                                heuristics = yaml.safe_load(hf).get("heuristics", {})
                        except Exception as e:
                            print(f"[REFINERY] Warning: Failed to load {heuristics_path}: {e}")
                    
                    for term, wt in heuristics.items():
                        if term in val_lower:
                            mapping[val] = wt
                            matched = True
                            break
            
            df[rating_col] = df[rating_col].map(mapping).astype(float)
            print(f"  * Dynamically mapped categories: {mapping}")

    # Remove zero/negative interactions if rating is numeric
    if rating_col and rating_col in df.columns and pd.api.types.is_numeric_dtype(df[rating_col]):
        df = df[df[rating_col] > 0]

    # For basket-style datasets: drop cancellation rows (e.g. invoice starting with 'C')
    # Only applied when a group column was actually found — never hardcoded
    if group_col and group_col in df.columns:
        before = len(df)
        df = df[~df[group_col].astype(str).str.startswith('C', na=False)]
        dropped = before - len(df)
        if dropped:
            print(f"[REFINERY] Dropped {dropped} cancellation rows (group_col={group_col!r})")

    print(f"[REFINERY] Cleaned Data: {len(df)} rows  (dropped {initial_len - len(df)} total)")

    # ------------------------------------------------------------------
    # Profiler — model selection
    # ------------------------------------------------------------------
    print("\n[PROFILER] Analyzing Data Patterns...")

    items_per_group = 1.0
    if group_col and group_col in df.columns:
        items_per_group = df.groupby(group_col).size().mean()
        print(f"  * Avg Items per Basket/Group: {items_per_group:.2f}")

    events_per_user = 1.0
    if timestamp_col and timestamp_col in df.columns:
        try:
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
            events_per_user = df.groupby(user_col)[timestamp_col].nunique().mean()
            print(f"  * Avg Unique Events per User: {events_per_user:.2f}")
        except Exception:
            print(f"  * [PROFILER] Could not parse timestamp column '{timestamp_col}', skipping.")

    if items_per_group > 3.0:
        recommendation  = "LightGCN (Basket-based)"
        suggested_model = "LightGCN"
    elif events_per_user > 2.0:
        recommendation  = "SASRec (Sequence-based)"
        suggested_model = "SASRec"
    else:
        recommendation  = "BPR (Simple Collaborative)"
        suggested_model = "BPR"

    print(f"\n[DECISION] Profiler recommends: {recommendation}")

    # ------------------------------------------------------------------
    # Export RecBole atomic files
    # ------------------------------------------------------------------
    print(f"\n[REFINERY] Exporting atomic files to: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    # rec.inter
    inter_cols = {
        "user_id:token": df[user_col].astype(str),
        "item_id:token": df[item_col].astype(str),
    }
    if rating_col and rating_col in df.columns:
        inter_cols["rating:float"] = df[rating_col].astype(float)
    if timestamp_col and timestamp_col in df.columns and pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
        inter_cols["timestamp:float"] = df[timestamp_col].astype("int64") // 10**9

    pd.DataFrame(inter_cols).to_csv(
        os.path.join(output_dir, "rec.inter"), index=False, sep="\t"
    )
    print(f"  [OK] rec.inter  ({len(inter_cols) - 2} extra cols: {list(inter_cols.keys())[2:]})")

    # rec.user  — any interaction_signals column that isn't the rating/group/time
    user_feats = {}
    reserved = {rating_col, group_col, timestamp_col, user_col, item_col}
    for col in config.get("interaction_signals", {}).keys():
        if col not in reserved and col in df.columns:
            suffix = "float" if pd.api.types.is_numeric_dtype(df[col]) else "token"
            user_feats[f"{col}:{suffix}"] = df.groupby(user_col)[col].first().values

    if user_feats:
        user_feat_df = pd.DataFrame({"user_id:token": list(df.groupby(user_col).groups.keys())})
        for feat_name, feat_vals in user_feats.items():
            user_feat_df[feat_name] = feat_vals
        user_feat_df.to_csv(os.path.join(output_dir, "rec.user"), index=False, sep="\t")
        print(f"  [OK] rec.user   ({list(user_feats.keys())})")
    else:
        print(f"  [-] rec.user   (no extra user features)")

    # rec.item  — content_features from config
    content_features = config.get("content_features", [])
    available_content = [f for f in content_features if f in df.columns]
    missing_content   = [f for f in content_features if f not in df.columns]

    if missing_content:
        print(f"  [WARN] content_features not found in CSV, skipping: {missing_content}")

    if available_content:
        item_feat_df = pd.DataFrame({"item_id:token": df.groupby(item_col).groups.keys()})
        for feat in available_content:
            suffix = "float" if pd.api.types.is_numeric_dtype(df[feat]) else "token"
            item_feat_df[f"{feat}:{suffix}"] = df.groupby(item_col)[feat].first().values
        item_feat_df.to_csv(os.path.join(output_dir, "rec.item"), index=False, sep="\t")
        print(f"  [OK] rec.item   ({list(item_feat_df.columns[1:])})")
    else:
        print(f"  [-] rec.item   (no content features available)")

    print(f"\n[REFINERY] Done.")

    with open(os.path.join(os.path.dirname(output_dir), "model_choice.json"), "w") as f:
        json.dump({"suggested_model": suggested_model}, f)

    # Rebuild YAML configurations dynamically from the new rec_config.json
    try:
        from Model_Engine.config_builder import build_configs
        build_configs(os.path.abspath(os.path.join(os.path.dirname(output_dir), "..")))
    except ImportError:
        try:
            from config_builder import build_configs
            build_configs(os.path.abspath(os.path.join(os.path.dirname(output_dir), "..")))
        except Exception as e:
            print(f"[REFINERY] Warning: could not run config_builder automatically: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Refinery: convert any CSV into RecBole atomic files")
    parser.add_argument("--data",   default=None, help="Path to source CSV (default: <project_root>/data.csv)")
    parser.add_argument("--config", default=None, help="Path to rec_config.json (default: <project_root>/output/rec_config.json)")
    parser.add_argument("--out",    default=None, help="Output directory (default: <project_root>/Model_Engine/rec)")
    args = parser.parse_args()

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    run_refinery(
        data_path  = args.data   or os.path.join(base_dir, "data.csv"),
        config_path= args.config or os.path.join(base_dir, "output", "rec_config.json"),
        output_dir = args.out    or os.path.join(base_dir, "Model_Engine", "rec"),
    )

