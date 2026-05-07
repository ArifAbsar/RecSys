import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Domain_loader.domain_loader import build_target_registry, build_keyword_intent_index
from embedding_matcher.embedding import Embedding
from Reranking.Reranker import rerank_candidates
from Reranking.Resolver import resolve_mappings
from sentence_transformers import CrossEncoder
import yaml
from tqdm import tqdm
import pandas as pd



def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the Mapping Engine: embed targets and match source columns."
    )
    parser.add_argument(
        "--schema",
        default="../ecommerce/schema.yaml",
        help="Path to the schema YAML file (default: ../ecommerce/schema.yaml)",
    )
    parser.add_argument(
        "--model",
        default="all-MiniLM-L6-v2",
        help="Sentence-transformer model name (default: all-MiniLM-L6-v2)",
    )
    parser.add_argument(
        "--columns",
        required=True,
        help="Path to a JSON file containing a list of source column dicts.",
    )
    parser.add_argument(
        "--keywords",
        default="../ecommerce/keywords.yaml",
        help="Path to the keywords YAML file (default: ../ecommerce/keywords.yaml)",
    )
    parser.add_argument(
        "--metrics",
        default="../ecommerce/metrics.yaml",
        help="Path to the metrics YAML file (default: ../ecommerce/metrics.yaml)",
    )
    parser.add_argument(
        "--rules",
        default="../ecommerce/rules.yaml",
        help="Path to the rules YAML file (default: ../ecommerce/rules.yaml)",
    )
    parser.add_argument(
        "--validations",
        default="../ecommerce/validations.yaml",
        help="Path to the validations YAML file (default: ../ecommerce/validations.yaml)",
    )
    parser.add_argument(
        "--questions",
        default="../ecommerce/questions.yaml",
        help="Path to the questions YAML file (default: ../ecommerce/questions.yaml)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.60,
        help="Confidence threshold below which a column is marked custom_field (default: 0.70)",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        help="Minimum final_confidence to include in top_candidates output (default: 0.0 = all)",
    )
    return parser.parse_args()




def load_columns_from_csv(path: str) -> list:
    print(f"Loading columns from CSV: {path}")
    df = pd.read_csv(path, nrows=100)
    table_name = os.path.basename(path).split(".")[0]
    
    columns = []
    for col in df.columns:
        sample = df[col].dropna()
        if sample.empty:
            dtype = "string"
        else:
            if pd.api.types.is_numeric_dtype(df[col]):
                dtype = "number"
            elif pd.api.types.is_datetime64_any_dtype(df[col]) or "date" in col.lower() or "time" in col.lower():
                dtype = "datetime"
            else:
                dtype = "string"
        
        columns.append({
            "table": table_name,
            "column": col,
            "data_type": dtype
        })
    return columns


def load_columns(path: str) -> list:
    if path.endswith(".csv"):
        return load_columns_from_csv(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array in {path}, got {type(data).__name__}")
    return data


def map_column(
    source_column: dict,
    matcher: Embedding,
    threshold: float,
    min_score: float,
    cross_encoder: CrossEncoder,
) -> dict:
    raw_candidates = matcher.match_column(source_column)
    ranked_candidates = rerank_candidates(source_column, raw_candidates, cross_encoder=cross_encoder)
    best = ranked_candidates[0]

    if best["final_confidence"] >= 0.90:
        mapped_status = "auto_accepted"
        recommended_mapping = best
    elif best["final_confidence"] >= threshold:
        mapped_status = "needs_review"
        recommended_mapping = best
    else:
        mapped_status = "custom_field"
        recommended_mapping = {
            "field_id": "ecommerce.PRODUCT.custom_attribute",
            "entity": "PRODUCT",
            "field": "custom_attribute",
            "attribute_name": source_column["column"],
            "confidence_percent": best["confidence_percent"],
            "risk": "medium",
            "reason": "No strong standard schema match found. Keep as a custom product attribute.",
        }

    qualified_candidates = [
        c for c in ranked_candidates
        if c["final_confidence"] >= min_score
    ] or ranked_candidates[:1]

    return {
        "source_table": source_column["table"],
        "source_column": source_column["column"],
        "status": mapped_status,
        "recommended_mapping": recommended_mapping,
        "qualified_candidates": qualified_candidates,
        "user_action_required": mapped_status != "auto_accepted",
    }





def main(schema_path, model_name, columns_path, keywords_path, metrics_path, rules_path, validations_path, questions_path, threshold, min_score):
    print(f"[config] bi-encoder    : {model_name}")
    print(f"[config] cross-encoder : stsb-distilroberta-base")
    print(f"[config] schema        : {schema_path}")
    print(f"[config] columns       : {columns_path}")
    print(f"[config] keywords      : {keywords_path}")
    print(f"[config] metrics       : {metrics_path}")
    print(f"[config] rules         : {rules_path}")
    print(f"[config] validations   : {validations_path}")
    print(f"[config] questions     : {questions_path}")
    print(f"[config] threshold     : {threshold}")
    print(f"[config] min_score     : {min_score}\n")
    
    source_columns = load_columns(columns_path)
    print(f"Loaded {len(source_columns)} source column(s).")
    
    print("Initializing Domain Registry (Schema + Keywords + Metrics + Rules + Validations)...")
    targets = build_target_registry(
        schema_path, 
        keywords_path=keywords_path, 
        metrics_path=metrics_path,
        rules_path=rules_path,
        validations_path=validations_path,
        questions_path=questions_path
    )
    
    print("Initializing AI models...")
    matcher = Embedding(model_name)
    cross_encoder = CrossEncoder('cross-encoder/stsb-distilroberta-base')
    matcher.fit_targets(targets)

    keyword_intent = build_keyword_intent_index(keywords_path)

    all_source_results = []
    pbar = tqdm(source_columns, desc="Mapping Columns")
    for col in pbar:
        pbar.set_postfix(column=col.get("column", "unknown"))
        raw_cands = matcher.match_column(col)
        refined = rerank_candidates(col, raw_cands, cross_encoder, keyword_intent=keyword_intent)
        all_source_results.append({
            "source":     col,
            "candidates": refined
        })

    final_output = resolve_mappings(all_source_results, threshold)

    print("\n" + "=" * 110)
    print(f"{'SOURCE COLUMN':<35} | {'MAPPED TARGET':<45} | {'CONF':<7} | {'STATUS'}")
    print("-" * 110)
    
    for entry in final_output:
        src = f"{entry['source_table']}.{entry['source_column']}"
        mapping = entry['recommended_mapping']
        
        target = f"{mapping.get('entity')}.{mapping.get('field')}"
        conf = f"{int(entry['confidence'] * 100)}%"
        status = entry['status'].upper()
        
        print(f"{src:<35} | {target:<45} | {conf:<7} | {status}")
        
        cands = entry.get('top_candidates', [])
        for i, cand in enumerate(cands[:5]):
            prefix = " └─ " if i == len(cands[:5]) - 1 else " ├─ "
            cand_name = f"{cand.get('entity')}.{cand.get('field')}"
            cand_conf = f"{round(cand.get('final_confidence', 0) * 100)}%"
            print(f"  {prefix} {cand_name:<40} ({cand_conf})")
        print("-" * 110)
    
    print("=" * 110 + "\n")


if __name__ == "__main__":
    args = parse_args()
    main(
        schema_path=args.schema,
        model_name=args.model,
        columns_path=args.columns,
        keywords_path=args.keywords,
        metrics_path=args.metrics,
        rules_path=args.rules,
        validations_path=args.validations,
        questions_path=args.questions,
        threshold=args.threshold,
        min_score=args.min_score,
    )