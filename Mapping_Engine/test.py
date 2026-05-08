"""
test.py
───────
Master test script for the Mapping Engine + Business Meaning Layer.
Merged version: User structure + Data-driven classification logic + JSON Numpy Fix.
"""

import argparse
import json
import os
import sys
import yaml
from tqdm import tqdm
import pandas as pd
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, _ROOT)

from Domain_loader.domain_loader import build_target_registry, build_keyword_intent_index
from embedding_matcher.embedding import Embedding
from Reranking.Reranker import rerank_candidates
from Reranking.Resolver import resolve_mappings
from sentence_transformers import CrossEncoder
from Business_Meaning.business_layer import BusinessMeaningLayer

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def parse_args():
    parser = argparse.ArgumentParser(description="Run the Mapping Engine")
    
    # Domain configuration
    parser.add_argument("--domain-dir", default="../ecommerce", help="Directory containing domain manifests")
    
    # Individual file overrides (defaults are relative to --domain-dir)
    parser.add_argument("--schema",      help="Path to schema.yaml")
    parser.add_argument("--keywords",    help="Path to keywords.yaml")
    parser.add_argument("--metrics",     help="Path to metrics.yaml")
    parser.add_argument("--rules",       help="Path to rules.yaml")
    parser.add_argument("--validations", help="Path to validations.yaml")
    parser.add_argument("--questions",   help="Path to questions.yaml")
    parser.add_argument("--rules-manifest", help="Path to classifier_rules.yaml (BML)")
    
    parser.add_argument("--columns", required=True, help="Path to source CSV")
    parser.add_argument("--model", default="all-MiniLM-L6-v2")
    parser.add_argument("--threshold", type=float, default=0.60)
    parser.add_argument("--output", default="output")
    parser.add_argument("--skip-business-layer", action="store_true")
    parser.add_argument("--auto", action="store_true")

    args = parser.parse_args()

    d = args.domain_dir
    if not args.schema:      args.schema      = os.path.join(d, "schema.yaml")
    if not args.keywords:    args.keywords    = os.path.join(d, "keywords.yaml")
    if not args.metrics:     args.metrics     = os.path.join(d, "metrics.yaml")
    if not args.rules:       args.rules       = os.path.join(d, "rules.yaml")
    if not args.validations: args.validations = os.path.join(d, "validations.yaml")
    if not args.questions:   args.questions   = os.path.join(d, "questions.yaml")
    if not args.rules_manifest: args.rules_manifest = os.path.join(d, "classifier_rules.yaml")

    return args

def load_columns_from_csv(path: str) -> list:
    print(f"Loading columns from CSV: {path}")
    df = pd.read_csv(path, nrows=100)
    table_name = os.path.basename(path).split(".")[0]
    columns = []
    for col in df.columns:
        columns.append({"table": table_name, "column": col, "data_type": "string"})
    return columns

def main(args):
    output_dir = args.output if os.path.isabs(args.output) else os.path.join(_ROOT, args.output)
    os.makedirs(output_dir, exist_ok=True)
    
    source_columns = load_columns_from_csv(args.columns)
    
    print("Initializing Domain Registry...")
    targets = build_target_registry(args.schema, args.keywords, args.metrics, args.rules, args.validations, args.questions)
    
    print("Initializing AI models...")
    matcher = Embedding(args.model)
    cross_encoder = CrossEncoder('cross-encoder/stsb-distilroberta-base')
    matcher.fit_targets(targets)
    keyword_intent = build_keyword_intent_index(args.keywords)

    all_source_results = []
    pbar = tqdm(source_columns, desc="Mapping Columns")
    for col in pbar:
        pbar.set_postfix(column=col["column"])
        raw_cands = matcher.match_column(col)
        refined = rerank_candidates(col, raw_cands, cross_encoder, keyword_intent=keyword_intent)
        all_source_results.append({"source": col, "candidates": refined})

    pbar.close()

    final_output = resolve_mappings(all_source_results, args.threshold)

    path = os.path.join(output_dir, "resolved_mappings.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2, cls=NpEncoder)
    print(f"[pipeline] Step 1 complete → {path}")

    if not args.skip_business_layer:
        print("\n[pipeline] Running Step 2: Business Meaning Layer ...")
        layer = BusinessMeaningLayer(
            output_dir=output_dir,
            questions_path=os.path.abspath(args.questions),
            auto_mode=args.auto,
            rules_path=os.path.abspath(args.rules_manifest)
        )
        layer.run(final_output)
        print(f"[pipeline] Step 2 complete.")

if __name__ == "__main__":
    main(parse_args())