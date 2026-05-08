"""
run.py
──────
CLI entry point for the Business Meaning Layer (Step 2).
Fully data-driven and domain-agnostic.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Allow running from RecSys/ root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Business_Meaning.business_layer import BusinessMeaningLayer

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Business Meaning Layer — classify mapped columns into rec_config.json"
    )

    # Mode A: pre-resolved mappings JSON
    parser.add_argument(
        "--mappings",
        help="Path to resolved_mappings.json produced by Mapping Engine (Mode A)",
    )

    # Mode B: run full pipeline
    parser.add_argument("--domain-dir", default="../ecommerce", help="Directory containing domain manifests")
    parser.add_argument("--schema",      help="Path to schema.yaml")
    parser.add_argument("--columns",     help="Source columns JSON or CSV")
    parser.add_argument("--keywords",    help="Path to keywords.yaml")
    parser.add_argument("--metrics",     help="Path to metrics.yaml")
    parser.add_argument("--rules",       help="Path to rules.yaml")
    parser.add_argument("--validations", help="Path to validations.yaml")
    parser.add_argument("--questions",   help="Path to questions.yaml")
    parser.add_argument("--rules-manifest", help="Path to classifier_rules.yaml (BML)")
    
    parser.add_argument("--threshold",   type=float, default=0.60)
    parser.add_argument("--model",       default="all-MiniLM-L6-v2")

    parser.add_argument(
        "--output",
        default="output",
        help="Directory to write rec_config.json (default: ./output/)",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Non-interactive mode — ambiguous columns get default role",
    )

    args = parser.parse_args()

    # Dynamic resolution for domain files
    d = args.domain_dir
    if not args.schema:      args.schema      = os.path.join(d, "schema.yaml")
    if not args.keywords:    args.keywords    = os.path.join(d, "keywords.yaml")
    if not args.metrics:     args.metrics     = os.path.join(d, "metrics.yaml")
    if not args.rules:       args.rules       = os.path.join(d, "rules.yaml")
    if not args.validations: args.validations = os.path.join(d, "validations.yaml")
    if not args.questions:   args.questions   = os.path.join(d, "questions.yaml")
    if not args.rules_manifest: args.rules_manifest = os.path.join(d, "classifier_rules.yaml")

    return args


def load_mappings_from_file(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    raise ValueError(f"Expected a JSON array in {path}, got {type(data).__name__}")


def run_mapping_engine(args: argparse.Namespace) -> list[dict]:
    """Run Steps 1 (Mapping Engine) and return resolved mappings."""
    from Mapping_Engine.Domain_loader.domain_loader import (
        build_target_registry, build_keyword_intent_index
    )
    from Mapping_Engine.embedding_matcher.embedding import Embedding
    from Mapping_Engine.Reranking.Reranker import rerank_candidates
    from Mapping_Engine.Reranking.Resolver import resolve_mappings
    from sentence_transformers import CrossEncoder
    import pandas as pd
    from tqdm import tqdm

    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)

    def _resolve(p: str) -> str:
        if os.path.isabs(p):
            return p
        return os.path.normpath(os.path.join(root, p))

    schema_path      = _resolve(args.schema)
    keywords_path    = _resolve(args.keywords)
    metrics_path     = _resolve(args.metrics)
    rules_path       = _resolve(args.rules)
    validations_path = _resolve(args.validations)
    questions_path   = _resolve(args.questions)
    columns_path     = _resolve(args.columns)

    if columns_path.endswith(".csv"):
        df = pd.read_csv(columns_path, nrows=100)
        table_name = os.path.basename(columns_path).split(".")[0]
        source_columns = []
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                dtype = "number"
            elif "date" in col.lower() or "time" in col.lower():
                dtype = "datetime"
            else:
                dtype = "string"
            source_columns.append({"table": table_name, "column": col, "data_type": dtype})
    else:
        with open(columns_path, "r", encoding="utf-8") as f:
            source_columns = json.load(f)

    print(f"[pipeline] Loaded {len(source_columns)} source column(s).")

    targets = build_target_registry(
        schema_path,
        keywords_path=keywords_path,
        metrics_path=metrics_path,
        rules_path=rules_path,
        validations_path=validations_path,
        questions_path=questions_path,
    )
    matcher       = Embedding(args.model)
    cross_encoder = CrossEncoder("cross-encoder/stsb-distilroberta-base")
    matcher.fit_targets(targets)
    keyword_intent = build_keyword_intent_index(keywords_path)

    all_source_results = []
    for col in tqdm(source_columns, desc="Mapping Columns"):
        raw_cands = matcher.match_column(col)
        refined   = rerank_candidates(col, raw_cands, cross_encoder, keyword_intent=keyword_intent)
        all_source_results.append({"source": col, "candidates": refined})

    return resolve_mappings(all_source_results, args.threshold)


def main() -> None:
    args = parse_args()

    here   = os.path.dirname(os.path.abspath(__file__))
    root   = os.path.dirname(here)
    output_dir = args.output if os.path.isabs(args.output) \
                 else os.path.normpath(os.path.join(root, args.output))

    def _resolve(p: str) -> str:
        if os.path.isabs(p): return p
        return os.path.normpath(os.path.join(root, p))

    questions_path = _resolve(args.questions)
    rules_manifest = _resolve(args.rules_manifest)

    if args.mappings:
        print(f"[run] Loading pre-resolved mappings from: {args.mappings}")
        resolved_mappings = load_mappings_from_file(args.mappings)
    elif args.columns:
        print("[run] Running full pipeline (Mapping Engine → Business Layer)")
        resolved_mappings = run_mapping_engine(args)
    else:
        print("Provide --mappings <file> or --columns <file>", file=sys.stderr)
        sys.exit(1)

    print(f"[run] {len(resolved_mappings)} resolved mapping(s) received.")
    layer = BusinessMeaningLayer(
        output_dir=output_dir,
        questions_path=questions_path,
        rules_path=rules_manifest,
        auto_mode=args.auto,
    )
    config = layer.run(resolved_mappings)

    print(f"[run] rec_config.json written to: {output_dir}")
    return config


if __name__ == "__main__":
    main()
