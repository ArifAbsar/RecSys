"""
test_business_layer.py
──────────────────────
Smoke test for Step 2 — Business Meaning Layer.

Uses the same sample_columns.json from the Mapping Engine and runs
the full pipeline end-to-end:

  Mapping Engine (Step 1) → resolve_mappings()
                          → BusinessMeaningLayer.run()
                          → rec_config.json

Run from RecSys/ root:
  python Business_Meaning/test_business_layer.py

Or supply a pre-resolved JSON:
  python Business_Meaning/test_business_layer.py --mappings output/resolved.json
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Business_Meaning.business_layer import BusinessMeaningLayer

# ── Minimal synthetic resolved mappings to exercise all roles ──────────────
SYNTHETIC_MAPPINGS = [
    # identity
    {
        "source_table": "customers", "source_column": "customer_id",
        "status": "AUTO_ACCEPTED", "confidence": 0.99,
        "recommended_mapping": {
            "entity": "CUSTOMER", "field": "customer_id",
            "semantic_role": "identifier"
        }
    },
    {
        "source_table": "products", "source_column": "product_id",
        "status": "AUTO_ACCEPTED", "confidence": 0.99,
        "recommended_mapping": {
            "entity": "PRODUCT", "field": "product_id",
            "semantic_role": "identifier"
        }
    },
    # content_features
    {
        "source_table": "products", "source_column": "category",
        "status": "AUTO_ACCEPTED", "confidence": 0.93,
        "recommended_mapping": {
            "entity": "PRODUCT", "field": "category",
            "semantic_role": "attribute"
        }
    },
    {
        "source_table": "products", "source_column": "brand",
        "status": "AUTO_ACCEPTED", "confidence": 0.91,
        "recommended_mapping": {
            "entity": "PRODUCT", "field": "brand",
            "semantic_role": "attribute"
        }
    },
    # interaction_signals — named purchase event
    {
        "source_table": "user_events", "source_column": "event_type",
        "status": "AUTO_ACCEPTED", "confidence": 0.87,
        "recommended_mapping": {
            "entity": "SESSION_BEHAVIOR", "field": "event_type",
            "semantic_role": "categorical"
        }
    },
    # interaction_signals — monetary amount (the key example from the spec)
    {
        "source_table": "orders", "source_column": "total_amt",
        "status": "NEEDS_REVIEW", "confidence": 0.76,
        "recommended_mapping": {
            "entity": "TRANSACTION", "field": "total_amount",
            "semantic_role": "measure"
        }
    },
    # filters
    {
        "source_table": "inventory", "source_column": "is_available",
        "status": "AUTO_ACCEPTED", "confidence": 0.94,
        "recommended_mapping": {
            "entity": "INVENTORY", "field": "quantity_available",
            "semantic_role": "measure"
        }
    },
    {
        "source_table": "inventory", "source_column": "stock_quantity",
        "status": "AUTO_ACCEPTED", "confidence": 0.92,
        "recommended_mapping": {
            "entity": "INVENTORY", "field": "quantity_available",
            "semantic_role": "measure"
        }
    },
    # ignore — PII
    {
        "source_table": "tbl_user_v3", "source_column": "first_name",
        "status": "AUTO_ACCEPTED", "confidence": 0.88,
        "recommended_mapping": {
            "entity": "CUSTOMER", "field": "customer_name",
            "semantic_role": "pii"
        }
    },
    # ignore — timestamp
    {
        "source_table": "customers", "source_column": "created_at",
        "status": "AUTO_ACCEPTED", "confidence": 0.90,
        "recommended_mapping": {
            "entity": "CUSTOMER", "field": "created_at",
            "semantic_role": "timestamp"
        }
    },
    # ignore — custom field
    {
        "source_table": "internal_logs", "source_column": "temp_xyz",
        "status": "CUSTOM_FIELD", "confidence": 0.12,
        "recommended_mapping": {
            "entity": "CUSTOM", "field": "temp_xyz",
            "semantic_role": ""
        }
    },
]


def main() -> None:
    here       = os.path.dirname(os.path.abspath(__file__))
    root       = os.path.dirname(here)
    output_dir = os.path.join(root, "output")
    q_path     = os.path.join(root, "ecommerce", "questions.yaml")

    # Support --mappings flag for pre-resolved JSON
    if "--mappings" in sys.argv:
        idx = sys.argv.index("--mappings")
        mappings_path = sys.argv[idx + 1]
        print(f"Loading resolved mappings from: {mappings_path}")
        with open(mappings_path, "r", encoding="utf-8") as f:
            mappings = json.load(f)
    else:
        print("Using synthetic test mappings (18 columns across all roles)")
        mappings = SYNTHETIC_MAPPINGS

    layer = BusinessMeaningLayer(
        output_dir=output_dir,
        questions_path=q_path,
        auto_mode=True,   # non-interactive for smoke test
    )

    config = layer.run(mappings)

    # ── Assertions ────────────────────────────────────────────────────────
    print("\n[assertions]")

    assert "customer_id" in config["identity"],         "customer_id must be identity"
    assert "product_id"  in config["identity"],         "product_id must be identity"
    assert "category"    in config["content_features"], "category must be content_features"
    assert "brand"       in config["content_features"], "brand must be content_features"
    assert "total_amt"   in config["interaction_signals"], "total_amt must be interaction_signals"
    assert config["interaction_signals"]["total_amt"] == 0.8, \
        f"total_amt weight must be 0.8, got {config['interaction_signals']['total_amt']}"
    assert "first_name"  in config["ignore"],           "first_name must be ignore"
    assert "created_at"  in config["ignore"],           "created_at must be ignore"
    assert "temp_xyz"    in config["ignore"],           "temp_xyz must be ignore"

    # Filters — INVENTORY entity triggers filter role
    filter_cols = config["filters"]
    assert any("stock" in c or "available" in c for c in filter_cols), \
        f"Expected a filter column, got: {filter_cols}"

    print("  ✅  All assertions passed!")
    print(f"  📄  rec_config.json → {os.path.join(output_dir, 'rec_config.json')}")


if __name__ == "__main__":
    main()
