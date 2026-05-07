# Autonomous Mapping Engine

A data-driven, AI-powered schema mapping engine designed to automatically resolve source data columns to a canonical target schema. It uses a hybrid approach of semantic embeddings, cross-encoders, and business-logic boosts to achieve high-confidence mappings.

##  Key Features

- **Hybrid AI Scoring**: Combines Bi-Encoders (fast retrieval) and Cross-Encoders (deep contextual ranking).
- **Data-Driven Logic**: No hardcoded field names. All business logic is injected via YAML manifests (`rules.yaml`, `keywords.yaml`, etc.).
- **Automatic Type Inference**: Supports both JSON column definitions and direct CSV ingestion with automatic type detection.
- **Global 1:1 Resolution**: Enforces integrity by ensuring each source column maps to exactly one target field across the entire dataset.

##  Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare your Schema**:
   Ensure your target schema and business rules are defined in the `ecommerce/` directory (or your custom path).

##  Usage

### Running with a CSV File
To automatically map columns from a raw data file:
```bash
python Mapping_Engine/test.py --columns data.csv
```

### Running with JSON Column Definitions
To map a pre-defined set of source columns:
```bash
python Mapping_Engine/test.py --columns Mapping_Engine/sample_columns.json
```

### Configuration Options
- `--schema`: Path to the target schema YAML.
- `--threshold`: Confidence threshold for auto-acceptance (default: 0.60).
- `--model`: Sentence-transformer model for semantic matching.

##  Architecture

The engine follows a 5-stage pipeline:
1. **Registry Building**: Consolidates schema, keywords, and rules into a searchable index.
2. **Retrieval**: Bi-Encoder finds the top-K semantic candidates.
3. **Reranking**: Cross-Encoder performs deep comparison of source vs. target descriptions and business rules.
4. **Logic Boosting**: Applies scores for synonym matches and keyword intents defined in manifests.
5. **Global Resolution**: Greedily assigns mappings to enforce 1:1 constraints.

##  Directory Structure

```text
Mapping_Engine/
├── Domain_loader/      # Schema and manifest ingestion
├── embedding_matcher/  # Vector retrieval logic
├── Reranking/          # Cross-encoder and logic boost implementation
└── test.py            # Main entry point and CLI
ecommerce/             # Target schema and business rules (YAML)
```


