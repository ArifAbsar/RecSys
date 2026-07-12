# RecSys

RecSys is a Python recommendation platform with two main capabilities:

1. **Schema intelligence**: automatically map raw source columns to canonical ecommerce fields.
2. **Recommendation serving**: train/score recommendation models and generate final user-level recommendations.

It combines semantic NLP matching, rules-based business classification, and RecBole-based recommendation pipelines.

## High-level architecture

![System Architecture Flow Map](img/Recsys_clean.png)

The repository is organized into four major layers:

- **Mapping Engine (`Mapping_Engine/`)**: Step 1 schema-to-schema mapping.
- **Business Meaning Layer (`Business_Meaning/`)**: Step 2 role classification + config generation.
- **Model Engine (`Model_Engine/`)**: model training and inference/re-ranking.
- **API (`api/`)**: Flask service to run schema mapping jobs and deploy config outputs.

## Key technologies

- **Language/runtime**: Python
- **API framework**: Flask
- **ML/NLP**:
  - `sentence-transformers` (bi-encoder and semantic classification)
  - `CrossEncoder` reranking (`cross-encoder/stsb-distilroberta-base`)
  - `scikit-learn` (e.g., clustering)
- **Recommendation framework**: RecBole + PyTorch
- **Data/config**:
  - YAML-driven domain manifests (`ecommerce/*.yaml`)
  - Pandas/Numpy for ingestion and transformation
- **Optional cloud path**: AWS integrations in `api/routes/jobs_aws.py` (S3 + Glue)

## Repository structure

```text
RecSys/
├── api/                         # Flask app (health, jobs, orchestration pipeline)
│   ├── routes/                 # Environment-specific job routes (local/aws) + health
│   ├── pipeline.py             # Headless mapping pipeline used by API jobs
│   └── state.py                # Global model state and warmup
├── Business_Meaning/           # Step 2: role classification + rec_config generation
│   ├── business_layer.py       # Main orchestrator
│   ├── role_classifier.py      # Rule/semantic role assignment
│   ├── ambiguity_detector.py   # Human-review trigger logic
│   ├── question_flow.py        # Human-in-the-loop question resolution
│   ├── config_writer.py        # Writes output/rec_config.json
│   └── run.py                  # CLI entrypoint for Step 2 or full Step1→Step2 flow
├── Mapping_Engine/             # Step 1: semantic mapping to canonical schema
│   ├── Domain_loader/          # YAML parsing + target registry build
│   ├── embedding_matcher/      # Bi-encoder retrieval
│   ├── Reranking/              # Cross-encoder + rule-based reranking + resolver
│   ├── text_builder/           # Text construction for embeddings
│   └── test.py                 # CLI pipeline runner for mapping (+ optional Step 2)
├── Model_Engine/               # Training + inference recommendation pipeline
│   ├── pipeline/               # Inference, scoring, selection, reranking, profiling
│   ├── run_stage1.py           # RecBole model training entrypoint
│   ├── run_stage2.py           # User-tunable inference control panel entrypoint
│   ├── config_builder.py       # Builds model configs from business config
│   └── refinery.py             # Data preparation
├── ecommerce/                  # Domain manifests (schema, rules, metrics, validations)
├── output/                     # Generated artifacts (resolved mappings, rec config, recs)
├── data.csv                    # Sample input dataset
├── api_service.py              # Flask service entrypoint
└── requirements.txt            # Python dependencies
```

## How code is organized (execution paths)

### 1) Schema mapping + business classification

- **Mapping only**:

```bash
python Mapping_Engine/test.py --columns data.csv
```

- **Business layer full flow (runs Step 1 then Step 2)**:

```bash
python Business_Meaning/run.py --columns data.csv --domain-dir ecommerce
```

Primary outputs:
- `output/resolved_mappings.json`
- `output/rec_config.json`

### 2) API service for mapping jobs

```bash
python api_service.py
```

Core endpoints are under `/api/*` (health, schema details, job create/start/poll/update/deploy).

### 3) Recommendation model training + inference

- **Train retrieval/ranking models**:

```bash
python Model_Engine/run_stage1.py
```

- **Run stage-2 inference and final recommendation generation**:

```bash
python Model_Engine/run_stage2.py
```

Typical outputs include `output/final_recommendations.json` and `output/final_recommendations.csv`.

## Domain-driven configuration

All schema/business logic is externalized in `/ecommerce` manifests:

- `schema.yaml`: canonical entities and fields
- `keywords.yaml`: synonym and intent hints
- `rules.yaml`: context and boost logic
- `metrics.yaml`: business weighting rules
- `validations.yaml`: quality constraints
- `questions.yaml`: human clarification prompts
- `classifier_rules.yaml`: Business Meaning role classification policy

This keeps the system highly configurable without changing core Python logic.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
