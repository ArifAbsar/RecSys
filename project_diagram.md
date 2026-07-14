# RecSys — Complete Technical Architecture



---

## Table of Contents

| Part | Topic |
|------|-------|
| [Part 1](#part-1--project-file-structure) | Project File Structure |
| [Part 2](#part-2--api-boot-sequence) | API Boot Sequence |
| [Part 3](#part-3--domain-loading--yaml-manifests) | Domain Loading — YAML Manifests |
| [Part 4](#part-4--text-building--bi-encoder-embedding) | Text Building & Bi-Encoder Embedding |
| [Part 5](#part-5--reranker--5-signal-scoring) | Reranker — 5-Signal Scoring |
| [Part 6](#part-6--11-global-resolver) | 1:1 Global Resolver |
| [Part 7](#part-7--business-meaning-layer) | Business Meaning Layer |
| [Part 8](#part-8--refinery--config-builder) | Refinery & Config Builder |
| [Part 9](#part-9--model-training) | Model Training |
| [Part 10](#part-10--autotherapydiscovery--theme-engine) | AutoThemeDiscovery — Theme Engine |
| [Part 11](#part-11--inference-loop--scoring--selection) | Inference Loop — Scoring & Selection |
| [Part 12](#part-12--master-diagram) 

---

## Part 1 — Project File Structure

> Every file in the project and what it does.

```
e:/RecSys/
│
├── api_service.py                  ← Entry point. Creates Flask app, starts warmup thread.
│
├── api/
│   ├── __init__.py                 ← App factory: create_app(), registers blueprints
│   ├── state.py                    ← Global singletons (MATCHER, CROSS_ENCODER, etc.)
│   ├── pipeline.py                 ← run_headless_mapping_pipeline() — core job runner
│   └── routes/
│       ├── __init__.py             ← Re-exports health_bp, jobs_bp, orchestrator_bp
│       ├── health.py               ← GET /api/health, GET /api/schema-details
│       ├── jobs.py                 ← ENV switch: local vs AWS
│       ├── jobs_local.py           ← All 5 job routes — local file + InMemoryStore
│       └── jobs_aws.py             ← All 5 job routes — S3 + Glue + DynamoDB
│
├── Mapping_Engine/
│   ├── __init__.py
│   ├── Domain_loader/
│   │   ├── __init__.py
│   │   ├── domain_loader.py        ← build_target_registry(), build_keyword_intent_index()
│   │   ├── detection.py            ← Domain detection helpers
│   │   └── main.py                 ← CLI entrypoint for loader
│   ├── embedding_matcher/
│   │   ├── __init__.py
│   │   └── embedding.py            ← class Embedding: fit_targets(), match_column()
│   ├── text_builder/
│   │   ├── __init__.py
│   │   └── text_builder.py         ← build_source_text(), build_target_text()
│   ├── Reranking/
│   │   ├── __init__.py
│   │   ├── Reranker.py             ← rerank_candidates(), 5-signal scoring formula
│   │   └── Resolver.py             ← resolve_mappings(), global 1:1 constraint
│   ├── sample_columns.json         ← Example source columns for testing
│   └── test.py                     ← Manual test runner
│
├── Business_Meaning/
│   ├── __init__.py
│   ├── business_layer.py           ← BusinessMeaningLayer orchestrator
│   ├── role_classifier.py          ← RoleClassifier, classify_role()
│   ├── semantic_classifier.py      ← SemanticRoleClassifier (embedding fallback)
│   ├── weight_assigner.py          ← WeightAssigner, assign_weight()
│   ├── ambiguity_detector.py       ← is_ambiguous()
│   ├── question_flow.py            ← QuestionFlow, _load_questions(), _ask_role()
│   ├── config_writer.py            ← ConfigWriter, builds + writes rec_config.json
│   ├── run.py                      ← Standalone CLI runner for Business Meaning
│   └── test_business_layer.py      ← Unit tests
│
├── Model_Engine/
│   ├── refinery.py                 ← run_refinery(): CSV → RecBole atomic files
│   ├── config_builder.py           ← build_configs(): writes training YAML files
│   ├── run_stage1.py               ← Train LightGCN + DeepFM via RecBole
│   ├── run_stage2.py               ← User Control Panel → calls inference.py
│   ├── default_heuristics.yaml     ← Fallback rating heuristics
│   ├── intelligent_rules.yaml      ← Extra business rules
│   ├── ranking_config.yaml         ← [AUTO-GENERATED] DeepFM config
│   ├── retrieval_config.yaml       ← [AUTO-GENERATED] LightGCN/SASRec config
│   ├── model_choice.json           ← [AUTO-GENERATED] suggested model name
│   ├── checkpoints/                ← LightGCN-*.pth, DeepFM-*.pth
│   ├── rec/                        ← rec.inter, rec.item, rec.user (RecBole TSV)
│   └── pipeline/
│       ├── __init__.py
│       ├── config.py               ← @dataclass PipelineConfig — all tunable knobs
│       ├── data_utils.py           ← resolve_paths, load_recbole_model, popularity scores
│       ├── inference.py            ← run_stage2_semantic_inference() — main orchestrator
│       ├── theme_discovery.py      ← AutoThemeDiscovery — KMeans clustering
│       ├── user_profiler.py        ← UserInterestProfiler — theme weights per user
│       ├── scoring.py              ← compute_per_user_scores() — weighted formula
│       ├── selector.py             ← select_recommendations() — quota-based greedy
│       └── reranker.py             ← ThompsonReranker — Beta distribution multiplier
│
├── ecommerce/                      ← Domain Knowledge (the "brain" of Phase 1)
│   ├── schema.yaml      (180 KB)   ← Canonical entities + fields definition
│   ├── keywords.yaml     (80 KB)   ← Per-field synonyms + match priority
│   ├── rules.yaml        (29 KB)   ← Field/entity recognition + conflict rules
│   ├── metrics.yaml      (11 KB)   ← Derived metrics that reference fields
│   ├── validations.yaml  (77 KB)   ← Data quality checks per entity
│   ├── questions.yaml    (15 KB)   ← Business context Q&A for ambiguous fields
│   ├── classifier_rules.yaml(3 KB) ← Role classification rules + signal weights
│   └── manifest.yaml     (1.2 KB) ← Domain manifest metadata
│
├── output/
│   ├── rec_config.json             ← THE bridge: mapping result used by all later phases
│   ├── rec_config.lock             ← Lock file to prevent concurrent writes
│   ├── final_recommendations.json  ← Final output: per-user recommendations
│   └── uploads/                    ← Saved CSV files from API uploads
│
├── data/
│   ├── steam-200k.csv     (8.5 MB) ← Sample Steam gaming dataset
│   └── steam_with_headers.csv      ← Same with explicit headers
│
├── data.csv               (43 MB)  ← Main working dataset
└── requirements.txt                ← Python dependencies
```

---

## Part 2 — API Boot Sequence

> What happens from `python api_service.py` until the first request can be served.

```mermaid
sequenceDiagram
    participant OS  as Terminal
    participant AS  as api_service.py
    participant AF  as api/__init__.py
    participant HB  as health.py (Blueprint)
    participant JB  as jobs.py (Blueprint)
    participant ST  as api/state.py
    participant DL  as domain_loader.py
    participant EM  as embedding.py
    participant CE  as CrossEncoder

    OS->>AS: python api_service.py
    AS->>AF: create_app()

    AF->>HB: register health_bp
    Note over HB: GET /api/health<br/>GET /api/schema-details

    AF->>JB: register jobs_bp
    Note over JB: APP_ENV=local → jobs_local.py<br/>APP_ENV=production → jobs_aws.py

    AF->>ST: warm_models_background()
    Note over AF,ST: Flask is ALREADY running and answering<br/>requests at this point. Models load async.

    ST->>ST: Thread(daemon=True).start()

    rect rgb(40, 60, 80)
        Note over ST,CE: Background Thread — runs while Flask handles traffic
        ST->>DL: Step 1/4 — build_target_registry(schema, keywords, metrics, rules, validations, questions)
        DL-->>ST: TARGET_REGISTRY list of N dicts

        ST->>EM: Step 2/4 — Embedding("all-MiniLM-L6-v2")
        Note over EM: Downloads or loads from cache<br/>SentenceTransformer model

        ST->>EM: Step 3/4 — MATCHER.fit_targets(TARGET_REGISTRY)
        Note over EM: Encodes every target field as 384-dim vector<br/>Stores as (N × 384) matrix in RAM

        ST->>CE: Step 4/4 — CrossEncoder("cross-encoder/stsb-distilroberta-base")
        CE-->>ST: CROSS_ENCODER ready

        ST->>DL: build_keyword_intent_index(keywords.yaml)
        DL-->>ST: KEYWORD_INTENT dict

        ST->>ST: MODEL_READY = True
    end

    Note over OS: GET /api/health now returns {status: ok}
```

**Global singletons stored in `api/state.py`:**

| Variable | Type | What it holds |
|----------|------|---------------|
| `MODEL_READY` | `bool` | Gates all pipeline requests |
| `TARGET_REGISTRY` | `list[dict]` | All canonical fields loaded from 6 YAMLs |
| `MATCHER` | `Embedding` | Bi-encoder + pre-encoded target matrix |
| `CROSS_ENCODER` | `CrossEncoder` | Cross-encoder reranker model |
| `KEYWORD_INTENT` | `dict` | synonym → `{canonical_field, entity, priority}` |
| `ENTITY_THRESHOLD` | `float` | From `questions.yaml` — default 0.80 |
| `FIELD_THRESHOLD` | `float` | From `questions.yaml` — default 0.80 |

---

## Part 3 — Domain Loading — YAML Manifests

> How 6 YAML files are merged into one rich `TARGET_REGISTRY`.

```mermaid
flowchart TD
    subgraph YAMLS["ecommerce/ — 6 Input Files"]
        direction LR
        Y1["schema.yaml\n180 KB\nEntities + Fields\ndefinition"]
        Y2["keywords.yaml\n80 KB\nSynonyms per field\nmatch_priority"]
        Y3["rules.yaml\n29 KB\nField recognition\nConflict resolution"]
        Y4["metrics.yaml\n11 KB\nDerived metrics\nuses_fields references"]
        Y5["validations.yaml\n77 KB\nData quality checks\nper entity"]
        Y6["questions.yaml\n15 KB\nBusiness Q&A\nmaps_to: entity.field"]
    end

    subgraph DL["domain_loader.py — build_target_registry()"]
        direction TB
        D1["Parse schema.yaml\nLoop entities → loop fields\nExtract: name, data_type,\nsemantic_role, description,\nrequirement_level, source_system_hints"]

        D2["Merge keywords.yaml\nFor each field_name:\n  synonyms → list\n  match_priority → high/med/low"]

        D3["Merge metrics.yaml\nIf field in metric.uses_fields:\n  append context_rule string"]

        D4["Merge rules.yaml\nIf field_name in rule.logic OR name:\n  append context_rule string"]

        D5["Merge validations.yaml\nIf entity matches OR field in check:\n  append context_rule string"]

        D6["Merge questions.yaml\nIf field_id OR entity in maps_to:\n  append Business Ambiguity Question"]

        D1 --> D2 --> D3 --> D4 --> D5 --> D6
    end

    subgraph OUT["OUTPUT — One dict per field"]
        direction LR
        T["TARGET_REGISTRY entry\nschema_name: universal_ecommerce\nfield_id: PRODUCT.price\nentity: PRODUCT\nentity_purpose: ...\nentity_covers: list\nentity_business_types: list\nfield: price\ndata_type: float\nsemantic_role: pricing\ndescription: Current selling price\nrequirement_level: required\nrequired_when: list\nsource_system_hints: list\nsynonyms: cost, amount, value, charge\nmatch_priority: high\ncontext_rules: list of strings"]
    end

    subgraph KI["build_keyword_intent_index()"]
        K1["Loop keywords.yaml field_keywords\nFor each synonym in field:\n  key = synonym.lower().replace spaces/hyphens with _\n  index[key] = canonical_field, canonical_entity, match_priority\nReturn reverse-lookup dict"]
    end

    YAMLS --> DL --> OUT
    Y2 --> KI
```

**Example `context_rules` list on one field:**
```
"Metric 'revenue': total revenue earned by the business"
"Rule 'price_conflict': if price and cost both present, prioritize price"
"Validation 'positive_price': price must be > 0"
"Business Ambiguity Question: Is this the final sale price or the list price?"
```

---

## Part 4 — Text Building & Bi-Encoder Embedding

> How source columns and target fields are turned into semantic text, then compared.

```mermaid
flowchart LR
    subgraph SRC["Source Column Input"]
        SC["source_column dict\ntable: orders\ncolumn: purchase_amt\ndata_type: number\nsamples: 12.50, 99.99, 7.00"]
    end

    subgraph TGT["Target Field from Registry"]
        TC["target dict\nentity: PRODUCT\nfield: price\nsynonyms: cost, amount, value\nsemantic_role: pricing\ncontext_rules: list\ndescription: Current selling price\n...all 15+ fields"]
    end

    subgraph TB["text_builder.py"]
        direction TB
        BS["build_source_text(col)\nTable: orders\nColumn: purchase_amt\nData type: number\nSample values: 12.50, 99.99, 7.00\nNull rate: ...\nDomain: ecommerce"]

        BT["build_target_text(target)\nSchema: universal_ecommerce\nEntity: PRODUCT\nEntity purpose: ...\nEntity covers: ...\nField: price\nSynonyms: cost, amount, value, charge\nData type: float\nSemantic role: pricing\nDescription: Current selling price\nRequirement level: required\nRequired when: ...\nBusiness types: ...\nSource system hints: ...\nDomain Knowledge: Metric 'revenue'...\nDomain: ecommerce"]
    end

    subgraph EMB["embedding.py — class Embedding"]
        direction TB
        FT["fit_targets(TARGET_REGISTRY)\nCalled ONCE at startup\nEncodes ALL N target texts\nResults in (N × 384) matrix\nStored in self.target_embeddings"]

        MC["match_column(source_col)\nEncode source_text → (1 × 384) vector\ncosine_similarity(source_vec, target_matrix)\nReturns all N targets sorted by score DESC\nEach entry has score: float added"]
    end

    SRC --> BS
    TGT --> BT
    BS & BT --> EMB
    FT -.->|"pre-computed"| MC
```

**Why two separate text formats?**

- `build_source_text` keeps it **short** — only what we know about the raw column
- `build_target_text` is **rich** — includes synonyms, rules, metrics, questions — so the embedding captures full business context

---

## Part 5 — Reranker — 5-Signal Scoring

> How every candidate gets a precise confidence score using 5 independent signals.

```mermaid
flowchart TD
    IN["Input\nsource_column dict + candidates list\n+ CrossEncoder model + KEYWORD_INTENT"]

    subgraph PREP["Preparation"]
        P1["Resolve source_intent\nKEYWORD_INTENT.get(source_name)\n→ canonical_field, canonical_entity, priority\nOR None if not in index"]
        P2["Build CrossEncoder pairs\nFor each candidate:\n  s_text = 'Source column X from table Y'\n  t_text = 'Target field Z in entity W.\n            Business meaning: ...\n            Synonyms: ...\n            Rules: ...'"]
        P1 --> P2
    end

    subgraph SIGNALS["5 Signals — computed per candidate"]
        direction TB

        S1["Signal 1 — Bi-Encoder Score\nWeight: 25%\nSource: cosine_similarity from match_column()\nRange: 0.0 to 1.0\nMeaning: Semantic similarity of full text descriptions"]

        S2["Signal 2 — Cross-Encoder Score\nWeight: 50%\nSource: CrossEncoder.predict(pairs) → sigmoid(raw)\nRange: 0.0 to 1.0\nMeaning: Contextual relevance — understands relationships\nThe MOST important signal"]

        S3["Signal 3 — Name Similarity\nWeight: 15%\nSource: max(fuzz.token_sort_ratio, fuzz.partial_ratio) / 100\nRange: 0.0 to 1.0\nMeaning: Do the column name and field name look alike?\nExample: purchase_amt vs price → low\n         order_id vs order_id → 1.0"]

        S4["Signal 4 — Table-Entity Match\nWeight: 5%\nSource: table_score(source_table, target_entity)\nRegex strips: tbl_, stg_, fact_, dim_, v_ prefixes\nAlso strips trailing 's'\nMeaning: Does table name suggest this entity?\nExample: tbl_orders vs ORDER → 1.0"]

        S5["Signal 5 — Data Type Match\nWeight: 5%\nSource: dtype_score(source_type, target_type)\nNormalizes: int/float/double → number\n            varchar/text/char → string\n            date/timestamp → datetime\nMeaning: Types align? 1.0 or 0.0"]

        S6["Logic Boost — additive penalty/bonus\nWeight: ±0.30 max additive\nThree YAML-driven sub-signals:\n  A. Synonym match: source_name in candidate.synonyms\n     Boost += priority_boost (high=0.30, med=0.20, low=0.10)\n  B. Canonical intent match: source_intent matches target field+entity\n     Boost += priority_boost; if entity mismatch → Boost -= 0.08\n  C. Context rules: rule says 'prioritize TABLE over OTHER'\n     Boost += 0.20 or -= 0.20 based on position in rule"]

        S1 & S2 & S3 & S4 & S5 & S6
    end

    subgraph FORMULA["Final Confidence Formula"]
        F1["base_confidence =\n  0.25 × bi_semantic\n+ 0.50 × cross_semantic\n+ 0.15 × name_score\n+ 0.05 × table_score\n+ 0.05 × dtype_score"]

        F2["final_confidence = clamp(base + logic_boost, 0.0, 1.0)"]

        F3["Confidence label:\n  >= 0.80 → High\n  >= 0.60 → Medium\n  < 0.60  → Low"]

        F4["Reasoning text built from which signals fired:\n  'High confidence: semantic similarity,\n   contextual relevance, name match'"]

        F1 --> F2 --> F3 & F4
    end

    OUT["Output: reranked list\nSorted DESC by final_confidence\nEach candidate has:\n  final_confidence: float\n  signals: dict of all 5 raw values\n  reasoning: human-readable string"]

    IN --> PREP --> SIGNALS --> FORMULA --> OUT
```

---

## Part 6 — 1:1 Global Resolver

> Ensures no two source columns map to the same canonical field.

```mermaid
flowchart TD
    IN["Input\nall_source_results list\nEach item: source column + its reranked candidates\nthreshold = 0.60"]

    subgraph FLAT["Step 1 — Flatten all pairs"]
        FL["Create one big list of ALL possible mappings\nFor every source column:\n  For every candidate:\n    Add: source_key, target_field_id, score\nResult: potentially thousands of pairs"]
    end

    subgraph SORT["Step 2 — Sort globally"]
        SO["Sort ALL pairs DESC by final_confidence\nBest match in the whole dataset goes first\nNo per-column priority — global competition"]
    end

    subgraph GREEDY["Step 3 — Greedy assignment"]
        G1{"source_key already\nmapped?"}
        G2{"target_field_id\nalready taken?"}
        G3{"score >= threshold\n(0.60)?"}

        G4["ASSIGN this mapping\nMark source as mapped\nMark target as taken"]
        G5["status = NEEDS_REVIEW\n(0.60 ≤ score < 0.99)"]
        G6["status = AUTO_ACCEPTED\n(score ≥ 0.99)"]
        G7["Skip — both sides\nmust be free"]
        G8["status = CUSTOM_FIELD\nNo canonical match\nEntity = CUSTOM"]

        G1 -->|yes| G7
        G1 -->|no| G2
        G2 -->|yes| G7
        G2 -->|no| G3
        G3 -->|yes| G4 --> G5
        G5 -->|score ≥ 0.99| G6
        G3 -->|no| G8
    end

    subgraph UNMAPPED["Step 4 — Handle unmapped sources"]
        U1["Any source column not yet in final_mappings?\nAdd with status = CUSTOM_FIELD\nentity = CUSTOM\nfield = source column name\nconfidence = top candidate score"]
    end

    OUT["Output: list of resolved_mapping dicts\nEach has:\n  source_table, source_column\n  status: AUTO_ACCEPTED / NEEDS_REVIEW / CUSTOM_FIELD\n  recommended_mapping: entity + field\n  confidence: float\n  top_candidates: top 5 alternatives"]

    IN --> FLAT --> SORT --> GREEDY --> UNMAPPED --> OUT
```

**Why global greedy instead of per-column best match?**

If two columns both want the same field, the one with higher confidence wins. The loser gets its second-best choice instead — preventing duplicate mappings that would break the recommendation model.

---

## Part 7 — Business Meaning Layer

> Classifies what role each mapped column plays in the recommendation system.

```mermaid
flowchart TD
    IN["Input\nresolved_mappings from Resolver\nclassifier_rules.yaml loaded"]

    subgraph ROLE["Role Classification — role_classifier.py"]
        direction TB
        RC1["Extract from mapping:\n  source_col, entity, field, semantic_role"]

        RC2{"Check OVERRIDES list\nYAML exact match on\nentity + field/field_contains/field_starts\nor semantic_role"}
        RC2 -->|match| RC3["Return role, confidence=1.0\nHardest rule — always wins"]
        RC2 -->|no match| RC4

        RC4{"Check SCHEMA_ROLE_MAP\nsemantic_role string\nmaps directly to role"}
        RC4 -->|match| RC5["Return role, confidence=0.90"]
        RC4 -->|no match| RC6

        RC6["Loop SCORING_RULES\nWeighted rules with conditions:\n  entity match\n  field_contains keywords\n  field_matches list\n  field_starts prefixes\n  if_schema_role match\n  semantic_similarity threshold\nReturn best_role, best_weight"]

        RC6 -->|no rule matched| RC7["FALLBACK:\nSemanticRoleClassifier\nClassify field+source_col+entity\nvs pre-computed anchor embeddings\nReturn closest role + cosine score"]

        RC1 --> RC2
        RC5 & RC7 --> ROLE_OUT["role: identity / content_features\n      interaction_signals / filters / ignore\nrole_confidence: float 0-1"]
    end

    subgraph WEIGHT["Weight Assignment — weight_assigner.py\nOnly for role == interaction_signals"]
        W1["Loop interaction_signal_weights.rules in YAML\nFor each rule:\n  keywords: list of strings\n  weight: float (e.g. 0.9 for purchase)\nSearch keywords in col_name + field_name\nReturn first match weight"]
        W2["No match → default_weight, needs_review=True"]
        W1 -->|match| WOUT["weight: float\ndescription: string\nneeds_review: bool"]
        W1 -->|no match| W2 --> WOUT
    end

    subgraph AMBIG["Ambiguity Detection — ambiguity_detector.py"]
        A1["Check 3 conditions:\n  1. role == unknown_sentinel\n  2. role_confidence < role_threshold (0.80)\n  3. mapping_status == needs_review\n     AND map_conf < map_threshold (0.75)"]
        A2["Return flagged=True + reasons list\nor flagged=False"]
    end

    subgraph QFLOW["Question Flow — question_flow.py\nOnly for flagged columns"]
        Q1["_find_yaml_question(questions, entity, field)\nSearch questions.yaml maps_to list\nSupports exact match + dotted notation\n'customer.customer_id'"]
        Q2["_ask_role(col, role, reasons, yaml_question)\nprint reasons + current suggestion\nprint yaml question if found\nPrint role options 1-N or 's' to skip\nIf auto_mode → use auto_answers dict\nElse → input() from stdin"]
        Q1 --> Q2 --> QOUT["resolved_role\n(may differ from original role)"]
    end

    subgraph BUCKET["Role Bucketing"]
        B1["identity → role_buckets['identity'].append(col)"]
        B2["content_features → role_buckets['content_features'].append(col)"]
        B3["filters → role_buckets['filters'].append(col)"]
        B4["interaction_signals → interaction_signals[col] = weight"]
        B5["ignore → role_buckets['ignore'].append(col)"]
    end

    subgraph WRITE["Config Writer — config_writer.py"]
        CW1["check_lock()\nRaises if rec_config.lock exists\nPrevents overwrite while training reads file"]
        CW2["build_config(\n  role_buckets, interaction_signals,\n  ambiguous, provenance\n)\n→ dict with all roles + metadata"]
        CW3["_to_native() — convert numpy types\nto Python float/int for JSON serialization"]
        CW4["write() → output/rec_config.json"]
        CW1 --> CW2 --> CW3 --> CW4
    end

    IN --> ROLE --> WEIGHT & AMBIG
    AMBIG -->|flagged| QFLOW
    QFLOW --> BUCKET
    ROLE --> BUCKET
    WEIGHT --> BUCKET
    BUCKET --> WRITE
```

**The 5 roles explained:**

| Role | Meaning | Example columns |
|------|---------|-----------------|
| `identity` | Uniquely identifies a user or item | `customer_id`, `product_id` |
| `content_features` | Describes the item | `game_name`, `category`, `brand` |
| `interaction_signals` | How a user interacted (gets a weight) | `hours_played`, `purchase`, `rating` |
| `filters` | Used to filter the catalog | `platform`, `region`, `age_rating` |
| `ignore` | Not needed by the model | `row_id`, `internal_code` |

---

## Part 8 — Refinery & Config Builder

> Converts the raw CSV into training files the model can actually read.

```mermaid
flowchart TD
    IN1(["output/rec_config.json"])
    IN2(["data.csv  43 MB"])

    subgraph ROLES["Role Detection — _detect_roles(config)"]
        direction LR
        RD1["user_col\n1. provenance entity in USER_ENTITY_TYPES\n   CUSTOMER, USER, ACCOUNT, MEMBER, PLAYER\n2. identity list keyword scan\n   user, customer, member, player, account"]
        RD2["item_col\n1. provenance entity in ITEM_ENTITY_TYPES\n   PRODUCT, ITEM, CONTENT, GAME, ARTICLE, SKU\n2. identity field that is NOT user_col"]
        RD3["rating_col\nhighest-weighted key in interaction_signals dict"]
        RD4["timestamp_col\nprovenance field where name or mapped-field\ncontains: date, time, timestamp, ts, created, updated, at"]
        RD5["group_col optional basket\nprovenance field where name or mapped-field\ncontains: transaction_id, invoice, order_id, session_id, basket_id"]
    end

    subgraph CLEAN["Data Cleaning"]
        C1["_load_csv()\ntry UTF-8 → fallback ISO-8859-1"]
        C2["dropna on user_col + item_col\nCritical — model breaks without these"]
        C3["Rating column handling:\n  If numeric → keep as-is\n  If categorical text:\n    1. Try BML interaction_signal_rules\n       match keywords → assign weight\n    2. Fallback: default_heuristics.yaml\n       'purchase' → 1.0, 'view' → 0.3, etc.\n    3. pd.factorize for anything else"]
        C4["Remove zero/negative ratings\nDrop positive=False rows"]
        C5["If group_col found:\n  Drop rows where group_col starts with 'C'\n  (basket cancellations pattern)"]
        C1 --> C2 --> C3 --> C4 --> C5
    end

    subgraph PROFILER["PROFILER — Model Selection"]
        PR1["items_per_group =\n  df.groupby(group_col).size().mean()\n  Only if group_col found"]
        PR2["events_per_user =\n  df.groupby(user_col)[timestamp_col].nunique().mean()\n  Only if timestamp_col found"]
        PR3{"Decision tree"}
        PR3 -->|"items/group > 3.0"| ML["LightGCN\nBasket-based CF\nPeople buy groups of items"]
        PR3 -->|"events/user > 2.0"| MS["SASRec\nSequential model\nUsers have rich history"]
        PR3 -->|"else"| MB["BPR\nSimple CF\nSparse interaction data"]
        PR1 & PR2 --> PR3
        ML & MS & MB --> MC["Write model_choice.json\n{ suggested_model: LightGCN }"]
    end

    subgraph EXPORT["RecBole Atomic File Export"]
        direction TB
        E1["rec.inter  — TAB-separated\nuser_id:token  item_id:token\nrating:float  timestamp:float\nOne row per interaction"]
        E2["rec.item  — TAB-separated\nitem_id:token\nfor each content_feature:\n  feature_name:token or :float\nOne row per unique item"]
        E3["rec.user  — TAB-separated\nuser_id:token\nfor each extra signal column:\n  col_name:token or :float\nOne row per user (optional)"]
    end

    subgraph CB["Config Builder — config_builder.py"]
        direction TB
        CB1["Read rec_config.json:\n  interaction_signals → find rating_col\n  content_features → item_features list\nRead model_choice.json:\n  suggested_model name"]
        CB2["Write retrieval_config.yaml\n  model: LightGCN or SASRec or BPR\n  embedding_size: 64\n  load_col.inter: user_id, item_id, rating, timestamp\n  neg_sample: uniform=1\n  split: RS 80/10/10\n  metric: Recall@20"]
        CB3["Write ranking_config.yaml\n  model: DeepFM (always)\n  embedding_size: 16\n  LABEL_FIELD: rating\n  load_col.inter: user_id, item_id, rating, timestamp\n  load_col.item: item_id + item_features\n  load_col.user: user_id + user_features\n  threshold.rating: 0\n  eval mode: uni100"]
        CB1 --> CB2 & CB3
    end

    IN1 & IN2 --> ROLES --> CLEAN --> PROFILER
    CLEAN --> EXPORT
    PROFILER --> CB
```

---

## Part 9 — Model Training

> Two sequential training runs using RecBole framework.

```mermaid
flowchart LR
    IN1(["rec/rec.inter\nrec/rec.item\nrec/rec.user"])
    IN2(["retrieval_config.yaml\nranking_config.yaml"])

    subgraph COMPAT["Compatibility Patches — top of run_stage1.py"]
        direction TB
        CP1["numpy aliases\nnp.float_ = np.float64\nnp.int = np.int64\nnp.bool = np.bool_\nnp.complex_ = np.complex128\nnp.unicode_ = np.str_"]
        CP2["scipy patch\ndok_matrix._update = dok_matrix.update\nFixes RecBole sparse matrix bug"]
        CP3["torch.load patch\nForce weights_only=False\nfor older checkpoint format"]
    end

    subgraph RUN1["run_training(retrieval_config.yaml)"]
        direction TB
        R1["init_seed(42, True)\nReproducible training"]
        R2["Config(model=None, config_file_list=[yaml])\nLoads all settings from YAML"]
        R3["create_dataset(config)\nReads rec.inter + rec.item\nBuilds token ID maps\nHandles missing features"]
        R4["data_preparation(config, dataset)\nCreates train/valid/test splits\n80% / 10% / 10% random order"]
        R5["get_model('LightGCN')(config, dataset)\nGraph-based collaborative filtering\nUser-item bipartite graph embeddings\nembedding_size=64"]
        R6["get_trainer('Trainer', 'LightGCN')(config, model)\nAdam optimizer lr=0.001\nBatch size=2048"]
        R7["trainer.fit(train_data, valid_data, saved=True)\nEpochs: 10\nEarly stopping: 2 steps\nValidation metric: Recall@20\nSaves best model automatically"]
        R8["trainer.evaluate(test_data)\nFinal Recall@10,20,100\nNDCG, Precision metrics"]
        R1-->R2-->R3-->R4-->R5-->R6-->R7-->R8
    end

    subgraph RUN2["run_training(ranking_config.yaml)"]
        direction TB
        N1["Same setup steps as above\nDifferent config file"]
        N2["get_model('DeepFM')(config, dataset)\nDeep Factorization Machine\nLearns feature interactions\nembedding_size=16"]
        N3["Ranking mode: uni100\n100 random negative items\nper positive interaction\nModels point-wise regression"]
        N4["LABEL_FIELD=rating\nthreshold.rating=0\nfilters out zero-rating rows"]
        N5["trainer.fit() saves best checkpoint\nMetric: Recall@20"]
        N1-->N2-->N3-->N4-->N5
    end

    OUT(["Model_Engine/checkpoints/\nLightGCN-RecBole-*.pth\nDeepFM-RecBole-*.pth"])

    IN1 & IN2 --> COMPAT
    COMPAT --> RUN1 --> OUT
    COMPAT --> RUN2 --> OUT
```

---

## Part 10 — AutoThemeDiscovery — Theme Engine

> Automatically discovers what categories/themes exist in the item catalog.  
> Zero manual configuration. Works on any domain.

```mermaid
flowchart TD
    IN["Input\ndescriptions list[str]\nOne text description per item\nExtracted from rec_config.json content_features"]

    subgraph ENCODE["Step 1 — Encode All Item Descriptions"]
        E1["SentenceTransformer('all-MiniLM-L6-v2')\n.encode(descriptions, normalize=True, show_progress=True)\nResult: embeddings matrix (N_items × 384)"]
    end

    subgraph KSEL["Step 2 — Auto-Select Number of Clusters"]
        direction LR
        K1{"n items?"}
        K1 -->|"< 20"| KA["k = max(2, n//3)"]
        K1 -->|"< 100"| KB["k = sqrt(n) rounded"]
        K1 -->|">= 100"| KC["_auto_select_k(embeddings, k_min=5, k_max=50)\n\nSilhouette scan:\n  Sample 2000 items for speed\n  Try ~8 values from k_min to k_max\n  For each k:\n    MiniBatchKMeans(k, n_init=3).fit_predict(sample)\n    silhouette_score(sample, labels)\n  Pick k with best silhouette score"]
    end

    subgraph CLUSTER["Step 3 — Cluster Items"]
        C1["MiniBatchKMeans(\n  n_clusters=k,\n  random_state=42,\n  n_init=5\n).fit_predict(embeddings)\nResult: item_cluster_ids array (N_items,)\nAlso stores: cluster_centroids (k × 384)"]
    end

    subgraph LABEL["Step 4 — Label Each Cluster"]
        L1["For each cluster c:\n  members = items where cluster_id == c\n  Compute cosine_similarity(members, centroid_c)\n  rep_item = member closest to centroid\n  label = first 4 words of rep_item description\n  Deduplicate: if label used → append _c suffix"]
        L2["theme_labels dict:\n  {0: 'Action_RPG_Games',\n   1: 'Simulation_Strategy',\n   2: 'FPS_Multiplayer', ...}"]
        L1 --> L2
    end

    subgraph SCORES["Step 5 — Compute Strategic Scores"]
        S1["all_sims = cosine_similarity(embeddings, centroids)\nCached — reused for theme assignment\nShape: (N_items × k)"]
        S2["own_sim = all_sims[item, item.cluster_id]\nbest_other_sim = max(all_sims[item, all_other_clusters])"]
        S3["raw = clip(own_sim - best_other_sim, 0, +inf)\nNormalize: strategic_score = raw / raw.max()\nRange: [0, 1]\nMeaning: How much does this item define its cluster?\nHigh score = quintessential representative of its theme"]
    end

    subgraph ASSIGN["Item Theme Assignment — get_item_themes(idx)"]
        A1["primary_theme = theme_labels[item_cluster_ids[idx]]"]
        A2["secondary_theme check:\n  secondary_sim = best non-primary cluster similarity\n  If secondary_sim >= 0.85 × primary_sim:\n    also assign secondary theme\n  Max 2 themes per item"]
        A1 --> A2
    end

    OUT1["item_themes_map: list[list[str]]\nIndex i → list of theme strings for item i\nExample: item_42 → ['FPS_Multiplayer', 'Action_RPG_Games']"]
    OUT2["global_strategic_scores: np.ndarray\nShape (N_items,) float in [0,1]\nUsed as w_strategic signal in scoring"]

    IN --> ENCODE --> KSEL --> CLUSTER --> LABEL --> SCORES --> ASSIGN
    ASSIGN --> OUT1
    SCORES --> OUT2
```

---

## Part 11 — Inference Loop — Scoring & Selection

> The core algorithm that generates recommendations for every user.

```mermaid
flowchart TD
    IN1(["LightGCN checkpoint\nDeepFM checkpoint"])
    IN2(["rec_config.json\nbusiness_goals + campaign IDs"])

    subgraph INIT["One-time Initialization"]
        I1["load_intelligence_map(rec_config.json)\nBusiness goals: perso/curation/promo %\nPromoted + clearance item ID sets"]
        I2["_apply_business_goals(cfg)\nN_recs = total_recommendations from config\nperso_quota = N × perso_pct\ncuration_quota = N × curation_pct\npromo_quota = N × promo_pct\nAdjust cold/moderate/dense quotas to match"]
        I3["load_recbole_model(retrieval_config)\n→ LightGCN config, dataset, model.eval()"]
        I4["load_recbole_model(ranking_config)\n→ DeepFM config, dataset, model.eval()"]
        I5["compute_popularity_scores(dataset)\nlog1p(interaction_count) per item\nNormalized to [0,1]"]
        I6["AutoThemeDiscovery.fit(descriptions)\n→ item_themes_map, strategic_scores"]
        I7["UserInterestProfiler(dataset, themes)\n→ pre-indexes user→items dict O(I)"]
        I8["ThompsonReranker(item_limit)\n→ _counts array initialized to zeros"]
        I1-->I2-->I3-->I4-->I5-->I6-->I7-->I8
    end

    subgraph BATCH["Batch Loop — 64 users at a time"]
        direction LR
        BL1["_run_batch_inference()\nStage 1 LightGCN:\n  Interaction({user_ids: tensor})\n  .full_sort_predict() → scores all N_items\n  torch.topk(k=retrieval_top_k) → top candidates\nStage 2 DeepFM:\n  Build batch_tensors from:\n    user_feats[user_indices] repeated k times\n    item_rows[topk_indices]\n  ranking_model.predict(batch_tensors)\n  Scatter scores back to (users × N_items) array\n  Non-top-K items stay at -999.0\nReturn: batch_ai_scores (batch × N_items)"]
    end

    subgraph PERUSER["Per-User Processing"]
        direction TB
        U1["Normalize AI scores\nretrieved = ai_scores > -900.0\nai_norm = min-max on retrieved subset\nNon-retrieved items stay 0"]

        U2["Eligibility gate\nthreshold = percentile(ai_norm[retrieved], 10)\neligible = retrieved where ai_norm >= threshold\nFilter out items in user.past_item_indices"]

        U3["Personalization scores\nuser_profile = UserInterestProfiler.get_profile(user_idx)\nFor each eligible item:\n  perso[item] = max(theme_weights[t] for t in item_themes[item])\n               × perso_match_multiplier (1.5)\n               capped at 1.0"]

        U4["Jitter injection\nif max(perso[eligible]) < 0.05 threshold:\n  Add Uniform(0, 0.05) to eligible items\n  Prevents all cold users getting identical ranks"]

        U5["Tier detection\ncold     if n_interactions < 5\nmoderate if n_interactions < 20\ndense    otherwise\n\nWeights per tier (curation, strategic, perso, popularity):\ncold:     (0.65, 0.10, 0.15, 0.10)\nmoderate: (0.30, 0.10, 0.50, 0.10)\ndense:    (0.15, 0.10, 0.70, 0.05)"]

        U6["Base score\nbase =\n  w_curation    × ai_norm\n+ w_strategic   × strategic_scores\n+ w_perso       × perso_scores\n+ w_popularity  × popularity_scores\n+ 0.40          × campaign_flag\n+ jitter"]

        U7["Thompson Sampling multiplier\nα = 1 + 50 × (ai_norm + pop_scores)\nβ = 1 + _counts[item]  ← impressions so far\nmultiplier = random.beta(α, β)\nHigh-impression items: β grows → multiplier shrinks\nCold items: β=1 → wide uncertain curve → exploration"]

        U8["Final score\nfinal = clip(base × multiplier + mask, -100, 2.0)\nmask = 0 for eligible, -100 for ineligible"]

        U1-->U2-->U3-->U4-->U5-->U6-->U7-->U8
    end

    subgraph SELECT["Quota-Based Selection"]
        direction TB
        Q1["Adaptive quotas by user tier\ncold:     (1 perso, 8 curation, 1 promo)\nmoderate: (5 perso, 4 curation, 1 promo)\ndense:    (7 perso, 2 curation, 1 promo)"]

        Q2["dynamic_threshold = percentile(perso_scores, 80)\ncuration_threshold = percentile(ai_norm, 60)"]

        Q3["Greedy loop over eligible sorted by final_score DESC:\n\n  1. is_over_exposed(item)?\n     Campaign items capped at 60% of users\n     Regular items: never blocked (Thompson handles it)\n     If over-exposed → skip\n\n  2. per_theme_cap: max 2 items per primary theme\n     If this theme full → skip\n\n  3. Classify item type:\n     is_admin_push (in promoted_ids OR clearance_ids)?\n       → promotion (if enable_promotion=True)\n     perso_score >= dynamic_threshold OR theme in user.top_themes?\n       → personalization\n     else\n       → curation\n\n  4. Check quota:\n     If natural_type slot available → assign there\n     Else → find any underfilled slot (fallback)\n     If all slots full → break\n\n  5. Record:\n     reranker.record_selection(idx) → _counts[idx] += 1\n     theme_usage[primary_theme] += 1"]

        Q4["Build recommendation dict:\n  item_id, description\n  recommendation_type\n  business_boosted, campaign_type\n  scores: final, ai_relevance, strategic,\n          personalization_match, popularity\n  explanation: matched_themes, reason string"]

        Q1-->Q2-->Q3-->Q4
    end

    OUT(["output/final_recommendations.json\n{user_token: [10 recs each with\n  scores + explanations]}"])

    IN1 & IN2 --> INIT --> BATCH --> PERUSER --> SELECT --> OUT
```

---

## Part 12 — Master Diagram

> Everything connected. Read top-to-bottom, left-to-right within each phase.

```mermaid
flowchart TD

    %% ═══════════════════════════════════════════
    %% INPUTS
    %% ═══════════════════════════════════════════
    CSV(["🗃️  Raw CSV\ndata.csv  43 MB\nAny domain"])
    YAMLS(["📚 ecommerce/ YAMLs\nschema · keywords · rules\nmetrics · validations\nquestions · classifier_rules"])

    %% ═══════════════════════════════════════════
    %% PHASE 0 — STARTUP
    %% ═══════════════════════════════════════════
    subgraph P0["⚡ PHASE 0 — API STARTUP  api_service.py"]
        direction LR

        subgraph FLASK["Flask App  api/__init__.py"]
            direction TB
            F1["health_bp\n/api/health\n/api/schema-details"]
            F2["jobs_bp\n/api/jobs CRUD\n/deploy /start /update"]
        end

        subgraph STATE["Background Thread  api/state.py"]
            direction TB
            S1["1. build_target_registry()\ndomain_loader.py\n→ TARGET_REGISTRY"]
            S2["2. Embedding('all-MiniLM-L6-v2')\n→ MATCHER"]
            S3["3. MATCHER.fit_targets()\nEncode all target fields\n→ cosine matrix in RAM"]
            S4["4. CrossEncoder('stsb-distilroberta-base')\n→ CROSS_ENCODER"]
            S5["5. build_keyword_intent_index()\n→ KEYWORD_INTENT dict"]
            READY(["MODEL_READY = True"])
            S1-->S2-->S3-->S4-->S5-->READY
        end
    end

    %% ═══════════════════════════════════════════
    %% PHASE 1 — SCHEMA MAPPING
    %% ═══════════════════════════════════════════
    subgraph P1["🔍 PHASE 1 — SCHEMA MAPPING  api/pipeline.py"]
        direction TB

        subgraph HTTP["HTTP Layer  jobs_local.py / jobs_aws.py"]
            direction LR
            H1["POST /api/jobs\nParse CSV headers\nDetect dtype + samples\nCreate job in Store"]
            H2["POST /api/jobs/id/start\nthread → run_headless_mapping_pipeline()"]
            H1-->H2
        end

        subgraph LOOP["Per-Column Loop"]
            direction LR

            subgraph MAPENG["Mapping Engine"]
                direction TB
                ME1["text_builder.py\nbuild_source_text(col)\nbuild_target_text(target)"]
                ME2["Embedding.match_column()\nCosine similarity\nSource vs all targets\n→ N candidates ranked"]
                ME3["Reranker.rerank_candidates()\n0.25 × bi-encoder\n0.50 × cross-encoder\n0.15 × name fuzzy\n0.05 × table match\n0.05 × dtype match\n±0.30 YAML boost"]
                ME1-->ME2-->ME3
            end

            RES["Resolver.resolve_mappings()\nGlobal greedy 1:1 constraint\nSort all pairs DESC\nAssign if both sides free\nthreshold=0.60\n→ AUTO_ACCEPTED / NEEDS_REVIEW / CUSTOM_FIELD"]
            ME3-->RES
        end

        subgraph BMLAYER["Business Meaning Layer  business_layer.py"]
            direction LR

            subgraph RC["Role Classifier\nrole_classifier.py"]
                direction TB
                RC1["1. Overrides → 1.0"]
                RC2["2. schema_role_map → 0.90"]
                RC3["3. scoring_rules scan"]
                RC4["4. SemanticRoleClassifier\nfallback embedding"]
                RC1-->RC2-->RC3-->RC4
            end

            WA["WeightAssigner\nweight_assigner.py\nKeyword → float weight\nfor interaction_signals"]

            AD["AmbiguityDetector\nambiguity_detector.py\nrole_conf < 0.80?\nmap_conf < 0.75?\n→ flagged + reasons"]

            QF["QuestionFlow\nquestion_flow.py\nauto_answers or stdin\nResolves ambiguous cols"]

            CW["ConfigWriter\nconfig_writer.py\nbuild_config()\nwrite rec_config.json"]

            RC-->WA-->AD-->QF-->CW
        end

        HTTP-->LOOP-->BMLAYER
    end

    %% ═══════════════════════════════════════════
    %% ARTIFACTS
    %% ═══════════════════════════════════════════
    RCJ(["📄 output/rec_config.json\nidentity · content_features\nfilters · interaction_signals\nmapping_provenance\nbusiness_goals · campaigns"])

    %% ═══════════════════════════════════════════
    %% PHASE 2 — REFINEMENT
    %% ═══════════════════════════════════════════
    subgraph P2["🔧 PHASE 2 — DATA REFINEMENT  Model_Engine/refinery.py"]
        direction LR

        subgraph REF["Refinery"]
            direction TB
            R1["_detect_roles()\nProvenance → user/item/rating\ntimestamp/group columns"]
            R2["_load_csv()  UTF-8 / ISO-8859-1"]
            R3["Clean\nDrop nulls · Map ratings\nRemove cancellations"]
            R4["PROFILER\nitems/group>3 → LightGCN\nevents/user>2 → SASRec\nelse → BPR"]
            R5["Export TSV\nrec.inter  rec.item  rec.user"]
            R1-->R2-->R3-->R4-->R5
        end

        subgraph CFG["Config Builder\nconfig_builder.py"]
            direction TB
            CB1["Read rec_config.json\n+ model_choice.json"]
            CB2["Write retrieval_config.yaml\nLightGCN · SASRec · BPR\nembedding_size=64"]
            CB3["Write ranking_config.yaml\nDeepFM\nembedding_size=16"]
            CB1-->CB2 & CB3
        end

        REF-->CFG
    end

    %% ═══════════════════════════════════════════
    %% PHASE 3 — TRAINING
    %% ═══════════════════════════════════════════
    subgraph P3["🏋️ PHASE 3 — MODEL TRAINING  Model_Engine/run_stage1.py"]
        direction LR

        subgraph RET["Retrieval Model"]
            direction TB
            T1["RecBole Config + Dataset"]
            T2["LightGCN\nGraph collaborative filtering\nembedding_size=64"]
            T3["Trainer.fit()\nRecall@20 early stopping\nSave best checkpoint"]
            T1-->T2-->T3
        end

        subgraph RANK["Ranking Model"]
            direction TB
            U1["RecBole Config + Dataset"]
            U2["DeepFM\nFeature interaction model\nembedding_size=16"]
            U3["Trainer.fit()\nuni100 negative sampling\nSave best checkpoint"]
            U1-->U2-->U3
        end
    end

    CHKP(["💾 checkpoints/\nLightGCN-*.pth\nDeepFM-*.pth"])

    %% ═══════════════════════════════════════════
    %% PHASE 4 — INFERENCE
    %% ═══════════════════════════════════════════
    subgraph P4["🎯 PHASE 4 — INFERENCE  Model_Engine/run_stage2.py → pipeline/inference.py"]
        direction TB

        subgraph SETUP["One-time Setup"]
            direction LR
            ST1["AutoThemeDiscovery.fit()\nEncode items → KMeans\nAuto-k via silhouette\n→ themes + strategic scores"]
            ST2["UserInterestProfiler\nBuild user→items index O(I)\nRecency-decayed theme weights"]
            ST3["ThompsonReranker\n_counts per item\nBeta distribution"]
            ST1-->ST2-->ST3
        end

        subgraph INF["Batch Inference  _run_batch_inference()"]
            direction LR
            I1["LightGCN\nfull_sort_predict()\ntorch.topk → Top-K"]
            I2["DeepFM\npredict(top-K subset)\nScatter → full score array"]
            I1-->I2
        end

        subgraph SCORE["Per-User Scoring  scoring.py"]
            direction LR
            SC1["Normalize AI scores\nEligibility percentile gate\nRemove seen items"]
            SC2["Personalization scores\ntheme_weights × 1.5\njitter for cold users"]
            SC3["Weighted base score\nAI + Strategic + Perso\n+ Popularity + Campaign"]
            SC4["Thompson multiplier\nBeta(α,β) per item\nself-regulating diversity"]
            SC5["final = clip(base × thompson\n+ eligibility_mask)"]
            SC1-->SC2-->SC3-->SC4-->SC5
        end

        subgraph SEL["Quota Selector  selector.py"]
            direction LR
            SE1["Adaptive quotas by tier\nCOLD:     1p 8c 1promo\nMODERATE: 5p 4c 1promo\nDENSE:    7p 2c 1promo"]
            SE2["Classify each item\nPromo → Perso → Curation"]
            SE3["Greedy fill quotas\nTheme cap 2 per cluster\nRecord impressions"]
            SE1-->SE2-->SE3
        end

        SETUP-->INF-->SCORE-->SEL
    end

    FINAL(["📄 output/final_recommendations.json\nPer-user list:\nitem_id · description · type\nscores · explanation"])

    %% ═══════════════════════════════════════════
    %% CONNECTIONS BETWEEN PHASES
    %% ═══════════════════════════════════════════
    YAMLS -->|"6 YAML files\ndomain knowledge"| STATE
    STATE -.->|"MATCHER · CROSS_ENCODER\nKEYWORD_INTENT · TARGET_REGISTRY"| MAPENG

    CSV -->|"POST /api/jobs"| HTTP
    BMLAYER -->|"writes"| RCJ

    RCJ -->|"role detection\n+ rating mapping"| REF
    CSV -->|"read data"| REF
    REF -->|"model_choice.json"| CFG

    CFG -->|"YAML configs"| P3
    R5 -->|"rec.inter\nrec.item"| P3
    T3 & U3 --> CHKP

    CHKP -->|"load checkpoints"| INF
    RCJ  -->|"business_goals\ncontent_features\ncampaign IDs"| SETUP

    SEL --> FINAL
```

---

## Quick Reference — Key Numbers

| Metric | Value |
|--------|-------|
| Bi-encoder model | `all-MiniLM-L6-v2` — 384-dim vectors |
| Cross-encoder model | `cross-encoder/stsb-distilroberta-base` |
| Confidence threshold (mapping) | 0.60 |
| Auto-accept threshold | 0.99 |
| Role confidence threshold | 0.80 |
| Signal weights | 25% bi + 50% cross + 15% name + 5% table + 5% dtype |
| Cold user cutoff | < 5 interactions |
| Moderate user cutoff | < 20 interactions |
| Thompson alpha scale | 50.0 |
| Campaign exposure cap | 60% of users |
| Theme cap per user | 2 items per primary theme |
| Default quotas (dense) | 7 perso · 2 curation · 1 promo per 10 recs |
