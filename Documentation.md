# Recommendation System - Highly Enriched Architecture Documentation

This documentation provides an exhaustive, highly detailed technical overview of the `RecSys` repository. It breaks down every single module, file, class, and function, explicitly explaining their parameters, internal logic, and how they connect to form the overall autonomous recommendation pipeline.

---

## 1. Business Meaning Engine
**Purpose:** Step 2 of the pipeline. It consumes the semantic mappings from the Mapping Engine and dynamically classifies the business role of each column (e.g., personalization, promotion, curation, user metadata) based on configurable rules.

### `Business_Meaning/business_layer.py`
**Purpose:** The central orchestrator for the Business Meaning phase.
- **`BusinessMeaningLayer` (Class)**
  - `__init__(output_dir, questions_path, auto_mode, auto_answers, rules_path)`: Initializes the layer, loading the configuration rules from `classifier_rules.yaml`. Instantiates the `RoleClassifier`, `WeightAssigner`, and `QuestionFlow`.
  - `run(resolved_mappings, save_output)`: The main execution loop. It iterates over the `resolved_mappings` output by the Mapping Engine. For each column, it calls `classify_role`. If it's an interaction column, it calls the `WeightAssigner`. It creates provenance traces. Then, it calls the `ambiguity_detector`. If ambiguous, it uses `QuestionFlow` to prompt the user (or uses auto-answers). It partitions the results into `role_buckets` (features, promotion, curation, etc.) and interaction signals. Finally, invokes the `ConfigWriter`.
- **`_make_provenance(rec, confidence, status)`**: Helper function that creates a metadata trace dictionary for debugging exactly why a column was assigned a specific role (e.g., "Matched regex rule X with confidence 0.95").
- **`_print_summary(...)`**: Outputs a rich visual summary to the terminal detailing the final categorization of columns before it is saved to disk.

### `Business_Meaning/role_classifier.py`
**Purpose:** Classifies the business intent of a column based on a cascading rule engine.
- **`RoleClassifier` (Class)**
  - `__init__(rules, semantic_model)`: Sets up the rule engine and the underlying semantic classifier model.
  - `classify(mapping)`: Iterates through the ordered rules in `classifier_rules.yaml`. It checks semantic matches, exact string matches, and regex conditions using `_check_condition`. Returns the first successfully matched role and its confidence score.
  - `_check_condition(rule, entity, field, schema_sr, source_col)`: Evaluates if a column meets the specific criteria of a rule (e.g., checking if the field is named exactly "is_promoted" or matches a regex pattern).
- **`classify_role(...)`**: A global functional wrapper to instantiate the class and run the classification logic.

### `Business_Meaning/semantic_classifier.py`
**Purpose:** Employs NLP dense embeddings to classify unstructured column names based on semantic similarity to known business concepts.
- **`SemanticRoleClassifier` (Class)**
  - `__init__(anchors, model_name)`: Loads a HuggingFace SentenceTransformer model and a dictionary of baseline intent anchors (conceptual phrases).
  - `_fit_anchors()`: Pre-computes and caches dense vectors (embeddings) for all standard anchor texts for fast cosine similarity lookups.
  - `classify(text, custom_anchors)`: Encodes the input `text` (the raw column name) and compares it via cosine similarity to all anchor embeddings, returning the highest scoring match. This catches columns like "was_purchased" matching to the "interaction" anchor.

### `Business_Meaning/weight_assigner.py`
**Purpose:** Assigns numerical weights to user interaction signals (e.g., clicks, purchases, adds-to-cart) to dictate their impact in the downstream ranking algorithm.
- **`WeightAssigner` (Class)**
  - `__init__(rules)`: Loads the weighting rules configuration.
  - `assign(col_name, field_name)`: Looks up the field in the rules dict and returns the numerical weight, a human-readable description, and a boolean indicating whether the weight is high-impact enough to require human review.
- **`assign_weight(...)`**: A global functional wrapper for the class.

### `Business_Meaning/ambiguity_detector.py`
**Purpose:** Detects low-confidence classifications and determines if human intervention is required.
- **`is_ambiguous(col_name, role, role_confidence, mapping_confidence, mapping_status, config, weight_needs_review)`**: Checks thresholds from `classifier_rules.yaml`. If `role_confidence` or `mapping_confidence` is below the configured threshold, or the mapping status is already flagged as "AMBIGUOUS", it returns True. It also returns True if the user needs to manually review a high-impact interaction weight.

### `Business_Meaning/question_flow.py`
**Purpose:** Manages the human-in-the-loop interaction for ambiguous columns.
- **`QuestionFlow` (Class)**
  - `__init__(config, questions_path, auto_mode, auto_answers)`: Parses the YAML questions file to prepare the prompts.
  - `resolve(col_name, current_role, reasons, entity, field)`: Displays context to the user via the terminal explaining why a column is ambiguous, then asks a multiple-choice question to manually assign the correct `role`. If `auto_mode` is True, it bypasses the terminal and consults the `auto_answers` dictionary instead.
- **`_load_questions(...)`**: Reads `questions.yaml` into memory.
- **`_find_yaml_question(...)`**: Looks up the specific question template based on the canonical entity and field involved in the ambiguity.
- **`_ask_role(...)`**: Handles the terminal `input()` loop, validating user responses against a list of valid business roles.

### `Business_Meaning/config_writer.py`
**Purpose:** Safely serializes the finalized business configuration into JSON.
- **`ConfigWriter` (Class)**
  - `__init__(output_dir)`: Sets paths for `rec_config.json` and `rec_config.lock`.
  - `check_lock()`: Ensures that the lock file does not exist, preventing race conditions where the Model Engine tries to read the config while it is still being written.
  - `build_config(role_buckets, interaction_signals, ambiguous, provenance, weighted_role)`: Transforms the Python dictionaries into the strict JSON schema required by the Model Engine, separating nodes for "feature_columns", "interaction_signals", and "business_goals".
  - `write(config)`: Safely writes the JSON to disk, handling numpy data types via `_to_native`.
  - `load_existing()`: Parses an already-written config if the pipeline is instructed to skip Step 2.
- **`_to_native(obj)`**: A recursive helper function to cast numpy floats/ints to standard Python types for JSON serialization.

### `Business_Meaning/run.py`
**Purpose:** CLI entry point for the Business Meaning Layer.
- **`parse_args()`**: Parses CLI arguments like `--auto` and `--auto_answers`.
- **`load_mappings_from_file(path)`**: Reads the JSON output from Step 1.
- **`run_mapping_engine(args)`**: A wrapper that optionally triggers the Mapping Engine programmatically if Step 1 output doesn't exist yet.
- **`main()`**: The entry point that orchestrates the Business Meaning execution.

### `Business_Meaning/test_business_layer.py`
**Purpose:** Integration testing.
- **`main()`**: A self-contained smoke test that runs the entire business logic against a sample dataset to ensure no syntax or logical flow errors exist.

---

## 2. Mapping Engine
**Purpose:** Step 1 of the pipeline. It automatically maps raw, messy, and arbitrary database schemas to canonical, standardized e-commerce entities (like "Product", "User", "Interaction").

### `Mapping_Engine/Domain_loader/domain_loader.py`
**Purpose:** Loads and parses the domain logic from various YAML files.
- **`build_target_registry(schema, keywords, metrics, rules, validations, questions)`**: Central registry builder. It loads multiple YAML manifests and stitches them together into a unified target schema. This includes attaching business rules, validation criteria, and disambiguation questions to each specific field.
- **`build_keyword_intent_index(keywords_path)`**: Parses `keywords.yaml` and builds a fast O(1) lookup dictionary mapping lowercase synonyms directly to their canonical `entity` and `field`.

### `Mapping_Engine/Domain_loader/detection.py`
**Purpose:** Utility functions for parsing YAML structures.
- **`load_yaml(path)`**: Safe YAML loading utility with error handling.
- **`detect_list_key(detect)`**: Helper to extract a list of values from a dynamic YAML node.
- **`detect_name_key(detect_names)`**: Helper to extract naming conventions.
- **`flatten_record(record, skip_keys)`**: Flattens highly nested YAML dictionaries for easier algorithmic processing downstream.

### `Mapping_Engine/embedding_matcher/embedding.py`
**Purpose:** Uses semantic embeddings to match raw columns to canonical entities.
- **`Embedding` (Class)**
  - `__init__(model_name)`: Loads the SentenceTransformer model.
  - `fit_targets(target)`: Generates semantic embeddings for all canonical fields in the target registry based on their textual descriptions.
  - `match_column(source_column)`: Calculates cosine similarity between a raw source column's name/description and all target embeddings, returning the top K nearest semantic candidate fields.

### `Mapping_Engine/Reranking/Reranker.py`
**Purpose:** Refines the raw semantic matches using exact logic, context, and data types.
- **`normalize_type(dtype)`**: Standardizes diverse database data types (e.g., `varchar` -> `string`, `int64` -> `integer`) to allow for apples-to-apples comparison.
- **`sigmoid(x)`**: Standard sigmoid activation function for score normalization.
- **`dtype_score(source_type, target_type)`**: Calculates a penalty or boost based on whether the data types of the source and target match.
- **`name_score(source_col, target_field)`**: Calculates a string similarity score between the raw column name and the target field name.
- **`table_score(source_table, target_entity)`**: Calculates context similarity. For example, if a column resides in a "users" table, it receives a significant score boost for "User" entity targets.
- **`_confidence_label(score)`**: Converts a continuous probability score into categorical labels (e.g., "HIGH", "MEDIUM", "AMBIGUOUS").
- **`_build_reason(signals, label)`**: Formats the numerical signals into a human-readable explanation string for the provenance trace.
- **`apply_dynamic_logic(source_name, source_table, candidate, source_intent)`**: The core logic function. It applies heuristic rules, checking the `keyword_intent` to see if there's an exact synonym match, and applies contextual rules defined in the YAML.
- **`rerank_candidates(source_column, candidates, cross_encoder, keyword_intent)`**: Takes the top K candidates from the embedding matcher, scores them using the heuristic functions, applies dynamic logic, and sorts them by final, normalized confidence.

### `Mapping_Engine/Reranking/Resolver.py`
**Purpose:** Finalizes the mappings across the entire schema.
- **`resolve_mappings(all_source_results, threshold)`**: Enforces a strict 1:1 mapping constraint. If two source columns both confidently map to the exact same target field, it resolves the conflict by taking the one with the highest confidence score, leaving the other unmapped or mapping it to its runner-up target.

### `Mapping_Engine/text_builder/text_builder.py`
**Purpose:** Textual preparation for embeddings.
- **`build_target_text(target)`**: Constructs a rich, concatenated string of a canonical field's properties (name, description, rules, keywords) to create a dense semantic representation for the embedding matcher.
- **`build_source_text(source_column)`**: Constructs a semantic string for the raw database column to be embedded.

---

## 3. Model Engine / Pipeline
**Purpose:** The final execution phase. It builds the system configuration, runs the atomic data refinery, executes the underlying RecBole recommendation model, and performs the adaptive inference logic for personalized scoring and quota-based selection.

### `Model_Engine/pipeline/config.py`
**Purpose:** Centralized configuration state.
- **`PipelineConfig` (Dataclass)**: A strict dataclass holding all tunable constants. Includes `item_limit`, `batch_size`, `promo_threshold_percentile`, and the target quotas (`perso_pct`, `curation_pct`, `promo_pct`). This is the single source of truth for tuning the entire recommendation engine without altering code logic.

### `Model_Engine/pipeline/inference.py`
**Purpose:** The main entry point for generating dynamic recommendations.
- **`_apply_overrides(cfg)`**: Updates the `PipelineConfig` dataclass with optional runtime arguments provided from the CLI or API (Control Panel settings).
- **`_apply_business_goals(cfg, intelligence_map, ...)`**: Parses the `rec_config.json` business goals block (e.g., the target distribution of personalization vs promotion) and overrides the active config accordingly.
- **`_run_batch_inference(model, batch_tokens, dataset, item_rows, item_limit, config)`**: Constructs the massive interaction tensors, pushes them through the deep RecBole PyTorch model, and returns the raw AI collaborative filtering scores for the user batch.
- **`run_stage2_semantic_inference(...)`**: The master orchestrator. It loads configurations, models, and datasets. It extracts metadata, computes popularity arrays, builds boolean campaign flags, runs theme discovery, builds historical user profiles, loops over batches to compute adaptive weights and aggregate scores, and finally selects recommendations using the greedy quota selector.

### `Model_Engine/pipeline/scoring.py`
**Purpose:** Computes the final, weighted, and ranked scores for recommendations.
- **`compute_adaptive_weights(dataset, cfg)`**: Inspects dataset sparsity. If the dataset is dense (many interactions per user), it heavily weighs the AI Collaborative Filtering model (`w_personalization`). If cold/sparse, it dynamically shifts weight toward `w_curation` (themes) and `w_popularity` to prevent garbage recommendations. Returns the calculated dynamic weights.
- **`compute_per_user_scores(ai_scores, user_idx, ...)`**: Executes within the inner inference loop for a specific user. It computes the final ranking score by performing a weighted sum of the raw AI score (normalized), the strategic score (from theme clustering), the global popularity score, and the personalization match score (comparing the user's past themes against item themes). Returns the sorted indices of eligible items and their finalized scores.

### `Model_Engine/pipeline/selector.py`
**Purpose:** Selects the final items based on strict business quotas.
- **`select_recommendations(...)`**: Implements greedy quota-based selection on the sorted eligible items. It fills business quotas sequentially based on priority hierarchy: 1) Promotion (campaign items), 2) Personalization (high AI score items), 3) Curation (theme-based items). It applies diversity penalties during selection. If a quota pool is exhausted, it gracefully falls back to the next available pool to ensure the user receives a full set of `N` recommendations. Returns the final list of JSON-serializable recommendation dictionaries.

### `Model_Engine/pipeline/theme_discovery.py`
**Purpose:** Discovers implicit catalog themes without manual tagging using unsupervised NLP.
- **`AutoThemeDiscovery` (Class)**
  - `__init__(...)`: Sets up the SentenceTransformer model and the KMeans clustering algorithms.
  - `fit(descriptions)`: Generates dense text embeddings for all item descriptions. It dynamically selects the optimal `k` number of clusters using silhouette scores via `_auto_select_k`. It then fits the KMeans model to partition the entire catalog into thematic clusters.
  - `get_item_themes(item_idx)`: Returns the integer cluster label (theme) for a specific item.
  - `build_theme_map(n_items)`: Iterates through all items and builds a comprehensive mapping of `item_id -> cluster_id`.
  - `build_strategic_scores(n_items)`: Calculates how mathematically representative an item is of its specific theme by measuring the cosine similarity between the item's embedding vector and its cluster centroid. Normalizes these distances to [0,1] to be utilized as a `strategic_score` in the ranking algorithm.
- **`_auto_select_k(embeddings, k_min, k_max, sample_size)`**: A crucial helper function that iteratively tests various values of `k` (number of clusters) and computes the silhouette score to automatically pinpoint the mathematically optimal number of catalog themes without requiring human input.

### `Model_Engine/pipeline/user_profiler.py`
**Purpose:** Builds a persistent profile of a user's interests based on historical interactions.
- **`UserInterestProfiler` (Class)**
  - `__init__(dataset, item_id_to_desc, item_themes_map, cfg)`: Parses the user-item interaction history upfront in an optimized O(I) pass. It groups interactions by user and tallies the frequency of each theme the user has historically interacted with.
  - `get_profile(user_idx)`: Returns a dictionary of `{theme_id: interaction_count}` representing a specific user's historical interest distribution.
  - `get_personalization_match(user_idx, item_themes)`: Checks if a candidate item's theme exists in the user's historical profile. Returns a continuous score representing the strength of the match based on past interaction frequency.

### `Model_Engine/pipeline/reranker.py`
**Purpose:** Ensures diversity and limits over-exposure of viral items.
- **`DiversityReranker` (Class)**
  - `__init__(catalog_size, cfg)`: Computes an adaptive, non-linear exposure cap based on the formula `clip(10 / sqrt(catalog_size), 0.04, 0.30)`. This guarantees that items are not recommended to an excessive percentage of the total user base, scaling dynamically with catalog size.
  - `is_over_exposed(item_id, total_users, is_campaign)`: Tracks how many times an item has been selected globally during inference. If `(selection_count / total_users) > cap`, it flags the item as over-exposed (bypassing this check if it's an explicitly promoted campaign item).
  - `get_penalty(item_id, total_users)`: Returns a scalar penalty to be aggressively subtracted from an item's score if it's approaching its exposure cap, effectively pushing diverse items up the ranking stack.
  - `record_selection(item_id)`: Increments the global recommendation tracking counter for the given item.

### `Model_Engine/pipeline/data_utils.py`
**Purpose:** Essential helper functions for loading, parsing, and caching data.
- **`resolve_paths(cfg, base_dir)`**: Centralizes path management, returning absolute paths for models, configs, and system outputs to prevent runtime file-not-found errors.
- **`load_intelligence_map(mapping_path)`**: Reads the `rec_config.json` generated by the Business Meaning layer.
- **`load_recbole_model(config_path, cp_dir, model_name)`**: Uses the external RecBole API to initialize the modeling configuration, load the massive PyTorch dataset into memory, and instantiate the trained collaborative filtering model weights.
- **`extract_item_metadata(dataset, intelligence_map)`**: Extracts feature arrays from the RecBole dataset, specifically grabbing the `item_id` and textual `description` features.
- **`compute_popularity_scores(dataset, item_limit)`**: Tallies total global interactions for each item, applies a `log1p` transformation to dampen extreme viral outliers, and normalizes the array strictly to [0,1].
- **`build_campaign_flags(item_ids, intelligence_map, item_limit)`**: Parses the `intelligence_map` to pinpoint which item IDs belong to "promoted" or "clearance" campaigns. It returns boolean numpy arrays designed for ultra-fast, vectorized checks during the inner inference loop.

### `Model_Engine/config_builder.py`
**Purpose:** Bridging Business Logic to Model Logic.
- **`build_configs(base_dir)`**: Dynamically generates the RecBole YAML configuration files based on the structure of `rec_config.json`. It securely maps the identified business roles into RecBole's expected, rigid feature engineering schemas.

### `Model_Engine/refinery.py`
**Purpose:** The Data Preparation Engine.
- **`run_refinery(data_path, config_path, output_dir)`**: Processes the raw CSV interaction data. Executes data cleaning, NaN handling, categorical label encoding, and feature extraction based strictly on the dynamic configuration, outputting the finalized `.inter`, `.user`, and `.item` TSV files required by the Model Engine for training.

### `Model_Engine/run_stage1.py`
**Purpose:** Model Training execution.
- **`patched_load()`**: Applies a Python monkey-patch to bypass specific RecBole library bugs related to numpy data loading in newer Python versions.
- **`run_training(config_file)`**: Executes the massive PyTorch model training loop using the auto-generated configuration, evaluating epochs and saving the optimized trained checkpoints to disk.
