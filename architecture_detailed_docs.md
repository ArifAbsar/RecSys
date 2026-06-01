# Recommendation System Architecture Documentation

This document provides a comprehensive overview of the files, classes, methods, and functions in the recommendation pipeline.

## Module: Business_Meaning

### File: `Business_Meaning/ambiguity_detector.py`

ambiguity_detector.py
─────────────────────
Decides whether a classified column needs human review.
Fully data-driven using the 'config' section of classifier_rules.yaml.

#### Global Functions:
- **`is_ambiguous(col_name, role, role_confidence, mapping_confidence, mapping_status, config, weight_needs_review)`**: Determine if a column classification needs human confirmation.

### File: `Business_Meaning/business_layer.py`

business_layer.py
─────────────────
BusinessMeaningLayer — the orchestrator for Step 2.
Fully data-driven using the 'config' section of classifier_rules.yaml.

#### Class: `BusinessMeaningLayer`
Orchestrates the classification, weight assignment, and ambiguity resolution.

Methods:
- **`__init__(self, output_dir, questions_path, auto_mode, auto_answers, rules_path)`**: No description available.
- **`run(self, resolved_mappings, save_output)`**: Processes mappings and returns the rec_config dict.

#### Global Functions:
- **`_make_provenance(rec, confidence, status)`**: No description available.
- **`_print_summary(config_path, role_buckets, interaction_signals, ambiguous, weighted_role, promoted, clearance)`**: No description available.

### File: `Business_Meaning/config_writer.py`

config_writer.py
────────────────
Serialises the classified column assignments into `rec_config.json`.
Fully data-driven version supporting dynamic roles.

#### Class: `LockFileExistsError`
Raised when rec_config.lock exists.


#### Class: `ConfigWriter`
Writes rec_config.json and manages the lock file.

Methods:
- **`__init__(self, output_dir)`**: No description available.
- **`check_lock(self)`**: Raise LockFileExistsError if Feature Engineering holds the lock.
- **`build_config(self, role_buckets, interaction_signals, ambiguous, provenance, weighted_role)`**: Assemble the configuration dictionary dynamically.
- **`write(self, config)`**: Write the configuration dictionary to rec_config.json.
- **`load_existing(self)`**: Load an existing rec_config.json if present.

#### Global Functions:
- **`_to_native(obj)`**: Recursively convert non-native types (numpy.float32, etc.) to standard Python types so they can be JSON serialized.

### File: `Business_Meaning/question_flow.py`

question_flow.py
────────────────
Data-driven interactive question flow for ambiguous columns.
Reads roles and config from classifier_rules.yaml.

#### Class: `QuestionFlow`
Orchestrates the data-driven ambiguity resolution Q&A.

Methods:
- **`__init__(self, config, questions_path, auto_mode, auto_answers)`**: No description available.
- **`resolve(self, col_name, current_role, reasons, entity, field)`**: Run the question flow for one ambiguous column.

#### Global Functions:
- **`_load_questions(questions_path)`**: Load all questions from questions.yaml into a flat list.
- **`_find_yaml_question(questions, entity, field)`**: Search questions.yaml for a matching question.
- **`_ask_role(col_name, current_role, reasons, yaml_question, auto_answers, valid_roles)`**: Present a role-clarification prompt.

### File: `Business_Meaning/role_classifier.py`

role_classifier.py
─────────────────
Advanced Multi-Factor Scoring Engine.
Data-driven using classifier_rules.yaml.

#### Class: `RoleClassifier`
No description available.

Methods:
- **`__init__(self, rules, semantic_model)`**: No description available.
- **`classify(self, mapping)`**: No description available.
- **`_check_condition(self, rule, entity, field, schema_sr, source_col)`**: Centralized condition evaluator for all rule types.

#### Global Functions:
- **`classify_role(mapping, semantic_model, rules)`**: No description available.

### File: `Business_Meaning/run.py`

run.py
──────
CLI entry point for the Business Meaning Layer (Step 2).
Fully data-driven and domain-agnostic.

#### Global Functions:
- **`parse_args()`**: No description available.
- **`load_mappings_from_file(path)`**: No description available.
- **`run_mapping_engine(args)`**: Run Steps 1 (Mapping Engine) and return resolved mappings.
- **`main()`**: No description available.

### File: `Business_Meaning/semantic_classifier.py`

semantic_classifier.py
──────────────────────
Intelligent role classification using semantic embeddings.

#### Class: `SemanticRoleClassifier`
Intelligent role classification using semantic embeddings. Compares column meanings against conceptual anchors or custom phrases.

Methods:
- **`__init__(self, anchors, model_name)`**: No description available.
- **`_fit_anchors(self)`**: Pre-compute embeddings for all conceptual anchors.
- **`classify(self, text, custom_anchors)`**: Compare the input text against anchors and return the best match. If custom_anchors is provided, it compares against those phrases instead.

### File: `Business_Meaning/test_business_layer.py`

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

#### Global Functions:
- **`main()`**: No description available.

### File: `Business_Meaning/weight_assigner.py`

weight_assigner.py
──────────────────
Data-driven weight assignment for interaction signals.

#### Class: `WeightAssigner`
Assigns weights to interaction signals using YAML rules.

Methods:
- **`__init__(self, rules)`**: No description available.
- **`assign(self, col_name, field_name)`**: Returns (weight, description, needs_review)

#### Global Functions:
- **`assign_weight(col_name, field_name, rules)`**: Functional wrapper for WeightAssigner.

## Module: Mapping_Engine

### File: `Mapping_Engine/test.py`

test.py
───────
Master test script for the Mapping Engine + Business Meaning Layer.
Merged version: User structure + Data-driven classification logic + JSON Numpy Fix.

#### Class: `NpEncoder`
No description available.

Methods:
- **`default(self, obj)`**: No description available.

#### Global Functions:
- **`parse_args()`**: No description available.
- **`load_columns(path)`**: No description available.
- **`main(args)`**: No description available.

### File: `Mapping_Engine/Domain_loader/detection.py`

No module description.

#### Global Functions:
- **`load_yaml(path)`**: No description available.
- **`detect_list_key(detect)`**: No description available.
- **`detect_name_key(detect_names)`**: No description available.
- **`flatten_record(record, skip_keys)`**: No description available.

### File: `Mapping_Engine/Domain_loader/domain_loader.py`

No module description.

#### Global Functions:
- **`build_target_registry(schema_path, keywords_path, metrics_path, rules_path, validations_path, questions_path)`**: Combines the structural schema with business-logic manifests. Extracts deep metadata (purpose, roles, business types) for the AI.
- **`build_keyword_intent_index(keywords_path)`**: Builds a reverse lookup: synonym (lowercase) → canonical intent. Used by the reranker to resolve source column names to their canonical field/entity without any hardcoded field name checks.  Returns:     { "revenue": {canonical_field, canonical_entity, match_priority}, ... }

### File: `Mapping_Engine/Domain_loader/main.py`

No module description.

#### Global Functions:
- **`main()`**: No description available.

### File: `Mapping_Engine/embedding_matcher/embedding.py`

No module description.

#### Class: `Embedding`
No description available.

Methods:
- **`__init__(self, model_name)`**: No description available.
- **`fit_targets(self, target)`**: No description available.
- **`match_column(self, source_column)`**: No description available.

### File: `Mapping_Engine/Reranking/Reranker.py`

No module description.

#### Global Functions:
- **`normalize_type(dtype)`**: No description available.
- **`sigmoid(x)`**: No description available.
- **`dtype_score(source_type, target_type)`**: No description available.
- **`name_score(source_col, target_field)`**: No description available.
- **`table_score(source_table, target_entity)`**: No description available.
- **`_confidence_label(score)`**: No description available.
- **`_build_reason(signals, label)`**: No description available.
- **`apply_dynamic_logic(source_name, source_table, candidate, source_intent)`**: Fully data-driven boost/penalty computation.  All signals come from YAML manifests — no hardcoded field names:   1. Synonym match      — source_name found in candidate["synonyms"] (keywords.yaml)   2. Canonical intent   — source_intent tells us the expected canonical target (keywords.yaml)   3. Rules-based boost  — YAML context_rules already attached to the candidate  Parameters ---------- source_intent : dict | None     Resolved from build_keyword_intent_index(keywords_path) in test.py.     Keys: canonical_field, canonical_entity, match_priority.
- **`rerank_candidates(source_column, candidates, cross_encoder, keyword_intent)`**: No description available.

### File: `Mapping_Engine/Reranking/Resolver.py`

No module description.

#### Global Functions:
- **`resolve_mappings(all_source_results, threshold)`**: Enforces a Global 1:1 Mapping constraint.  Args:     all_source_results: List of { 'source': col_dict, 'candidates': reranked_list }     threshold: Confidence threshold for auto-acceptance  Returns:     List of finalized mapping results.

### File: `Mapping_Engine/text_builder/text_builder.py`

No module description.

#### Global Functions:
- **`build_target_text(target)`**: Constructs a rich semantic string for standard ecommerce fields. Includes business rules and clarifying questions as disambiguation context.
- **`build_source_text(source_column)`**: Constructs a semantic string for the user's source database column.

## Module: Model_Engine

### File: `Model_Engine/config_builder.py`

No module description.

#### Global Functions:
- **`build_configs(base_dir)`**: Step 4.2: The Two-Stage Config Builder (Autonomous Version) No hardcoded feature names. Everything derived from BML Config.

### File: `Model_Engine/refinery.py`

No module description.

#### Global Functions:
- **`run_refinery(data_path, config_path, output_dir)`**: Step 4.1: The Profiler & Atomic Refinery (Autonomous Version) No hardcoded column names. Everything derived from BML Config.

### File: `Model_Engine/run_stage1.py`

No module description.

#### Global Functions:
- **`patched_load()`**: No description available.
- **`run_training(config_file)`**: Step 4.3: Model Training Execution Runs a RecBole training pipeline using the specified YAML config.

### File: `Model_Engine/pipeline/config.py`

pipeline/config.py
==================
PipelineConfig — single source of truth for every tunable constant.

NOTE: You do NOT edit this file to tune the pipeline.
      Open run_stage2.py instead — every knob here is exposed
      in the User Control Panel at the top of that file.

#### Class: `PipelineConfig`
All tunable knobs for the adaptive recommendation pipeline.  Deployment usage ---------------- Default: promotion OFF, personalization dominates. To enable promotion set enable_promotion=True and lower promo_threshold_percentile (e.g. 70 = top-30% strategic score). All other defaults are safe starting points — override per-deployment by mutating the cfg object created in run_stage2_semantic_inference().


### File: `Model_Engine/pipeline/data_utils.py`

pipeline/data_utils.py
=======================
Data loading and preparation utilities:
  - resolve_paths          : derive all file paths from PipelineConfig
  - load_recbole_model     : set up RecBole Config, Dataset, and load model checkpoint
  - extract_item_metadata  : pull item IDs, descriptions, and id→desc mapping
  - compute_popularity_scores : log-normalised global popularity array
  - build_campaign_flags   : promoted/clearance sets + boolean item array

Extracted verbatim from run_stage2.py (lines 448-608).
No logic has been changed.

#### Global Functions:
- **`resolve_paths(cfg, base_dir)`**: Return a dict of all file paths derived from PipelineConfig.
- **`load_intelligence_map(mapping_path)`**: Load rec_config.json (the intelligence / business goals map).
- **`load_recbole_model(config_path, cp_dir, model_name)`**: Set up RecBole Config + Dataset, then load the latest model checkpoint. Returns (config, dataset, model).
- **`extract_item_metadata(dataset, intelligence_map)`**: Extract item IDs, text descriptions, and the id→description mapping. Returns (item_ids, descriptions, item_id_to_desc).
- **`compute_popularity_scores(dataset, item_limit)`**: Compute log-normalised global popularity scores for all items. Returns a float array of shape (item_limit,) in [0, 1].
- **`build_campaign_flags(item_ids, intelligence_map, item_limit)`**: Build promoted/clearance id sets and the is_campaign_item boolean array. Returns (promoted_ids, clearance_ids, is_campaign_item).

### File: `Model_Engine/pipeline/inference.py`

pipeline/inference.py
======================
run_stage2_semantic_inference — the main orchestrator.

This function is now thin: it calls helpers from the other pipeline modules
in sequence. All logic is identical to the original run_stage2.py —
only the structure has changed.

#### Global Functions:
- **`_apply_overrides(cfg)`**: Apply non-None control panel values on top of PipelineConfig defaults.
- **`_apply_business_goals(cfg, intelligence_map, override_n_recs, override_perso_pct, override_curation_pct, override_promo_pct)`**: Parse CLI args + rec_config.json business_goals, then mutate cfg. Extracted verbatim from run_stage2.py (lines 467-513).
- **`_run_batch_inference(model, batch_tokens, dataset, item_rows, item_limit, config)`**: Build the batch interaction tensor and run the model forward pass. Extracted verbatim from run_stage2.py (lines 619-648). Returns (all_user_indices, batch_ai_scores).
- **`run_stage2_semantic_inference(override_n_recs, override_perso_pct, override_curation_pct, override_promo_pct)`**: Main entry point for the Adaptive Semantic Inference Pipeline. All arguments are optional — omit any to use the PipelineConfig default.

### File: `Model_Engine/pipeline/reranker.py`

pipeline/reranker.py
=====================
DiversityReranker — adaptive exposure cap + diversity penalty.

Exposure cap scales with catalog size so it fires correctly regardless of
whether there are 50 or 1,000,000 products.

Extracted verbatim from run_stage2.py (lines 360-403).

#### Class: `DiversityReranker`
Exposure cap scales with catalog size so it fires correctly regardless of whether there are 50 or 1 000 000 products.  Formula: cap = clip(10 / sqrt(catalog_size), 0.04, 0.30)   • 3 000 items  → cap ≈ 0.18   (close to original 0.15, sensible)   • 50 items     → cap = 0.30   (clamped up; tiny catalogue)   • 100 000 items→ cap ≈ 0.032  (clamped down to 0.04)

Methods:
- **`__init__(self, catalog_size, cfg)`**: No description available.
- **`get_penalty(self, item_id, total_users)`**: No description available.
- **`is_over_exposed(self, item_id, total_users, is_campaign)`**: No description available.
- **`record_selection(self, item_id)`**: No description available.

### File: `Model_Engine/pipeline/scoring.py`

pipeline/scoring.py
====================
Scoring utilities:
  - compute_adaptive_weights : dataset-level tier selection (cold/moderate/dense)
  - compute_per_user_scores  : per-user score aggregation extracted from the main loop

Extracted verbatim from run_stage2.py (lines 410-731).
No logic has been changed — code is identical, just wrapped in functions.

#### Global Functions:
- **`compute_adaptive_weights(dataset, cfg)`**: Weights shift based on how data-rich the dataset is (thresholds & tiers all come from PipelineConfig so operators can tune without touching code).  Returns (w_curation, w_strategic, w_personalization, w_popularity).
- **`compute_per_user_scores(ai_scores, user_idx, item_limit, cfg, global_strategic_scores, popularity_scores, is_campaign_item, item_ids, item_themes_map, profiler, reranker, total_users)`**: Compute final scores for all items for a single user.  Extracted verbatim from run_stage2.py (lines 655-731). Returns (final_scores, ai_norm, eligible_indices, personalization_scores).

### File: `Model_Engine/pipeline/selector.py`

pipeline/selector.py
=====================
select_recommendations — quota-based greedy recommendation selector.

Extracted verbatim from run_stage2.py (lines 738-858).
No logic has been changed — code is identical, just wrapped in a function.

#### Global Functions:
- **`select_recommendations(eligible_sorted, final_scores, ai_norm, personalization_scores, global_strategic_scores, popularity_scores, item_ids, item_themes_map, promoted_ids, clearance_ids, is_campaign_item, item_id_to_desc, user_profile, reranker, cfg, total_users)`**: Greedy quota-based selection of top-N recommendations for a single user.  Hierarchy: Promotion > Personalization > Curation. Falls back gracefully when a quota type is exhausted.  Returns a list of recommendation dicts ready for JSON output.

### File: `Model_Engine/pipeline/theme_discovery.py`

pipeline/theme_discovery.py
============================
AutoThemeDiscovery — data-driven replacement for SemanticRuleEngine + YAML rules.
Works on any domain with zero manual configuration.

Extracted verbatim from run_stage2.py (lines 142-285).

#### Class: `AutoThemeDiscovery`
Data-driven replacement for SemanticRuleEngine + YAML rules.  Pipeline:   1. Encode all item descriptions with SentenceTransformer.   2. Cluster embeddings with MiniBatchKMeans (k auto-selected via silhouette).   3. Label each cluster from its most centroid-proximal item.   4. Strategic score = cosine similarity of each item to its own cluster centroid      (normalised globally to [0, 1]).  No hand-written rules, no domain knowledge required.

Methods:
- **`__init__(self, model_name, n_clusters, k_min, k_max, secondary_cluster_threshold, cfg)`**: No description available.
- **`fit(self, descriptions)`**: No description available.
- **`get_item_themes(self, item_idx)`**: No description available.
- **`build_theme_map(self, n_items)`**: No description available.
- **`build_strategic_scores(self, n_items)`**: No description available.

#### Global Functions:
- **`_auto_select_k(embeddings, k_min, k_max, sample_size)`**: Pick the number of clusters that maximises sampled silhouette score.

### File: `Model_Engine/pipeline/user_profiler.py`

pipeline/user_profiler.py
==========================
UserInterestProfiler — builds a per-user interest profile from interaction history.

Optimization: interaction index built once in __init__ (O(I) upfront)
instead of scanning inter_feat per user (was O(U × I) total).

#### Class: `UserInterestProfiler`
No description available.

Methods:
- **`__init__(self, dataset, item_id_to_desc, item_themes_map, cfg)`**: No description available.
- **`get_profile(self, user_idx)`**: No description available.
- **`get_personalization_match(self, user_idx, item_themes)`**: No description available.

