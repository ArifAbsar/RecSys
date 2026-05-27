import sys
import contextlib
from io import StringIO
from datetime import datetime, timezone

from Mapping_Engine.Reranking.Reranker import rerank_candidates
from Mapping_Engine.Reranking.Resolver import resolve_mappings
from Business_Meaning.business_layer import BusinessMeaningLayer
from Business_Meaning.question_flow import _load_questions, _find_yaml_question
from Business_Meaning.role_classifier import classify_role
from Business_Meaning.weight_assigner import assign_weight
from Business_Meaning.ambiguity_detector import is_ambiguous

from api.jobs_db import JOBS, save_job_state
import api.state as state

@contextlib.contextmanager
def capture_stdout(log_list):
    """Intercepts print calls and appends them to a log list in real-time."""
    class LogStream(StringIO):
        def write(self, s):
            if s.strip():
                timestamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
                log_list.append(f"[{timestamp}] {s.strip()}")
            super().write(s)
    
    stream = LogStream()
    old_stdout = sys.stdout
    sys.stdout = stream
    try:
        yield
    finally:
        sys.stdout = old_stdout


def run_headless_mapping_pipeline(job_id):
    """Runs the full schema mapping pipeline for a specific job state."""
    job = JOBS[job_id]
    job["status"] = "running"
    job["progress"] = {"percentage": 10, "message": "Initializing headless schema mapping job..."}
    save_job_state(job_id)

    log_capture = []
    
    try:
        with capture_stdout(log_capture):
            print(f"Starting headless schema integration mapping for job {job_id}")
            source_columns = job["source_columns"]

            job["progress"] = {"percentage": 30, "message": "Analyzing schema synonyms dictionary entries..."}
            save_job_state(job_id)

            all_source_results = []
            num_cols = len(source_columns)
            
            for i, col in enumerate(source_columns):

                percentage = int(35 + (i / num_cols) * 40)
                message = f"Computing vector embeddings & reranking column {i+1} of {num_cols}: {col['column']}..."
                
                job["progress"] = {"percentage": percentage, "message": message}
                save_job_state(job_id)
                
                print(f"Retrieving semantic candidates for column: {col['column']}")
                raw_cands = state.MATCHER.match_column(col)
                refined = rerank_candidates(
                    col, raw_cands, state.CROSS_ENCODER, keyword_intent=state.KEYWORD_INTENT
                )
                all_source_results.append({"source": col, "candidates": refined})

            job["progress"] = {"percentage": 80, "message": "Solving global 1:1 schema mapping constraints..."}
            save_job_state(job_id)
            
            threshold = 0.60
            resolved_mappings = resolve_mappings(all_source_results, threshold)
            print(f"Global 1:1 resolver completed. Resolved {len(resolved_mappings)} mapping(s).")

            job["progress"] = {"percentage": 90, "message": "Classifying business meaning functional roles..."}
            save_job_state(job_id)
            
            layer = BusinessMeaningLayer(
                output_dir=state.OUTPUT_DIR,
                questions_path=state.QUESTIONS_PATH,
                rules_path=state.RULES_MANIFEST,
                auto_mode=True,
                auto_answers={},
            )
            
            final_mappings = []
            yaml_questions = _load_questions(state.QUESTIONS_PATH)

            for rec in resolved_mappings:
                col_name = rec.get("source_column")
                map_conf = float(rec.get("confidence", 0.0))
                status = rec.get("status", "NEEDS_REVIEW")

                _rec_map = rec.get("recommended_mapping", {})
                entity = _rec_map.get("entity", "CUSTOM")
                field = _rec_map.get("field", col_name)

                role, role_conf = classify_role(
                    rec, semantic_model=layer._semantic_model, rules=layer._rules
                )
                weight_needs_review = False
                if role == layer._weighted_role:
                    _, _, weight_needs_review = assign_weight(
                        col_name, field, rules=layer._rules
                    )

                flagged, reasons = is_ambiguous(
                    col_name=col_name,
                    role=role,
                    role_confidence=role_conf,
                    mapping_confidence=map_conf,
                    mapping_status=status,
                    config=layer._config,
                    weight_needs_review=weight_needs_review,
                )

                yaml_q = _find_yaml_question(yaml_questions, entity, field)

                if map_conf < state.ENTITY_THRESHOLD or map_conf < state.FIELD_THRESHOLD:
                    flagged = True
                    conf_msg = f"Confidence score {round(map_conf * 100)}% falls below the questions.yaml thresholds."
                    if conf_msg not in reasons:
                        reasons.append(conf_msg)

                alignment_status = "Mapped"
                if flagged or status == "NEEDS_REVIEW":
                    alignment_status = "Needs Review"
                
                if map_conf >= 0.80:
                    alignment_status = "Mapped"
                    flagged = False
                    reasons = [r for r in reasons if "falls below" not in r]

                if role == "ignore" or entity == "IGNORED":
                    alignment_status = "Ignored"

                role_display_map = {
                    "identity": "Identity",
                    "content_features": "Content Item",
                    "filters": "Filter Node",
                    "interaction_signals": "Signal Action",
                    "ignore": "Ignored"
                }

                col_samples = next((c["samples"] for c in source_columns if c["column"] == col_name), [])
                sample_row = col_samples[0] if col_samples else ""

                final_mappings.append({
                    "source_column": col_name,
                    "sample_record": sample_row,
                    "detected_entity_node": entity,
                    "mapped_canonical_attribute": field,
                    "role_designation": role_display_map.get(role, "Content Item"),
                    "confidence_score": round(map_conf * 100, 1),
                    "alignment_status": alignment_status,
                    "review_metadata": {
                        "flagged": flagged,
                        "reasons": reasons,
                        "business_context_prompt": yaml_q or ""
                    }
                })

            total_columns = len(final_mappings)
            mapped_matches = sum(1 for m in final_mappings if m["alignment_status"] == "Mapped")
            ignored_unmapped = sum(1 for m in final_mappings if m["alignment_status"] == "Ignored" or m["mapped_canonical_attribute"] == "Unmapped")
            verification_warnings = sum(1 for m in final_mappings if m["alignment_status"] == "Needs Review")

            job["summary"] = {
                "total_columns": total_columns,
                "mapped_matches": mapped_matches,
                "ignored_unmapped": ignored_unmapped,
                "verification_warnings": verification_warnings
            }

            job["mappings"] = final_mappings
            job["status"] = "completed"
            job["progress"] = {"percentage": 100, "message": "Headless schema mapping completed successfully."}
            print("Mapping job successfully completed headlessly.")

    except Exception as e:
        log_capture.append(f"[ERROR] Pipeline run failed: {str(e)}")
        job["status"] = "failed"
        job["progress"] = {"percentage": 100, "message": f"Pipeline failed: {str(e)}"}
        job["error"] = str(e)
    
    job["logs"].extend(log_capture)
    save_job_state(job_id)
