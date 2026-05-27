import os
import uuid
import yaml
import threading
import pandas as pd
from datetime import datetime, timezone
from flask import Blueprint, request, jsonify

import api.state as state
from api.jobs_db import JOBS, save_job_state, load_job_state
from api.pipeline import run_headless_mapping_pipeline
from Business_Meaning.weight_assigner import assign_weight

jobs_bp = Blueprint("jobs", __name__)


def _ingest_csv(file_stream, filename) -> tuple[pd.DataFrame, str]:
    """Parse a CSV stream and return (dataframe, saved_path)."""
    job_id = str(uuid.uuid4())
    uploads_dir = os.path.join(state.OUTPUT_DIR, "uploads")
    os.makedirs(uploads_dir, exist_ok=True)
    saved_path = os.path.join(uploads_dir, f"{job_id}_{filename}")
    file_stream.seek(0)
    with open(saved_path, "wb") as buf:
        buf.write(file_stream.read())
    df = pd.read_csv(saved_path, nrows=100)
    return df, saved_path, job_id


def _build_source_columns(df: pd.DataFrame, table_name: str) -> list:
    """Build the source_columns metadata list from a DataFrame."""
    source_columns = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            dtype = "number"
        elif "date" in col.lower() or "time" in col.lower():
            dtype = "datetime"
        else:
            dtype = "string"
        samples = [str(x) for x in df[col].dropna().head(5).tolist()]
        source_columns.append({
            "table": table_name,
            "column": col,
            "data_type": dtype,
            "samples": samples
        })
    return source_columns


def _create_job_record(job_id: str, filename: str, saved_path: str, source_columns: list) -> dict:
    """Build and store a new job record in the in-memory DB."""
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    job = {
        "job_id": job_id,
        "filename": filename,
        "file_path": saved_path,
        "status": "created",
        "progress": {"percentage": 0, "message": "CSV file processed and initialized. Ready to start mapping job."},
        "summary": None,
        "source_columns": source_columns,
        "mappings": [],
        "logs": [
            f"[{ts}] Job created from {filename}.",
            f"[{ts}] Associated CSV file path set to {saved_path}."
        ],
        "created_at": datetime.now(timezone.utc).isoformat()
    }
    JOBS[job_id] = job
    save_job_state(job_id)
    return job


# ---------------------------------------------------------------------------
# POST /api/jobs  — Create a job via file upload OR local file_path JSON
# ---------------------------------------------------------------------------
@jobs_bp.route("/api/jobs", methods=["POST"])
def upload_csv_and_create_job():
    """Accepts a CSV upload OR a local file_path (from JSON, Form, or URL query parameter) and initializes a job."""
    
    print(f"[Jobs API] Incoming request: Content-Type={request.content_type}", flush=True)
    print(f"[Jobs API] request.json={request.get_json(silent=True)}", flush=True)
    print(f"[Jobs API] request.form={request.form.to_dict()}", flush=True)
    print(f"[Jobs API] request.files={list(request.files.keys())}", flush=True)
    print(f"[Jobs API] request.args={request.args.to_dict()}", flush=True)

    file_path = None
    if request.is_json:
        file_path = request.json.get("file_path")
    if not file_path and request.form:
        file_path = request.form.get("file_path")
    if not file_path and request.args:
        file_path = request.args.get("file_path")

    if file_path:
        if not os.path.exists(file_path):
            return jsonify({"error": f"Local file not found at: {file_path}"}), 404
        filename = os.path.basename(file_path)
        saved_path = os.path.abspath(file_path)
        job_id = str(uuid.uuid4())
        try:
            df = pd.read_csv(saved_path, nrows=100)
        except Exception as e:
            return jsonify({"error": f"Failed to parse local CSV: {str(e)}"}), 400

    else:
        if "file" not in request.files:
            return jsonify({"error": "No file part in the request (upload a file or pass 'file_path')"}), 400
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400
        filename = file.filename
        try:
            df, saved_path, job_id = _ingest_csv(file.stream, filename)
            print(f"[Jobs] Physical CSV persisted at: {saved_path}", flush=True)
        except Exception as e:
            return jsonify({"error": f"Failed to parse uploaded CSV: {str(e)}"}), 400

    try:
        table_name = filename.split(".")[0]
        source_columns = _build_source_columns(df, table_name)
        job = _create_job_record(job_id, filename, saved_path, source_columns)
        return jsonify({
            "job_id": job_id,
            "status": "created",
            "columns": [{"name": c["column"], "data_type": c["data_type"], "samples": c["samples"]}
                        for c in source_columns]
        })
    except Exception as e:
        return jsonify({"error": f"Failed to initialize job: {str(e)}"}), 500


# ---------------------------------------------------------------------------
# POST /api/jobs/<job_id>/start  — Launch background mapping thread
# ---------------------------------------------------------------------------
@jobs_bp.route("/api/jobs/<job_id>/start", methods=["POST"])
def start_mapping_job(job_id):
    """Launches the non-interactive mapping pipeline as a background thread."""
    if not state.MODEL_READY:
        return jsonify({"error": "Models still loading, try again shortly"}), 503
    if job_id not in JOBS and not load_job_state(job_id):
        return jsonify({"error": "Job not found"}), 404

    job = JOBS[job_id]
    if job["status"] == "running":
        return jsonify({"error": "Job is already running"}), 409

    t = threading.Thread(target=run_headless_mapping_pipeline, args=(job_id,), daemon=True)
    t.start()
    return jsonify({"status": "running", "message": "Job launched in background thread.", "job_id": job_id})


# ---------------------------------------------------------------------------
# GET /api/jobs/<job_id>  — Poll job status and results
# ---------------------------------------------------------------------------
@jobs_bp.route("/api/jobs/<job_id>", methods=["GET"])
def get_job_status(job_id):
    """Returns current status, progress, mappings, and logs for a job."""
    if job_id not in JOBS and not load_job_state(job_id):
        return jsonify({"error": "Job not found"}), 404
    job = JOBS[job_id]
    return jsonify({
        "job_id": job_id,
        "filename": job.get("filename"),
        "file_path": job.get("file_path"),
        "status": job["status"],
        "progress": job.get("progress", {}),
        "summary": job.get("summary"),
        "mappings": job.get("mappings", []),
        "logs": job.get("logs", []),
        "created_at": job.get("created_at")
    })


# ---------------------------------------------------------------------------
# POST /api/jobs/<job_id>/update  — Apply manual overrides to mappings
# ---------------------------------------------------------------------------
@jobs_bp.route("/api/jobs/<job_id>/update", methods=["POST"])
def update_job_mappings(job_id):
    """Applies manual mapping overrides to a completed job."""
    if job_id not in JOBS and not load_job_state(job_id):
        return jsonify({"error": "Job not found"}), 404

    data = request.get_json()
    if not data or "mappings" not in data:
        return jsonify({"error": "Request body must contain 'mappings' array"}), 400

    job = JOBS[job_id]
    for update in data["mappings"]:
        col_name = update.get("source_column")
        for mapping in job["mappings"]:
            if mapping["source_column"] == col_name:
                mapping["detected_entity_node"] = update.get("detected_entity_node")
                mapping["mapped_canonical_attribute"] = update.get("mapped_canonical_attribute")
                mapping["role_designation"] = update.get("role_designation")
                mapping["alignment_status"] = update.get("alignment_status", "Mapped")
                mapping["confidence_score"] = "Manual"

    # Recalculate summary
    job["summary"] = {
        "total_columns": len(job["mappings"]),
        "mapped_matches": sum(1 for m in job["mappings"] if m["alignment_status"] == "Mapped"),
        "ignored_unmapped": sum(1 for m in job["mappings"] if m["alignment_status"] == "Ignored" or m["mapped_canonical_attribute"] == "Unmapped"),
        "verification_warnings": sum(1 for m in job["mappings"] if m["alignment_status"] == "Needs Review")
    }

    save_job_state(job_id)
    return jsonify({"status": "updated", "job_id": job_id})


# ---------------------------------------------------------------------------
# POST /api/jobs/<job_id>/deploy  — Write final rec_config.json
# ---------------------------------------------------------------------------
@jobs_bp.route("/api/jobs/<job_id>/deploy", methods=["POST"])
def deploy_job_mappings(job_id):
    """Builds, locks, and writes the finalized configuration to output/rec_config.json."""
    if job_id not in JOBS and not load_job_state(job_id):
        return jsonify({"error": "Job not found"}), 404

    job = JOBS[job_id]
    if job["status"] not in ("completed", "deployed") and not job.get("mappings"):
        return jsonify({"error": "Cannot deploy: job mappings not calculated"}), 400

    try:
        from Business_Meaning.config_writer import ConfigWriter, LockFileExistsError
        writer = ConfigWriter(state.OUTPUT_DIR)
        writer.check_lock()

        role_buckets = {"identity": [], "content_features": [], "filters": [], "ignore": []}
        interaction_signals = {}
        provenance = {}

        role_reverse_map = {
            "Identity": "identity",
            "Content Item": "content_features",
            "Filter Node": "filters",
            "Signal Action": "interaction_signals",
            "Ignored": "ignore"
        }

        with open(state.RULES_MANIFEST, "r", encoding="utf-8") as rf:
            classifier_rules = yaml.safe_load(rf) or {}

        for mapping in job["mappings"]:
            col_name = mapping["source_column"]
            entity = mapping["detected_entity_node"]
            field = mapping["mapped_canonical_attribute"]
            role = role_reverse_map.get(mapping["role_designation"], "content_features")

            status_map = {
                "Mapped": "AUTO_ACCEPTED",
                "Needs Review": "NEEDS_REVIEW",
                "Ignored": "CUSTOM_FIELD"
            }
            status = status_map.get(mapping["alignment_status"], "AUTO_ACCEPTED")

            if role == "interaction_signals":
                signal_weight, _, _ = assign_weight(col_name, field, rules=classifier_rules)
                if signal_weight is None:
                    signal_weight = float(classifier_rules.get("config", {}).get("default_weight", 0.1))
                interaction_signals[col_name] = signal_weight
            elif role in role_buckets:
                role_buckets[role].append(col_name)

            conf_raw = mapping["confidence_score"]
            provenance[col_name] = {
                "entity": entity,
                "field": field,
                "confidence": 1.0 if conf_raw == "Manual" else round(float(conf_raw) / 100, 4),
                "status": status
            }

        config = writer.build_config(
            role_buckets=role_buckets,
            interaction_signals=interaction_signals,
            ambiguous=[],
            provenance=provenance
        )
        config_path = writer.write(config)

        job["status"] = "deployed"
        save_job_state(job_id)

        return jsonify({
            "status": "deployed",
            "config_path": config_path,
            "job_id": job_id,
            "config": config
        })

    except Exception as e:
        return jsonify({"error": f"Deployment failed: {str(e)}"}), 500
