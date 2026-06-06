import os
import uuid
import yaml
import pandas as pd
from io import BytesIO
from datetime import datetime, timezone
from flask import Blueprint, request, jsonify

import boto3
import api.state as state
from api.jobs import get_store
from Business_Meaning.weight_assigner import assign_weight

jobs_bp = Blueprint("jobs", __name__)

GLUE_JOB_NAME = os.environ.get("GLUE_JOB_NAME", "recsys-mapping")
S3_BUCKET = os.environ.get("S3_BUCKET")

s3_client = boto3.client("s3")
glue_client = boto3.client("glue")


def _build_source_columns(df: pd.DataFrame, table_name: str) -> list:
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


# ---------------------------------------------------------------------------
# POST /api/jobs — upload CSV to S3, create job in DynamoDB
# ---------------------------------------------------------------------------
@jobs_bp.route("/api/jobs", methods=["POST"])
def create_job():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    filename = file.filename
    job_id = str(uuid.uuid4())

    try:
        raw = file.read()
        s3_key = f"uploads/{job_id}/{filename}"
        s3_client.put_object(Bucket=S3_BUCKET, Key=s3_key, Body=raw)
        s3_path = f"s3://{S3_BUCKET}/{s3_key}"

        df = pd.read_csv(BytesIO(raw), nrows=100)
    except Exception as e:
        return jsonify({"error": f"Failed to upload or parse CSV: {str(e)}"}), 400

    table_name = filename.split(".")[0]
    source_columns = _build_source_columns(df, table_name)

    store = get_store()
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    job = {
        "job_id": job_id,
        "filename": filename,
        "file_path": s3_path,
        "status": "created",
        "progress": {"percentage": 0, "message": "File uploaded to S3. Ready to start."},
        "summary": None,
        "source_columns": source_columns,
        "mappings": [],
        "logs": [f"[{ts}] Job created. File uploaded to {s3_path}."],
        "created_at": datetime.now(timezone.utc).isoformat()
    }
    store.put(job_id, job)

    return jsonify({
        "job_id": job_id,
        "status": "created",
        "columns": [{"name": c["column"], "data_type": c["data_type"], "samples": c["samples"]}
                    for c in source_columns]
    })


# ---------------------------------------------------------------------------
# POST /api/jobs/<job_id>/start — trigger Glue job
# ---------------------------------------------------------------------------
@jobs_bp.route("/api/jobs/<job_id>/start", methods=["POST"])
def start_job(job_id):
    store = get_store()
    job = store.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    if job["status"] == "running":
        return jsonify({"error": "Job is already running"}), 409

    try:
        run = glue_client.start_job_run(
            JobName=GLUE_JOB_NAME,
            Arguments={"--job_id": job_id}
        )
        store.update(job_id, {
            "status": "running",
            "glue_run_id": run["JobRunId"],
            "progress": {"percentage": 10, "message": "Glue job triggered."}
        })
        return jsonify({"status": "running", "job_id": job_id, "glue_run_id": run["JobRunId"]})
    except Exception as e:
        return jsonify({"error": f"Failed to start Glue job: {str(e)}"}), 500


# ---------------------------------------------------------------------------
# GET /api/jobs/<job_id> — poll status from DynamoDB, sync Glue if running
# ---------------------------------------------------------------------------
@jobs_bp.route("/api/jobs/<job_id>", methods=["GET"])
def get_job(job_id):
    store = get_store()
    job = store.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404

    # Sync latest state from Glue while the job is still running
    if job["status"] == "running" and job.get("glue_run_id"):
        try:
            glue_run = glue_client.get_job_run(
                JobName=GLUE_JOB_NAME,
                RunId=job["glue_run_id"]
            )["JobRun"]
            glue_state = glue_run["JobRunState"]  # STARTING, RUNNING, SUCCEEDED, FAILED, STOPPED

            if glue_state == "FAILED":
                error_msg = glue_run.get("ErrorMessage", "Unknown Glue error")
                store.update(job_id, {"status": "failed", "error": error_msg})
                job["status"] = "failed"

            elif glue_state == "STOPPED":
                store.update(job_id, {"status": "failed", "error": "Glue job was stopped"})
                job["status"] = "failed"

        except Exception as e:
            store.append_log(job_id, f"[WARN] Could not fetch Glue status: {str(e)}")

    return jsonify({
        "job_id": job_id,
        "filename": job.get("filename"),
        "status": job["status"],
        "progress": job.get("progress", {}),
        "summary": job.get("summary"),
        "mappings": job.get("mappings", []),
        "logs": job.get("logs", []),
        "created_at": job.get("created_at")
    })


# ---------------------------------------------------------------------------
# POST /api/jobs/<job_id>/update — manual mapping overrides
# ---------------------------------------------------------------------------
@jobs_bp.route("/api/jobs/<job_id>/update", methods=["POST"])
def update_job(job_id):
    store = get_store()
    job = store.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404

    data = request.get_json()
    if not data or "mappings" not in data:
        return jsonify({"error": "Request body must contain 'mappings'"}), 400

    for update in data["mappings"]:
        col_name = update.get("source_column")
        for mapping in job["mappings"]:
            if mapping["source_column"] == col_name:
                mapping["detected_entity_node"] = update.get("detected_entity_node")
                mapping["mapped_canonical_attribute"] = update.get("mapped_canonical_attribute")
                mapping["role_designation"] = update.get("role_designation")
                mapping["alignment_status"] = update.get("alignment_status", "Mapped")
                mapping["confidence_score"] = "Manual"

    job["summary"] = {
        "total_columns": len(job["mappings"]),
        "mapped_matches": sum(1 for m in job["mappings"] if m["alignment_status"] == "Mapped"),
        "ignored_unmapped": sum(1 for m in job["mappings"] if m["alignment_status"] == "Ignored"),
        "verification_warnings": sum(1 for m in job["mappings"] if m["alignment_status"] == "Needs Review")
    }

    store.update(job_id, job)
    return jsonify({"status": "updated", "job_id": job_id})


# ---------------------------------------------------------------------------
# POST /api/jobs/<job_id>/deploy — write final rec_config.json
# ---------------------------------------------------------------------------
@jobs_bp.route("/api/jobs/<job_id>/deploy", methods=["POST"])
def deploy_job(job_id):
    store = get_store()
    job = store.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    if not job.get("mappings"):
        return jsonify({"error": "No mappings to deploy"}), 400

    try:
        from Business_Meaning.config_writer import ConfigWriter
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
                "status": mapping["alignment_status"]
            }

        config = writer.build_config(
            role_buckets=role_buckets,
            interaction_signals=interaction_signals,
            ambiguous=[],
            provenance=provenance
        )
        config_path = writer.write(config)
        store.update(job_id, {"status": "deployed"})

        return jsonify({"status": "deployed", "config_path": config_path, "job_id": job_id, "config": config})

    except Exception as e:
        return jsonify({"error": f"Deployment failed: {str(e)}"}), 500
