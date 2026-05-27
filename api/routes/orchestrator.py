import os
import uuid
import pandas as pd
from datetime import datetime, timezone
from flask import Blueprint, request, jsonify

import api.state as state
from api.jobs_db import JOBS, save_job_state
from api.pipeline import run_headless_mapping_pipeline
from api.routes.jobs import _ingest_csv, _build_source_columns, _create_job_record
from Business_Meaning.weight_assigner import assign_weight

orchestrator_bp = Blueprint("orchestrator", __name__)


@orchestrator_bp.route("/api/jobs/run-all", methods=["POST"])
def run_all():
    """
    Single-call end-to-end orchestrator:
      1. Creates the job (file upload or local file_path)
      2. Runs the full NLP mapping pipeline SYNCHRONOUSLY (no polling needed)
      3. Deploys rec_config.json automatically
      4. Returns the complete result in a single response

    Use this for scripted/CLI triggers when you don't want to poll.
    """
    if not state.MODEL_READY:
        return jsonify({"error": "Models still loading, try again shortly"}), 503

    print(f"[Orchestrator] Incoming request: Content-Type={request.content_type}", flush=True)
    print(f"[Orchestrator] request.json={request.get_json(silent=True)}", flush=True)
    print(f"[Orchestrator] request.form={request.form.to_dict()}", flush=True)
    print(f"[Orchestrator] request.files={list(request.files.keys())}", flush=True)
    print(f"[Orchestrator] request.args={request.args.to_dict()}", flush=True)

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
        except Exception as e:
            return jsonify({"error": f"Failed to parse uploaded CSV: {str(e)}"}), 400

    table_name = filename.split(".")[0]
    source_columns = _build_source_columns(df, table_name)
    _create_job_record(job_id, filename, saved_path, source_columns)

    run_headless_mapping_pipeline(job_id)

    job = JOBS[job_id]
    if job["status"] == "failed":
        return jsonify({"error": f"Pipeline failed: {job.get('error', 'Unknown error')}"}), 500

    import yaml
    from Business_Meaning.config_writer import ConfigWriter, LockFileExistsError

    try:
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

    except LockFileExistsError as e:
        return jsonify({"error": str(e)}), 423
    except Exception as e:
        return jsonify({"error": f"Deployment failed: {str(e)}"}), 500


    return jsonify({
        "job_id": job_id,
        "status": "deployed",
        "config_path": config_path,
        "filename": filename,
        "summary": job.get("summary"),
        "mappings": job.get("mappings", []),
        "config": config
    })
