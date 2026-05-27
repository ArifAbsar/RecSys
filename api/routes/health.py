from flask import Blueprint, jsonify
import api.state as state

health_bp = Blueprint("health", __name__)

@health_bp.route("/api/health", methods=["GET"])
def health_check():
    """Instant readiness probe — returns current loading step while warming up."""
    if state.MODEL_READY:
        return jsonify({"status": "ok", "registry_size": len(state.TARGET_REGISTRY)})
    return jsonify({"status": "loading", "step": state.LOADING_STEP}), 503

@health_bp.route("/api/schema-details", methods=["GET"])
def get_schema_details():
    """Returns available entities and fields in the target canonical schema for UI dropdowns."""
    if not state.MODEL_READY:
        return jsonify({"error": "Models still loading"}), 503
    options = {}
    for target in state.TARGET_REGISTRY:
        entity_name = target.get("entity", "CUSTOM")
        field_name = target.get("field", "")
        if entity_name not in options:
            options[entity_name] = []
        if field_name and field_name not in options[entity_name]:
            options[entity_name].append(field_name)
    # Add ignore option
    options["IGNORED"] = ["Unmapped"]
    return jsonify(options)
