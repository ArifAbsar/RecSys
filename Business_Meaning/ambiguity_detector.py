"""
ambiguity_detector.py
─────────────────────
Decides whether a classified column needs human review.
Fully data-driven using the 'config' section of classifier_rules.yaml.
"""

from __future__ import annotations

def is_ambiguous(
    col_name: str,
    role: str,
    role_confidence: float,
    mapping_confidence: float,
    mapping_status: str,
    config: dict,
    weight_needs_review: bool = False,
) -> tuple[bool, list[str]]:
    """
    Determine if a column classification needs human confirmation.
    """
    reasons: list[str] = []
    
    t_config = config.get("thresholds", {})
    role_threshold = float(t_config.get("role_confidence", 0.80))
    map_threshold  = float(t_config.get("mapping_confidence", 0.75))
    unknown_sentinel = config.get("unknown_role", "unknown")

    if role == unknown_sentinel:
        reasons.append("The system couldn't determine the purpose of this column.")

    if role_confidence < role_threshold:
        reasons.append(
            f"The system is only {int(role_confidence * 100)}% sure about how to use this column "
            f"(requires {int(role_threshold * 100)}% for automatic choice)."
        )

    # Normalize mapping status to lowercase for comparison
    if mapping_status.lower() == "needs_review" and mapping_confidence < map_threshold:
        reasons.append(
            f"The mapping to the schema is slightly uncertain ({int(mapping_confidence * 100)}% confidence)."
        )

    return bool(reasons), reasons
