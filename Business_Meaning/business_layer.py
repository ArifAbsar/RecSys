"""
business_layer.py
─────────────────
BusinessMeaningLayer — the orchestrator for Step 2.
Fully data-driven using the 'config' section of classifier_rules.yaml.
"""

import os
import yaml
from typing import Optional, Dict, List


from .role_classifier    import classify_role
from .weight_assigner    import assign_weight
from .ambiguity_detector import is_ambiguous
from .question_flow      import QuestionFlow
from .config_writer      import ConfigWriter, LockFileExistsError
from .semantic_classifier import SemanticRoleClassifier


class BusinessMeaningLayer:
    """
    Orchestrates the classification, weight assignment, and ambiguity resolution.
    """

    def __init__(
        self,
        output_dir:     str,
        questions_path: Optional[str] = None,
        auto_mode:      bool = False,
        auto_answers:   Optional[dict] = None,
        rules_path:     Optional[str] = None,
    ):
        self._rules = {}
        rules_manifest = os.path.abspath(rules_path) if rules_path else ""
        if not rules_manifest or not os.path.exists(rules_manifest):
            raise FileNotFoundError(f"CRITICAL: Rules manifest not found at {rules_manifest}")

        with open(rules_manifest, 'r', encoding='utf-8') as f:
            self._rules = yaml.safe_load(f) or {}
            
        self._config = self._rules.get("config", {})
        self._roles = self._config.get("roles", ["identity", "content_features", "interaction_signals", "filters", "ignore"])
        self._unknown_sentinel = self._config.get("unknown_role", "unknown")
        self._weighted_role = self._config.get("weighted_role", "interaction_signals")

        self._writer = ConfigWriter(output_dir)
        self._qflow  = QuestionFlow(
            config=self._config,
            questions_path=questions_path,
            auto_mode=auto_mode,
            auto_answers=auto_answers,
        )
        
        anchors = self._rules.get("semantic_anchors", {})
        self._semantic_model = SemanticRoleClassifier(anchors=anchors)

    def run(self, resolved_mappings: list[dict], save_output: bool = True) -> dict:
        """Processes mappings and returns the rec_config dict."""
        try:
            self._writer.check_lock()
        except LockFileExistsError:
            raise

        role_buckets: Dict[str, List[str]] = {role: [] for role in self._roles}
        interaction_signals: Dict[str, float] = {}  
        
        ambiguous_records: list[dict] = []
        provenance: dict[str, dict] = {}

        for rec in resolved_mappings:
            col_name  = rec.get("source_column", "unknown")
            map_conf  = float(rec.get("confidence", 0.0))
            status    = rec.get("status", "NEEDS_REVIEW")

            _rec_map  = rec.get("recommended_mapping", {})
            entity    = _rec_map.get("entity", "CUSTOM")
            field     = _rec_map.get("field", col_name)

            role, role_conf = classify_role(
                rec, 
                semantic_model=self._semantic_model,
                rules=self._rules
            )

            weight_needs_review = False
            signal_weight: Optional[float] = None
            if role == self._weighted_role:
                signal_weight, _, weight_needs_review = assign_weight(
                    col_name, field, rules=self._rules
                )

            flagged, reasons = is_ambiguous(
                col_name=col_name,
                role=role,
                role_confidence=role_conf,
                mapping_confidence=map_conf,
                mapping_status=status,
                config=self._config,
                weight_needs_review=weight_needs_review,
            )

            if flagged:
                resolved_role = self._qflow.resolve(
                    col_name=col_name,
                    current_role=role,
                    reasons=reasons,
                    entity=entity,
                    field=field,
                )

                if resolved_role == self._unknown_sentinel:
                    ambiguous_records.append({
                        "source_column": col_name,
                        "entity": entity,
                        "field": field,
                        "reasons": reasons,
                    })
                    provenance[col_name] = _make_provenance(rec, map_conf, status)
                    continue
                role = resolved_role

            if role == self._weighted_role:
                interaction_signals[col_name] = signal_weight if signal_weight is not None else float(self._config.get("default_weight", 0.1))
            elif role in role_buckets:
                role_buckets[role].append(col_name)
            else:
                if "ignore" in role_buckets:
                    role_buckets["ignore"].append(col_name)

            provenance[col_name] = _make_provenance(rec, map_conf, status)

        config = self._writer.build_config(
            role_buckets=role_buckets,
            interaction_signals=interaction_signals,
            ambiguous=ambiguous_records,
            provenance=provenance,
            weighted_role=self._weighted_role
        )

        config_path = self._writer.write(config) if save_output else None

        _print_summary(
            config_path=config_path,
            role_buckets=role_buckets,
            interaction_signals=interaction_signals,
            ambiguous=ambiguous_records,
            weighted_role=self._weighted_role
        )

        return config


def _make_provenance(rec: dict, confidence: float, status: str) -> dict:
    _rec_map = rec.get("recommended_mapping", {})
    return {
        "entity":     _rec_map.get("entity", "CUSTOM"),
        "field":      _rec_map.get("field", ""),
        "confidence": round(confidence, 4),
        "status":     status,
    }


def _print_summary(
    config_path: Optional[str],
    role_buckets: Dict[str, List[str]],
    interaction_signals: Dict[str, float],
    ambiguous: list[dict],
    weighted_role: str
) -> None:
    W = 72
    print("\n" + "=" * W)
    print(f"  {'BUSINESS MEANING LAYER — Result Summary':^{W-2}}")
    print("=" * W)

    for role, items in role_buckets.items():
        if items and role != weighted_role:
            print(f"\n  [{role}]")
            for col in items:
                print(f"    • {col}")

    if interaction_signals:
        print(f"\n  [{weighted_role}]")
        for col, w in sorted(interaction_signals.items(), key=lambda x: -x[1]):
            bar = "█" * int(w * 10) if w > 0 else "░"
            print(f"    • {col:<30} weight={w:.2f}  {bar}")

    if ambiguous:
        print(f"\n  [STILL AMBIGUOUS — not written to config]")
        for a in ambiguous:
            print(f"    • {a['source_column']}")
            for r in a["reasons"]:
                print(f"        - {r}")

    if config_path:
        print(f"\n Written → {config_path}")
    print("=" * W + "\n")