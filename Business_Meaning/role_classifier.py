"""
role_classifier.py
─────────────────
Advanced Multi-Factor Scoring Engine.
Data-driven using classifier_rules.yaml.
"""

from __future__ import annotations
import sys
from typing import Tuple, Optional, Dict, List


class RoleClassifier:
    def __init__(self, rules: dict, semantic_model=None):
        self._rules = rules
        self._config = rules.get("config", {})
        
        self._overrides = rules.get("overrides", [])
        self._scoring_rules = rules.get("scoring_rules", [])
        self._schema_role_map = rules.get("schema_role_map", {})
        
        self._unknown_sentinel = self._config.get("unknown_role", "unknown")
        self._default_weight = float(self._config.get("default_weight", 0.5))
        
        self._semantic_model = semantic_model

    def classify(self, mapping: dict) -> Tuple[str, float]:

        source_col = str(mapping.get("source_column", "unknown")).lower()
        rec        = mapping.get("recommended_mapping", {})
        entity     = str(rec.get("entity", "CUSTOM")).upper()
        field      = str(rec.get("field", source_col)).lower()
        schema_sr  = str(rec.get("semantic_role", "")).lower()

        sys.stderr.write(f"\n[BML_DEBUG] Column: {source_col}\n")
        sys.stderr.write(f"            Entity: {entity}, Field: {field}, SchemaRole: {schema_sr}\n")

        for rule in self._overrides:
            if self._check_condition(rule, entity, field, schema_sr, source_col):
                role = rule.get("role", self._unknown_sentinel)
                sys.stderr.write(f"            -> MATCH: Override -> {role}\n")
                return role, 1.0
        if schema_sr and schema_sr in self._schema_role_map:
            role = self._schema_role_map[schema_sr]
            sys.stderr.write(f"            -> HIT: Schema Map ({schema_sr} -> {role})\n")
            return role, 0.90

        best_role:   str   = self._unknown_sentinel
        best_weight: float = -1.0

        for rule in self._scoring_rules:
            candidate_role   = rule.get("role", self._unknown_sentinel)
            candidate_weight = float(rule.get("weight", self._default_weight))
            
            if self._check_condition(rule, entity, field, schema_sr, source_col):
                if candidate_weight > best_weight:
                    best_role   = candidate_role
                    best_weight = candidate_weight
                    sys.stderr.write(f"            -> MATCH: Rule ({candidate_role} w={candidate_weight})\n")

        if best_weight >= 0:
            return best_role, best_weight

        if self._semantic_model:
            ai_input = f"{field} {source_col} entity:{entity}".strip()
            sys.stderr.write(f"            -> FALLBACK: Running AI on '{ai_input}'...\n")
            sem_role, sem_conf = self._semantic_model.classify(ai_input)
            sys.stderr.write(f"            -> AI result: {sem_role} ({sem_conf:.4f})\n")
            return sem_role, sem_conf

        sys.stderr.write(f"            -> ERROR: No classification found.\n")
        return self._unknown_sentinel, 0.0

    def _check_condition(self, rule: dict, entity: str, field: str, schema_sr: str, source_col: str) -> bool:
        """Centralized condition evaluator for all rule types."""
        cond = rule.get("condition", rule)
        
        if "entity" in cond:
            if entity != str(cond["entity"]).upper():
                return False
        if "field_contains" in cond:
            keywords = [str(k).lower() for k in cond["field_contains"]]
            if not any(kw in field for kw in keywords):
                return False
                
        if "if_field_matches" in cond or "field_matches" in cond:
            matches = cond.get("if_field_matches", cond.get("field_matches", []))
            if field not in [str(m).lower() for m in matches]:
                return False
        if "if_field_starts" in cond:
            prefixes = [str(p).lower() for p in cond["if_field_starts"]]
            if not any(field.startswith(p) for p in prefixes):
                return False
        if "if_schema_role" in cond:
            if schema_sr != str(cond["if_schema_role"]).lower():
                return False
        if "semantic_similarity" in cond:
            if not self._semantic_model:
                return False
            phrases = cond["semantic_similarity"]
            for phrase in phrases:
                _, sim = self._semantic_model.classify(f"{field} {source_col}", [phrase])
                if sim > 0.70:
                    return True
            return False

        return True


def classify_role(mapping: dict, semantic_model=None, rules: dict = None) -> Tuple[str, float]:
    classifier = RoleClassifier(rules or {}, semantic_model)
    return classifier.classify(mapping)