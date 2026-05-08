"""
weight_assigner.py
──────────────────
Data-driven weight assignment for interaction signals.
"""

from __future__ import annotations
from typing import Tuple, Dict, List, Optional


class WeightAssigner:
    """
    Assigns weights to interaction signals using YAML rules.
    """
    def __init__(self, rules: dict):
        self._config = rules.get("config", {})
        self._iw_config = rules.get("interaction_signal_weights", {})
        
        self.default_weight = self._iw_config.get(
            "default_weight", 
            self._config.get("default_weight", 0.5)
        )
        self.rules = self._iw_config.get("rules", [])

    def assign(self, col_name: str, field_name: str) -> Tuple[float, str, bool]:
        """
        Returns (weight, description, needs_review)
        """
        haystack = f"{col_name} {field_name}".lower()

        for rule in self.rules:
            keywords = [str(k).lower() for k in rule.get("keywords", [])]
            weight = rule.get("weight", self.default_weight)
            desc = rule.get("description", "matched keyword")

            if any(kw in haystack for kw in keywords):
                return float(weight), desc, False

        return float(self.default_weight), "No matching signal rule found", True


def assign_weight(
    col_name: str, 
    field_name: str, 
    rules: Optional[dict] = None
) -> Tuple[float, str, bool]:
    """Functional wrapper for WeightAssigner."""
    assigner = WeightAssigner(rules or {})
    return assigner.assign(col_name, field_name)
