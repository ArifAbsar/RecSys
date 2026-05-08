"""
question_flow.py
────────────────
Data-driven interactive question flow for ambiguous columns.
Reads roles and config from classifier_rules.yaml.
"""

from __future__ import annotations

import os
import yaml
from typing import Optional


def _load_questions(questions_path: str) -> list[dict]:
    """Load all questions from questions.yaml into a flat list."""
    if not questions_path or not os.path.exists(questions_path):
        return []
    try:
        with open(questions_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        questions: list[dict] = []
        for group in data.get("question_groups", []):
            questions.extend(group.get("questions", []))
        return questions
    except Exception:
        return []


def _find_yaml_question(
    questions: list[dict],
    entity: str,
    field: str,
) -> Optional[str]:
    """Search questions.yaml for a matching question."""
    target_entity = str(entity).lower()
    target_field  = str(field).lower()

    for query in questions:
        maps_to = [str(m).lower() for m in query.get("maps_to", [])]
        if target_entity in maps_to or target_field in maps_to:
            return query.get("question", "")
    return None


def _ask_role(
    col_name: str,
    current_role: str,
    reasons: list[str],
    yaml_question: Optional[str],
    auto_answers: Optional[dict],
    valid_roles: list[str],
) -> str:
    """Present a role-clarification prompt."""

    print(f"\n{'━'*70}")
    print(f"Column Review: {col_name!r}")
    print(f"{'─'*70}")
    print(f"The system flagged this column because:")
    for r in reasons:
        print(f"• {r}")
    
    print(f"\nAuto-detected suggestion: {current_role!r}")

    if yaml_question:
        print(f"\nBusiness Context:")
        print(f"{yaml_question}")

    print(f"\nWhat does this column represent?")
    for i, role in enumerate(valid_roles, 1):
        print(f"    [{i}] {role}")
    print(f"    [s] Skip (Keep current suggestion)")
    print(f"{'━'*70}")

    if auto_answers is not None:
        answer = str(auto_answers.get(col_name, "s")).strip().lower()
        print(f"  [AUTO] answer = {answer!r}")
    else:
        prompt = f"  Selection [1-{len(valid_roles)} or s]: "
        answer = input(prompt).strip().lower()

    if answer in ("s", ""):
        return current_role

    if answer.isdigit():
        idx = int(answer) - 1
        if 0 <= idx < len(valid_roles):
            return valid_roles[idx]

    if answer in valid_roles:
        return answer

    print(f"  Invalid selection — keeping {current_role!r}")
    return current_role


class QuestionFlow:
    """Orchestrates the data-driven ambiguity resolution Q&A."""

    def __init__(
        self,
        config: dict,
        questions_path: Optional[str] = None,
        auto_mode: bool = False,
        auto_answers: Optional[dict] = None,
    ):
        self._valid_roles = config.get("roles", [])
        if not self._valid_roles:
            self._valid_roles = ["identity", "content_features", "interaction_signals", "filters", "ignore"]
            
        self._questions    = _load_questions(questions_path) if questions_path else []
        self._auto_mode    = auto_mode
        self._auto_answers = auto_answers if auto_mode else None

    def resolve(
        self,
        col_name: str,
        current_role: str,
        reasons: list[str],
        entity: str = "",
        field: str = "",
    ) -> str:
        """Run the question flow for one ambiguous column."""
        yaml_q = _find_yaml_question(self._questions, entity, field)

        return _ask_role(
            col_name=col_name,
            current_role=current_role,
            reasons=reasons,
            yaml_question=yaml_q,
            auto_answers=self._auto_answers,
            valid_roles=self._valid_roles,
        )
