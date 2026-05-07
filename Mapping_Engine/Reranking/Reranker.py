import re
import math
from rapidfuzz import fuzz

TYPE_GROUPS = {
    "number":   {"int", "float", "double", "decimal", "numeric", "bigint", "smallint", "real"},
    "datetime": {"date", "time", "timestamp", "datetime", "interval"},
    "boolean":  {"bool", "boolean", "bit"},
    "string":   {"varchar", "text", "char", "string", "nvarchar", "clob"},
}

_TYPE_LOOKUP = {v: k for k, vals in TYPE_GROUPS.items() for v in vals}

def normalize_type(dtype: str) -> str:
    dtype = str(dtype).lower().split("(")[0].strip()
    return _TYPE_LOOKUP.get(dtype, "string")

def sigmoid(x: float) -> float:
    if 0.0 <= x <= 1.0: return x
    return 1 / (1 + math.exp(-x))

def dtype_score(source_type: str, target_type: str) -> float:
    return 1.0 if normalize_type(source_type) == normalize_type(target_type) else 0.0

def name_score(source_col: str, target_field: str) -> float:
    return max(
        fuzz.token_sort_ratio(source_col, target_field),
        fuzz.partial_ratio(source_col, target_field),
    ) / 100.0

def table_score(source_table: str, target_entity: str) -> float:
    table  = source_table.lower()
    entity = target_entity.lower()
    cleaned = re.sub(r"^(tbl_|stg_|fact_|dim_|v_)", "", table).rstrip("s")
    return 1.0 if (entity in table or cleaned in entity or entity in cleaned) else 0.0

def _confidence_label(score: float) -> str:
    if score >= 0.80: return "High"
    if score >= 0.60: return "Medium"
    return "Low"

def _build_reason(signals: dict, label: str) -> str:
    reasons = []
    if signals.get("bi_semantic",    0) >= 0.60: reasons.append("semantic similarity")
    if signals.get("cross_semantic", 0) >= 0.60: reasons.append("contextual relevance")
    if signals.get("name",           0) >= 0.80: reasons.append("name match")
    if signals.get("logic_boost",    0) >  0:    reasons.append("keyword intent match")
    return f"{label} confidence: " + (", ".join(reasons) if reasons else "weak signal")


_PRIORITY_BOOST = {"high": 0.30, "medium": 0.20, "low": 0.10}


def apply_dynamic_logic(
    source_name: str,
    source_table: str,
    candidate: dict,
    source_intent: dict | None = None,
) -> float:
    """
    Fully data-driven boost/penalty computation.

    All signals come from YAML manifests — no hardcoded field names:
      1. Synonym match      — source_name found in candidate["synonyms"] (keywords.yaml)
      2. Canonical intent   — source_intent tells us the expected canonical target (keywords.yaml)
      3. Rules-based boost  — YAML context_rules already attached to the candidate

    Parameters
    ----------
    source_intent : dict | None
        Resolved from build_keyword_intent_index(keywords_path) in test.py.
        Keys: canonical_field, canonical_entity, match_priority.
    """
    boost = 0.0
    target_field  = candidate.get("field",   "").lower()
    target_entity = candidate.get("entity",  "").lower()

    synonyms = {
        s.lower().replace("-", "_").replace(" ", "_")
        for s in candidate.get("synonyms", [])
    }
    match_priority = candidate.get("match_priority", "medium")

    if source_name in synonyms:
        boost += _PRIORITY_BOOST.get(match_priority, 0.15)

    if source_intent:
        canonical_field  = source_intent["canonical_field"].lower()
        canonical_entity = source_intent["canonical_entity"].lower()
        intent_priority  = source_intent["match_priority"]

        if target_field == canonical_field and target_entity == canonical_entity:
            boost += _PRIORITY_BOOST.get(intent_priority, 0.15)
        elif target_entity != canonical_entity:
            boost -= 0.08

    rules = candidate.get("context_rules", [])
    for rule in rules:
        rule_lower = rule.lower()
        if "prioritize" in rule_lower and source_table in rule_lower:
            if target_entity in rule_lower:
                if "over" in rule_lower:
                    parts = rule_lower.split("over")
                    if target_entity in parts[0]: boost += 0.20
                    else:                          boost -= 0.20
                else:
                    boost += 0.15

    return boost


def rerank_candidates(
    source_column: dict,
    candidates: list[dict],
    cross_encoder=None,
    keyword_intent: dict | None = None,
) -> list[dict]:
    source_name  = source_column.get("column",    "").lower()
    source_table = source_column.get("table",     "").lower()
    source_dtype = source_column.get("data_type", "string")

   
    source_intent: dict | None = None
    if keyword_intent:
        source_intent = keyword_intent.get(source_name)


    cross_scores = [0.0] * len(candidates)
    if cross_encoder and candidates:
        pairs = []
        for cand in candidates:
            s_text = f"Source column '{source_name}' from table '{source_table}'"

            synonyms_str = ", ".join(cand.get("synonyms",     []))
            description  = cand.get("description",            "")
            rules_str    = " ".join(cand.get("context_rules", []))

            t_text = (
                f"Target field '{cand['field']}' in entity '{cand['entity']}'. "
                f"Business meaning: {description}. "
                f"Synonyms: {synonyms_str}. "
                f"Rules: {rules_str}"
            ).replace("_", " ")

            pairs.append([s_text, t_text])

        raw_scores = cross_encoder.predict(pairs)
        if isinstance(raw_scores, float): raw_scores = [raw_scores]
        cross_scores = [sigmoid(s) for s in raw_scores]

    reranked = []
    for i, cand in enumerate(candidates):
        signals = {
            "bi_semantic":    cand.get("score", 0),           # Bug #1 fixed
            "cross_semantic": cross_scores[i],
            "name":           name_score(source_name, cand.get("field", "")),  # Bug #5 fixed
            "dtype":          dtype_score(source_dtype, cand.get("data_type", "string")),  # Bug #2 fixed
            "table":          table_score(source_table, cand.get("entity", "")),
            "logic_boost":    apply_dynamic_logic(            # Bug #4 fixed, now data-driven
                source_name, source_table, cand, source_intent=source_intent
            ),
        }

        base_confidence = (
            signals["bi_semantic"]    * 0.25 +
            signals["cross_semantic"] * 0.50 +
            signals["name"]           * 0.15 +
            signals["table"]          * 0.05 +
            signals["dtype"]          * 0.05
        )

        final_confidence = max(0.0, min(1.0, base_confidence + signals["logic_boost"]))

        cand["final_confidence"] = final_confidence
        cand["signals"]          = signals
        cand["reasoning"]        = _build_reason(signals, _confidence_label(final_confidence))

        reranked.append(cand)

    reranked.sort(key=lambda x: x["final_confidence"], reverse=True)
    return reranked