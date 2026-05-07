import os
import yaml
from pathlib import Path

def build_target_registry(schema_path: str, keywords_path: str = None, metrics_path: str = None, 
                          rules_path: str = None, validations_path: str = None, questions_path: str = None) -> list[dict]:
    """
    Combines the structural schema with business-logic manifests.
    Extracts deep metadata (purpose, roles, business types) for the AI.
    """
    if not os.path.exists(schema_path):
        raise FileNotFoundError(f"Schema file not found: {schema_path}")

    with open(schema_path, 'r') as f:
        schema = yaml.safe_load(f)

    schema_name = schema.get('schema_name', 'universal_ecommerce')
    keywords_data = {}
    if keywords_path and os.path.exists(keywords_path):
        with open(keywords_path, 'r') as f:
            keywords_data = yaml.safe_load(f)

    metrics_data = []
    if metrics_path and os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics_data = yaml.safe_load(f).get("derived_metrics", [])

    rules_data = []
    if rules_path and os.path.exists(rules_path):
        with open(rules_path, 'r') as f:
            data = yaml.safe_load(f)
            rules_data = data.get("field_recognition_rules", []) + \
                         data.get("conflict_resolution_rules", []) + \
                         data.get("requirement_resolution_rules", []) + \
                         data.get("entity_recognition_rules", [])

    validations_data = []
    if validations_path and os.path.exists(validations_path):
        with open(validations_path, 'r') as f:
            data = yaml.safe_load(f)
            validations_data = data.get("global_rules", []) + data.get("entity_rules", [])

    questions_data = []
    if questions_path and os.path.exists(questions_path):
        with open(questions_path, 'r') as f:
            data = yaml.safe_load(f)
            for group in data.get("question_groups", []):
                questions_data.extend(group.get("questions", []))

    registry = []
    entities_list = schema.get('entities', [])
    
    for entity_item in entities_list:
        entity_name = entity_item.get('entity', 'unknown')
        entity_purpose = entity_item.get('purpose', '')
        entity_covers = entity_item.get('covers', [])
        entity_business_types = entity_item.get('business_types_supported', [])
        
        fields_list = entity_item.get('fields', [])
        
        for field_item in fields_list:
            field_name = field_item.get('name', 'unknown')
            field_id = f"{entity_name}.{field_id}" if 'field_id' in locals() else f"{entity_name}.{field_name}"
            # Reset field_id correctly
            field_id = f"{entity_name}.{field_name}"
            
            target = {
                "schema_name":       schema_name,
                "field_id":          field_id,
                "entity":            entity_name,
                "entity_purpose":    entity_purpose,
                "entity_covers":     entity_covers,
                "entity_business_types": entity_business_types,
                "field":             field_name,
                "data_type":         field_item.get('data_type', 'string'),
                "semantic_role":     field_item.get('semantic_role', ''),
                "description":       field_item.get('description', ''),
                "requirement_level": field_item.get('requirement_level', 'optional'),
                "required_when":     field_item.get('required_when', []),
                "source_system_hints": field_item.get('source_system_hints', []),
                "synonyms":          [],
                "match_priority":    "medium",
                "context_rules":     []
            }

            if field_name in keywords_data:
                kw = keywords_data[field_name]
                target["synonyms"]       = kw.get("synonyms", [])
                target["match_priority"] = kw.get("match_priority", "medium")
            for m in metrics_data:
                if field_name in m.get("uses_fields", []):
                    target["context_rules"].append(f"Metric '{m['name']}': {m['business_use']}")

            for r in rules_data:
                logic = r.get("logic", "").lower()
                rule_name = r.get("name", "").lower()
                if field_name in logic or field_name in rule_name:
                    target["context_rules"].append(f"Rule '{r.get('name')}': {r.get('logic')}")

            for v in validations_data:
                if v.get("entity") == entity_name:
                    for check in v.get("recommended_checks", []):
                        if field_name in check.get("check", "").lower() or field_name in check.get("name", "").lower():
                            target["context_rules"].append(f"Validation '{check['name']}': {check['check']}")
                elif "check" in v and field_name in v["check"].lower():
                    target["context_rules"].append(f"Global Validation '{v['name']}': {v['check']}")

            for q in questions_data:
                maps_to = [m.lower() for m in q.get("maps_to", [])]
                if field_id.lower() in maps_to or entity_name.lower() in maps_to or field_name.lower() in maps_to:
                    target["context_rules"].append(f"Business Ambiguity Question: {q.get('question')}")

            registry.append(target)

    return registry


def build_keyword_intent_index(keywords_path: str) -> dict:
    """
    Builds a reverse lookup: synonym (lowercase) → canonical intent.
    Used by the reranker to resolve source column names to their
    canonical field/entity without any hardcoded field name checks.

    Returns:
        { "revenue": {canonical_field, canonical_entity, match_priority}, ... }
    """
    if not keywords_path or not os.path.exists(keywords_path):
        return {}

    with open(keywords_path, 'r') as f:
        raw = yaml.safe_load(f)

    index = {}
    for field_name, kw_data in raw.get("field_keywords", {}).items():
        canonical_entity   = kw_data.get("canonical_entity", "")
        match_priority     = kw_data.get("match_priority",   "medium")
        for syn in kw_data.get("synonyms", []):
            key = syn.lower().replace("-", "_").replace(" ", "_")
            if key not in index:
                index[key] = {
                    "canonical_field":  field_name,
                    "canonical_entity": canonical_entity,
                    "match_priority":   match_priority,
                }
    return index