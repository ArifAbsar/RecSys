def build_target_text(target: dict) -> str:
    """
    Text used for embedding standard ecommerce fields.
    """

    business_types = ", ".join(target.get("entity_business_types", []))
    covers = ", ".join(target.get("entity_covers", []))
    source_hints = ", ".join(target.get("source_system_hints", []))
    required_when = "; ".join(target.get("required_when", []))

    return f"""
Schema: {target.get("schema_name")}
Entity: {target.get("entity")}
Entity purpose: {target.get("entity_purpose")}
Entity covers: {covers}
Field: {target.get("field_name")}
Data type: {target.get("data_type")}
Semantic role: {target.get("semantic_role")}
Description: {target.get("description")}
Requirement level: {target.get("requirement_level")}
Required when: {required_when}
Business types: {business_types}
Source system hints: {source_hints}
Domain: ecommerce
""".strip()


def build_source_text(source_column: dict) -> str:
    """
    Text used for embedding user's source database column.
    """

    sample_values = ", ".join([str(v) for v in source_column.get("sample_values", [])[:5]])

    return f"""
Table: {source_column.get("table")}
Column: {source_column.get("column")}
Data type: {source_column.get("data_type")}
Sample values: {sample_values}
Null rate: {source_column.get("null_rate")}
Domain: ecommerce
""".strip()