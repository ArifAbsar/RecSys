import argparse
from pathlib import Path
try:
    from .detection import load_yaml, detect_list_key, detect_name_key, flatten_record
except ImportError:
    from detection import load_yaml, detect_list_key, detect_name_key, flatten_record


def build_target_registry(schema_path: str):
    schema = load_yaml(schema_path)

    schema_name = schema.get("schema_name", Path(schema_path).stem)
    version = schema.get("version", "unknown_version")

    entity_list_key = detect_list_key(
        {key: value for key, value in schema.items() if key not in ("schema_name", "version")}
    )
    if not entity_list_key:
        raise ValueError("Could not detect a list of entity objects in the YAML.")

    entities = schema[entity_list_key]
    targets = []

    for entity in entities:
        entity_name_key = detect_name_key(entity)
        entity_name = entity.get(entity_name_key, "unknown") if entity_name_key else "unknown"

        field_list_key = detect_list_key(entity)

        skip = {entity_name_key, field_list_key} if field_list_key else {entity_name_key}
        entity_meta = {}
        for key, value in entity.items():
            if key in skip:
                continue
            if key == "purpose":
                entity_meta["entity_purpose"] = value
            elif key == "covers":
                entity_meta["entity_covers"] = value
            elif key == "business_types":
                entity_meta["entity_business_types"] = value
            else:
                entity_meta[key] = value

        if not field_list_key:
            targets.append({
                "field_id": f"{schema_name}.{entity_name}",
                "schema_name": schema_name,
                "schema_version": version,
                "entity": entity_name,
                **entity_meta,
            })
            continue

        for field in entity.get(field_list_key, []):
            field_name_key = detect_name_key(field)
            field_name = field.get(field_name_key, "unknown") if field_name_key else "unknown"

            field_meta = flatten_record(field, skip_keys={field_name_key})

            targets.append({
                "field_id": f"{schema_name}.{entity_name}.{field_name}",
                "schema_name": schema_name,
                "schema_version": version,
                "entity": entity_name,
                **entity_meta,
                "field_name": field_name,
                **field_meta,
            })

    return targets


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the target field registry from a schema YAML.")
    parser.add_argument("--path", required=True, help="Path to the schema YAML file (e.g. ecommerce/schema.yaml)")
    args = parser.parse_args()

    targets = build_target_registry(args.path)
    print(f"Loaded {len(targets)} target fields")
    print(targets[0])