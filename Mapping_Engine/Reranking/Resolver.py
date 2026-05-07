from typing import List, Dict, Any

def resolve_mappings(all_source_results: List[Dict[str, Any]], threshold: float = 0.7) -> List[Dict[str, Any]]:
    """
    Enforces a Global 1:1 Mapping constraint.

    Args:
        all_source_results: List of { 'source': col_dict, 'candidates': reranked_list }
        threshold: Confidence threshold for auto-acceptance

    Returns:
        List of finalized mapping results.
    """

    candidates_by_source: Dict[str, list] = {}
    for res in all_source_results:
        source_col = res["source"]
        s_key = f"{source_col['table']}.{source_col['column']}"
        candidates_by_source[s_key] = res["candidates"]

    global_pairs = []
    for res in all_source_results:
        source_col = res["source"]
        source_key = f"{source_col['table']}.{source_col['column']}"

        for cand in res["candidates"]:
            global_pairs.append({
                "source_key": source_key,
                "source_col": source_col,
                "target":     cand,
                "score":      cand["final_confidence"],
            })

    global_pairs.sort(key=lambda x: x["score"], reverse=True)

    final_mappings: Dict[str, dict] = {}
    taken_targets:  set = set()
    mapped_sources: set = set()

    for pair in global_pairs:
        s_key = pair["source_key"]
        t_id  = pair["target"]["field_id"]

        if s_key not in mapped_sources and t_id not in taken_targets:
            score = pair["score"]

            if score >= threshold:
                status       = "AUTO_ACCEPTED" if score >= 0.99 else "NEEDS_REVIEW"
                mapping_info = pair["target"]
                taken_targets.add(t_id)
            else:
                status       = "CUSTOM_FIELD"
                mapping_info = {
                    "entity": "CUSTOM",
                    "field":  pair["source_col"]["column"],
                }

            top5 = candidates_by_source.get(s_key, [pair["target"]])[:5]

            final_mappings[s_key] = {
                "source_table":      pair["source_col"]["table"],
                "source_column":     pair["source_col"]["column"],
                "status":            status,
                "recommended_mapping": mapping_info,
                "confidence":        score,
                "top_candidates":    top5,
            }

            mapped_sources.add(s_key)

    for res in all_source_results:
        source_col = res["source"]
        s_key = f"{source_col['table']}.{source_col['column']}"

        if s_key not in mapped_sources:
            top5 = res["candidates"][:5]
            final_mappings[s_key] = {
                "source_table":    source_col["table"],
                "source_column":   source_col["column"],
                "status":          "CUSTOM_FIELD",
                "recommended_mapping": {
                    "entity": "CUSTOM",
                    "field":  source_col["column"],
                },
                "confidence":      top5[0]["final_confidence"] if top5 else 0.0,
                "top_candidates":  top5,
            }

    return list(final_mappings.values())
