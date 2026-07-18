"""
Faceting support for search: helpers for validating the facets toggle and
ensuring the metadata needed to compute facet counts is fetched.

Facet aggregation itself lives in the DB backends (qdrant.py / sql_base.py),
since it needs the raw candidate pool produced by the search arms. This module
only holds the backend-agnostic validation / required-fields helpers.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Tuple

from src.core.common.enums import SearchExcludeField


def required_fields_for_facets(facets_enabled: bool) -> set:
    """
    Fields that must be fetched regardless of exclude, because faceting needs
    the document's metadata to count values per indexed field.
    """
    if facets_enabled:
        return {SearchExcludeField.METADATA}
    return set()


def validate_facets(collection_config, facets_enabled: bool) -> None:
    """
    Validate a facets-enabled request against the collection config.

    Faceting runs over every field declared in metadata_indexes. If the collection
    has no metadata_indexes, faceting is a no-op (an empty facet_counts is returned),
    so this is not treated as an error.
    """
    # No per-field validation: metadata_indexes are validated at collection creation,
    # and we facet over all of them. Nothing to reject here.
    return None


def facet_value_key(raw: Any, idx_type: str) -> str:
    """Canonical string key for a facet value, normalized by declared index type
    so Qdrant (JSON payloads) and SQL (DECIMAL/TIMESTAMP columns) agree.

    - integer  -> str(int(raw))           e.g. "2020"
    - float    -> str(float(raw))          e.g. "1.5"
    - boolean  -> "true" / "false"
    - datetime -> isoformat() when available, else str(raw)
    - string   -> str(raw)
    """
    if idx_type == "integer":
        return str(int(raw))
    if idx_type == "float":
        return str(float(raw))
    if idx_type == "boolean":
        return "true" if raw else "false"
    if idx_type == "datetime":
        if hasattr(raw, "isoformat"):
            return raw.isoformat()
    return str(raw)


def compute_facet_counts(
    metadata_iter: Iterable[Optional[Dict]],
    indexed_fields: Iterable[Tuple[str, str]],
    max_values: int,
) -> Dict[str, Dict[str, int]]:
    """
    Count per-field facet values over an iterable of metadata dicts (the candidate
    pool). Documents missing a field are skipped for that field. Values are
    normalized to canonical string keys via facet_value_key, using each field's
    declared index type. Each field is truncated to the top-N values by count.

    indexed_fields is an iterable of (field_name, index_type) tuples.

    Used by backends that already hold candidate metadata in memory (Qdrant).
    """
    field_list = list(indexed_fields)
    counts: Dict[str, Dict[str, int]] = {name: {} for name, _ in field_list}
    for md in metadata_iter:
        if not md:
            continue
        for name, idx_type in field_list:
            v = md.get(name)
            if v is None:
                continue
            key = facet_value_key(v, idx_type)
            counts[name][key] = counts[name].get(key, 0) + 1
    return {
        name: dict(sorted(c.items(), key=lambda kv: -kv[1])[:max_values])
        for name, c in counts.items()
    }


