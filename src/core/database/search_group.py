"""
Grouping support for search: capping results per metadata field value, with
helpers for building the refetch filter used to exclude already-saturated
group values on subsequent rounds.
"""

from __future__ import annotations

from typing import Any, Callable, Hashable, List, Optional, Set, Tuple

from src.core.common.enums import SearchExcludeField
from src.core.database.common import AmgixValidationError
from src.core.models.vector import CollectionConfigInternal, MetadataFilter


def required_fields_for_group(group_field: Optional[str]) -> set:
    """
    Fields that must be fetched regardless of exclude, because grouping needs
    the document's metadata to compute its group value.
    """
    if group_field:
        return {SearchExcludeField.METADATA}
    return set()


def validate_group_field(collection_config: CollectionConfigInternal, group_field: str) -> None:
    """
    Validate that group_field is a key declared in collection metadata_indexes.

    Raises:
        AmgixValidationError: If group_field is not indexed.
    """
    indexed_keys = {idx.key for idx in (collection_config.metadata_indexes or [])}
    if group_field not in indexed_keys:
        raise AmgixValidationError(
            f"group_field '{group_field}' is not indexed in collection metadata_indexes"
        )


def apply_group_cap(
    fused_results: List[Tuple[Hashable, float]],
    group_value_fn: Callable[[Hashable], Any],
    group_max: int,
    limit: int,
) -> Tuple[List[Tuple[Hashable, float]], Set[Any], bool, bool]:
    """
    Walk fused_results in rank order, keeping at most group_max items per
    distinct group value (documents missing the group field share a single
    None group value), until `limit` items are selected.

    Returns:
        (selected, saturated_values, null_saturated, pool_exhausted)
        - selected: capped, rank-ordered results, at most `limit` items.
        - saturated_values: non-None group values that hit group_max among selected.
        - null_saturated: whether the None (missing group field) bucket hit group_max.
        - pool_exhausted: whether the whole fused_results list was scanned without
          reaching `limit` selected items, meaning a refetch with an even tighter
          filter cannot add anything a fresh fetch wouldn't also miss.
    """
    selected: List[Tuple[Hashable, float]] = []
    group_counts: dict = {}

    for item_id, score in fused_results:
        if len(selected) >= limit:
            break
        group_value = group_value_fn(item_id)
        count = group_counts.get(group_value, 0)
        if count >= group_max:
            continue
        group_counts[group_value] = count + 1
        selected.append((item_id, score))

    pool_exhausted = len(selected) < limit

    saturated_values = {
        value for value, count in group_counts.items()
        if value is not None and count >= group_max
    }
    null_saturated = group_counts.get(None, 0) >= group_max

    return selected, saturated_values, null_saturated, pool_exhausted


def build_group_exclusion_filter(
    existing_filter: Optional[MetadataFilter],
    group_field: str,
    saturated_values: Set[Any],
    null_saturated: bool,
) -> Optional[MetadataFilter]:
    """
    AND the existing metadata filter with conditions excluding already-saturated
    group_field values, so the next fetch round surfaces fresh candidates.
    """
    exclusions: List[MetadataFilter] = [
        MetadataFilter(not_=MetadataFilter(key=group_field, op="eq", value=value))
        for value in saturated_values
    ]
    if null_saturated:
        exclusions.append(MetadataFilter(not_=MetadataFilter(key=group_field, op="is_null")))

    if not exclusions:
        return existing_filter
    if existing_filter is not None:
        exclusions.append(existing_filter)
    if len(exclusions) == 1:
        return exclusions[0]
    return MetadataFilter(and_=exclusions)
