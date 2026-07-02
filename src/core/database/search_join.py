"""
Document enrichment: left-join documents from other collections.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Union

from src.core.common.functions import get_real_collection_name
from src.core.database.base import AmgixNotFound, DatabaseBase
from src.core.database.common import AmgixValidationError, validate_metadata_filter
from src.core.models.document import Document, DocumentWithVectors
from src.core.models.join_parser import JoinSideRef, JoinSpec, parse_joins
from src.core.models.vector import CollectionConfigInternal, MetadataFilter


def _parent_join_value(document: Document, ref: JoinSideRef) -> Any:
    if ref.kind == "id":
        return document.id
    if ref.meta_key:
        if not document.metadata:
            return None
        return document.metadata.get(ref.meta_key)
    return None


def _child_join_value(doc: Document, ref: JoinSideRef) -> Any:
    if ref.kind == "id":
        return doc.id
    if ref.meta_key:
        if not doc.metadata:
            return None
        return doc.metadata.get(ref.meta_key)
    return None


def _values_equal(a: Any, b: Any) -> bool:
    if a is None or b is None:
        return a is b
    return a == b


def _document_from_with_vectors(dwv: DocumentWithVectors) -> Document:
    data = dwv.model_dump()
    data.pop("vectors", None)
    data.pop("token_lengths", None)
    return Document.model_construct(**data)


def document_matches_metadata_filter(
    document: Document,
    metadata_filter: MetadataFilter,
    collection_config: CollectionConfigInternal,
) -> bool:
    """Evaluate a metadata filter against a single document (for post-fetch filtering)."""
    metadata = document.metadata or {}

    def eval_node(node: MetadataFilter) -> bool:
        if node.key is not None:
            value = metadata.get(node.key)
            cmp = node.value
            op = node.op
            if op == "eq":
                return _values_equal(value, cmp)
            if op == "neq":
                return not _values_equal(value, cmp)
            if value is None:
                return False
            if op == "gt":
                return value > cmp
            if op == "gte":
                return value >= cmp
            if op == "lt":
                return value < cmp
            if op == "lte":
                return value <= cmp
            return False

        if node.not_ is not None:
            return not eval_node(node.not_)
        if node.and_ is not None:
            return all(eval_node(child) for child in node.and_)
        if node.or_ is not None:
            return any(eval_node(child) for child in node.or_)
        return True

    return eval_node(metadata_filter)


def _validate_join_spec(spec: JoinSpec, child_config: CollectionConfigInternal) -> None:
    if spec.child_ref.kind == "meta":
        key = spec.child_ref.meta_key
        if not child_config.metadata_indexes:
            raise AmgixValidationError(
                f"Join child collection '{spec.collection_name}' has no metadata_indexes; "
                f"cannot join on metadata key '{key}'"
            )
        indexed_keys = {idx.key for idx in child_config.metadata_indexes}
        if key not in indexed_keys:
            raise AmgixValidationError(
                f"Join child metadata key '{key}' is not indexed in collection "
                f"'{spec.collection_name}'"
            )
    if spec.metadata_filter:
        validate_metadata_filter(child_config, spec.metadata_filter)


async def _fetch_children_for_join(
    database: DatabaseBase,
    spec: JoinSpec,
    join_values: List[Any],
    child_config: CollectionConfigInternal,
    max_documents: int,
) -> List[Document]:
    real_name = get_real_collection_name(spec.collection_name)
    if not join_values:
        return []

    if spec.child_ref.kind == "id":
        str_ids = [str(v) for v in join_values]
        fetched = await database.get_documents(real_name, str_ids, suppress_not_found=True)
        docs: List[Document] = []
        for dwv in fetched:
            if dwv is None:
                continue
            doc = _document_from_with_vectors(dwv)
            if spec.metadata_filter and not document_matches_metadata_filter(
                doc, spec.metadata_filter, child_config
            ):
                continue
            docs.append(doc)
        return docs

    key = spec.child_ref.meta_key
    return await database.fetch_documents_by_metadata_values(
        real_name,
        key,
        join_values,
        spec.metadata_filter,
        child_config,
        max_documents,
    )


def _group_children_by_join_key(
    children: List[Document],
    child_ref: JoinSideRef,
) -> Dict[str, List[Document]]:
    groups: Dict[str, List[Document]] = {}
    for doc in children:
        jv = _child_join_value(doc, child_ref)
        if jv is None:
            continue
        key = _join_value_key(jv)
        groups.setdefault(key, []).append(doc)
    return groups


def _join_value_key(value: Any) -> str:
    """Stable dict key for grouping join values."""
    if isinstance(value, (str, int, float, bool)):
        return json.dumps(value, sort_keys=True)
    return json.dumps(value, sort_keys=True, default=str)


async def enrich_documents_with_joins(
    database: DatabaseBase,
    documents: List[Document],
    join: Union[str, List[str]],
    limit: int,
) -> List[Document]:
    try:
        specs = parse_joins(join)
    except ValueError as e:
        raise AmgixValidationError(str(e)) from None

    for spec in specs:
        real_child = get_real_collection_name(spec.collection_name)
        try:
            child_config = await database.get_collection_info_internal(real_child)
        except AmgixNotFound:
            raise AmgixValidationError(
                f"Join collection '{spec.collection_name}' not found"
            ) from None
        if not child_config:
            raise AmgixValidationError(
                f"Join collection '{spec.collection_name}' not found"
            )
        _validate_join_spec(spec, child_config)

        join_values: List[Any] = []
        seen: set[str] = set()
        for document in documents:
            pv = _parent_join_value(document, spec.parent_ref)
            if pv is None:
                continue
            k = _join_value_key(pv)
            if k not in seen:
                seen.add(k)
                join_values.append(pv)

        max_documents = limit * len(documents)
        children = await _fetch_children_for_join(
            database, spec, join_values, child_config, max_documents
        )
        by_key = _group_children_by_join_key(children, spec.child_ref)

        for document in documents:
            if document.joined is None:
                document.joined = {}
            pv = _parent_join_value(document, spec.parent_ref)
            if pv is None:
                document.joined[spec.collection_name] = []
            else:
                document.joined[spec.collection_name] = by_key.get(
                    _join_value_key(pv), []
                )

    return documents
