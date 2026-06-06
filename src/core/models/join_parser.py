"""
Parse a search join expression string (Lark).

Syntax:
    <collection>                          → [$id=$$id], no filter
    <collection>[<parent>=<child>]        → optional join keys
    <collection>(<filter>)                → optional child metadata filter
    <collection>[<parent>=<child>](<filter>)

Parent refs: $id | $.meta.<key>
Child refs:  $$id | $$.meta.<key>
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Union

from lark import Lark, Token, v_args
from lark.exceptions import UnexpectedInput

from src.core.common.constants import MAX_COLLECTION_NAME_LENGTH, MAX_METADATA_KEY_LENGTH
from src.core.models.filter_parser import FILTER_EXPR_GRAMMAR, FilterExprTransformer
from src.core.models.vector import MetadataFilter

_JOIN_GRAMMAR = (
    r"""
    start: join_expr

    join_expr: COLLECTION join_suffix

    join_suffix: -> bare_join
                | filter_part -> filter_only_join
                | join_keys -> keys_only_join
                | join_keys filter_part -> keys_and_filter_join

    join_keys: "[" parent_ref "=" child_ref "]"

    parent_ref: "$id" -> parent_id
              | "$.meta." META_KEY -> parent_meta

    child_ref: "$$id" -> child_id
             | "$$.meta." META_KEY -> child_meta

    filter_part: "(" expr ")" -> join_filter

    COLLECTION: /[a-zA-Z0-9_-]+/
    META_KEY: /[a-zA-Z0-9_][a-zA-Z0-9_-]*/

    %import common.WS
    %ignore WS
"""
    + FILTER_EXPR_GRAMMAR
)

_parser = Lark(_JOIN_GRAMMAR, start="join_expr", parser="earley", ambiguity="resolve")


@dataclass(frozen=True)
class JoinSideRef:
    kind: Literal["id", "meta"]
    meta_key: Optional[str] = None


@dataclass(frozen=True)
class JoinSpec:
    collection_name: str
    parent_ref: JoinSideRef
    child_ref: JoinSideRef
    metadata_filter: Optional[MetadataFilter] = None


def _validate_collection_name(name: str) -> None:
    if not name or len(name) > MAX_COLLECTION_NAME_LENGTH:
        raise ValueError(
            f"Invalid collection name '{name}': must be 1–{MAX_COLLECTION_NAME_LENGTH} characters"
        )


def _validate_meta_key(key: str) -> None:
    if len(key) > MAX_METADATA_KEY_LENGTH:
        raise ValueError(
            f"Metadata key '{key}' exceeds {MAX_METADATA_KEY_LENGTH} character limit"
        )


@v_args(inline=True)
class _JoinTransformer(FilterExprTransformer):
    def join_expr(self, collection: Token, join_suffix: dict) -> JoinSpec:
        collection_name = str(collection)
        _validate_collection_name(collection_name)
        return JoinSpec(collection_name=collection_name, **join_suffix)

    def bare_join(self) -> dict:
        return {
            "parent_ref": JoinSideRef(kind="id"),
            "child_ref": JoinSideRef(kind="id"),
            "metadata_filter": None,
        }

    def filter_only_join(self, filter_part: MetadataFilter) -> dict:
        return {
            "parent_ref": JoinSideRef(kind="id"),
            "child_ref": JoinSideRef(kind="id"),
            "metadata_filter": filter_part,
        }

    def keys_only_join(self, join_keys: tuple) -> dict:
        parent_ref, child_ref = join_keys
        return {
            "parent_ref": parent_ref,
            "child_ref": child_ref,
            "metadata_filter": None,
        }

    def keys_and_filter_join(self, join_keys: tuple, filter_part: MetadataFilter) -> dict:
        parent_ref, child_ref = join_keys
        return {
            "parent_ref": parent_ref,
            "child_ref": child_ref,
            "metadata_filter": filter_part,
        }

    def join_keys(self, parent_ref: JoinSideRef, child_ref: JoinSideRef):
        return parent_ref, child_ref

    def parent_id(self) -> JoinSideRef:
        return JoinSideRef(kind="id")

    def parent_meta(self, key: Token) -> JoinSideRef:
        meta_key = str(key)
        _validate_meta_key(meta_key)
        return JoinSideRef(kind="meta", meta_key=meta_key)

    def child_id(self) -> JoinSideRef:
        return JoinSideRef(kind="id")

    def child_meta(self, key: Token) -> JoinSideRef:
        meta_key = str(key)
        _validate_meta_key(meta_key)
        return JoinSideRef(kind="meta", meta_key=meta_key)

    def join_filter(self, filter_dict: dict) -> MetadataFilter:
        return MetadataFilter.model_validate(filter_dict)


_transformer = _JoinTransformer()


def parse_join(expr: str) -> JoinSpec:
    """Parse a join expression. Raises ValueError on invalid input."""
    text = expr.strip()
    if not text:
        raise ValueError("Join expression cannot be empty")
    try:
        tree = _parser.parse(text)
    except UnexpectedInput as e:
        raise ValueError(f"Invalid join expression: {e.get_context(expr)}") from None
    return _transformer.transform(tree)


def parse_joins(join: Union[str, list]) -> list[JoinSpec]:
    """Parse join field from SearchQuery (string or list for multiple joins)."""
    if isinstance(join, str):
        specs = [parse_join(join)]
    elif isinstance(join, list):
        if not join:
            raise ValueError("Join list cannot be empty")
        specs = [parse_join(item) for item in join]
    else:
        raise ValueError("Join must be a string or list of strings")

    seen: set[str] = set()
    for spec in specs:
        if spec.collection_name in seen:
            raise ValueError(
                f"Duplicate join collection '{spec.collection_name}' in join list"
            )
        seen.add(spec.collection_name)
    return specs
