"""
Parse a filter expression string into a MetadataFilter-compatible dict tree.

Supported syntax:
    comparison  : FIELD OP value
                | FIELD IS NULL
                | FIELD IS NOT NULL
    OP          : "=" | "!=" | "<" | "<=" | ">" | ">="
    value       : string | integer | float | boolean | null
    boolean ops : AND, OR, NOT (case-insensitive)
    grouping    : parentheses

Examples:
    year > 2020
    status = "active" AND enabled = true
    (year > 2020 AND year < 2030) OR status = "draft"
    NOT deleted = true
    status != "archived"
    category IS NULL
    category IS NOT NULL

Operators map to MetadataFilter ops:
    =   -> eq
    !=  -> neq
    <   -> lt
    <=  -> lte
    >   -> gt
    >=  -> gte
    IS NULL     -> is_null
    IS NOT NULL -> not(is_null)
"""

from __future__ import annotations

from lark import Lark, Transformer, Token, v_args
from lark.exceptions import UnexpectedInput

FILTER_EXPR_GRAMMAR = r"""
    ?expr : expr _AND expr  -> and_expr
          | expr _OR expr   -> or_expr
          | _NOT expr       -> not_expr
          | "(" expr ")"
          | comparison

    comparison : FIELD OP value        -> comparison
               | FIELD IS_NOT_NULL     -> is_not_null_comparison
               | FIELD IS_NULL         -> is_null_comparison

    value : ESCAPED_STRING  -> string_val
          | SIGNED_FLOAT    -> float_val
          | SIGNED_INT      -> int_val
          | TRUE             -> bool_val
          | FALSE            -> bool_val
          | NULL             -> null_val

    OP     : "!=" | "=" | "<=" | ">=" | "<" | ">"
    FIELD  : /[a-zA-Z0-9_][a-zA-Z0-9_-]*/
    TRUE   : /true/i
    FALSE  : /false/i
    NULL   : /null/i
    IS_NOT_NULL.2 : /IS\s+NOT\s+NULL/i
    IS_NULL.1     : /IS\s+NULL/i

    _AND : /AND/i
    _OR  : /OR/i
    _NOT : /NOT/i

    %import common.ESCAPED_STRING
    %import common.SIGNED_FLOAT
    %import common.SIGNED_INT
    %import common.WS
    %ignore WS
"""

_GRAMMAR = "start: expr\n" + FILTER_EXPR_GRAMMAR

_OP_MAP = {
    "=":  "eq",
    "!=": "neq",
    "<":  "lt",
    ">":  "gt",
    "<=": "lte",
    ">=": "gte",
}

_parser = Lark(_GRAMMAR, start="expr", parser="earley", ambiguity="resolve")


@v_args(inline=True)
class FilterExprTransformer(Transformer):
    def and_expr(self, left, right):
        return {"and": [left, right]}

    def or_expr(self, left, right):
        return {"or": [left, right]}

    def not_expr(self, operand):
        return {"not": operand}

    def comparison(self, field: Token, op: Token, value):
        op_str = str(op)
        if op_str not in _OP_MAP:
            raise ValueError(f"Unsupported operator '{op_str}'")
        return {"key": str(field), "op": _OP_MAP[op_str], "value": value}

    def is_null_comparison(self, field: Token, _is_null: Token):
        return {"key": str(field), "op": "is_null"}

    def is_not_null_comparison(self, field: Token, _is_not_null: Token):
        return {"not": {"key": str(field), "op": "is_null"}}

    def string_val(self, s: Token):
        return str(s)[1:-1].replace('\\"', '"')

    def float_val(self, n: Token):
        return float(n)

    def int_val(self, n: Token):
        return int(n)

    def bool_val(self, b: Token):
        return str(b).lower() == "true"

    def null_val(self, _):
        return None


_transformer = FilterExprTransformer()


def parse_filter_to_dict(expr: str) -> dict:
    """Parse a filter expression string into a MetadataFilter-compatible dict.

    Raises ValueError with a human-readable message on parse error.
    """
    try:
        tree = _parser.parse(expr.strip())
    except UnexpectedInput as e:
        raise ValueError(f"Invalid filter expression: {e.get_context(expr)}") from None
    return _transformer.transform(tree)
