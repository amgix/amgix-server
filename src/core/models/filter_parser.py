"""
Parse a filter expression string into a MetadataFilter-compatible dict tree.

Supported syntax:
    comparison  : FIELD OP value
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

Operators map to MetadataFilter ops:
    =   -> eq
    !=  -> neq
    <   -> lt
    <=  -> lte
    >   -> gt
    >=  -> gte
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

    comparison : FIELD OP value

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
