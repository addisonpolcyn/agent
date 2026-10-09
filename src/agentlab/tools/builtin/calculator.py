"""Deterministic arithmetic. Parses the expression with ``ast``; never calls ``eval``."""

from __future__ import annotations

import ast
import math
from typing import TYPE_CHECKING, Any

from agentlab.tools.models import ToolError

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from agentlab.models import JSONObject

type Number = int | float

MAX_EXPRESSION_LENGTH = 200
MAX_EXPONENT = 100

# ``**`` can produce a complex number (e.g. ``(-8) ** 0.5``), hence the wider return type.
_BINARY_OPS: dict[type[ast.operator], Callable[[Number, Number], Number | complex]] = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.FloorDiv: lambda a, b: a // b,
    ast.Mod: lambda a, b: a % b,
    ast.Pow: lambda a, b: a**b,
}
_UNARY_OPS: dict[type[ast.unaryop], Callable[[Number], Number]] = {
    ast.UAdd: lambda a: +a,
    ast.USub: lambda a: -a,
}


def run(arguments: Mapping[str, Any]) -> JSONObject:
    expression = arguments.get("expression")
    if not isinstance(expression, str) or not expression.strip():
        raise ToolError("'expression' must be a non-empty string")
    if len(expression) > MAX_EXPRESSION_LENGTH:
        raise ToolError(f"expression longer than {MAX_EXPRESSION_LENGTH} characters")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ToolError(f"not a valid arithmetic expression: {expression!r}") from exc
    return {"result": _normalize(_evaluate(tree.body))}


def _evaluate(node: ast.expr) -> Number:
    match node:
        case ast.Constant(value=bool()):
            raise ToolError("booleans are not numbers")
        case ast.Constant(value=int() | float() as value):
            return value
        case ast.UnaryOp(op=op, operand=operand) if type(op) in _UNARY_OPS:
            return _UNARY_OPS[type(op)](_evaluate(operand))
        case ast.BinOp(left=left, op=op, right=right) if type(op) in _BINARY_OPS:
            return _apply(op, _evaluate(left), _evaluate(right))
        case _:
            raise ToolError(f"unsupported syntax: {ast.unparse(node)!r}")


def _apply(op: ast.operator, left: Number, right: Number) -> Number:
    if isinstance(op, ast.Pow) and abs(right) > MAX_EXPONENT:
        raise ToolError(f"exponent larger than {MAX_EXPONENT}")
    try:
        result = _BINARY_OPS[type(op)](left, right)
    except ZeroDivisionError as exc:
        raise ToolError("division by zero") from exc
    except OverflowError as exc:
        raise ToolError("result is too large") from exc
    if isinstance(result, complex):
        raise ToolError("result is not a real number")
    return result


def _normalize(value: Number) -> Number:
    """Present whole floats as ints (``6 / 2`` -> ``3``) so answers read naturally."""
    if isinstance(value, float) and not math.isfinite(value):
        raise ToolError("result is too large")
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value
