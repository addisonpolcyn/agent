from __future__ import annotations

import pytest

from agentlab.skills.builtin.calculator import run
from agentlab.skills.models import SkillError


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("120 * 3", 360),
        ("123 * 456", 56088),
        ("2 + 3 * 4", 14),
        ("(2 + 3) * 4", 20),
        ("-5 + 2", -3),
        ("7 / 2", 3.5),
        ("6 / 2", 3),
        ("7 // 2", 3),
        ("7 % 4", 3),
        ("2 ** 10", 1024),
    ],
)
def test_evaluates_arithmetic(expression: str, expected: float) -> None:
    assert run({"expression": expression}) == {"result": expected}


@pytest.mark.parametrize(
    ("expression", "message"),
    [
        ("1 / 0", "division by zero"),
        ("__import__('os')", "unsupported syntax"),
        ("x + 1", "unsupported syntax"),
        ("True + 1", "booleans"),
        ("2 ** 1000", "exponent"),
        ("1e300 ** 50", "too large"),
        ("1e308 * 10", "too large"),
        ("(-8) ** 0.5", "not a real number"),
        ("1 +", "not a valid arithmetic expression"),
        ("1" * 201, "longer than"),
    ],
)
def test_rejects_invalid_input(expression: str, message: str) -> None:
    with pytest.raises(SkillError, match=message):
        run({"expression": expression})


@pytest.mark.parametrize("arguments", [{}, {"expression": ""}, {"expression": 42}])
def test_requires_expression_string(arguments: dict[str, object]) -> None:
    with pytest.raises(SkillError, match="non-empty string"):
        run(arguments)
