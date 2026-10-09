"""Plans are checked by deterministic rules before anyone is asked to approve them."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from agentlab.learning.plan import MAX_STEPS, PlanError, parse_plan, validate_plan

if TYPE_CHECKING:
    from agentlab.models import JSONObject
    from agentlab.tools.catalog import ToolCatalog


def new(name: str = "word_count", capability: str = "text_statistics", **flags: Any) -> JSONObject:
    return {
        "summary": f"build {name}",
        "capability": capability,
        "new_tool": {"name": name, "purpose": "p", "inputs": "i", "outputs": "o"},
        **flags,
    }


def reuse(tool: str = "calculator") -> JSONObject:
    return {"summary": f"use {tool}", "capability": "arithmetic", "reuse": tool}


def verdict(catalog: ToolCatalog, *steps: JSONObject) -> tuple[str, str]:
    result = validate_plan(parse_plan({"goal": "g", "steps": list(steps)}), catalog)
    return result.outcome, result.reason


def test_accepts_a_small_generic_plan(catalog: ToolCatalog) -> None:
    assert verdict(catalog, new()) == ("accepted", "")
    assert verdict(catalog, *[reuse()] * (MAX_STEPS - 1), new())[0] == "accepted"


def test_refuses_too_many_steps(catalog: ToolCatalog) -> None:
    outcome, reason = verdict(catalog, *[reuse()] * MAX_STEPS, new())
    assert outcome == "refused_too_large"
    assert "Split the task" in reason


def test_refuses_too_many_new_tools(catalog: ToolCatalog) -> None:
    steps = [new(f"tool_{i}", f"cap_{i}") for i in range(3)]
    assert verdict(catalog, *steps)[0] == "refused_too_large"


@pytest.mark.parametrize("flags", [{"needs_network": True}, {"has_side_effects": True}])
def test_network_and_side_effects_are_learnable(catalog: ToolCatalog, flags: Any) -> None:
    # The user approves the plan; no rule refuses what a tool may touch.
    assert verdict(catalog, new("fetch_page", "web_fetch", **flags)) == ("accepted", "")


def test_prefers_existing_tools(catalog: ToolCatalog) -> None:
    assert verdict(catalog, new("calculator", "math"))[0] == "refused_reuse"
    outcome, reason = verdict(catalog, new("adder", "arithmetic"))
    assert outcome == "refused_reuse"
    assert "'calculator' already provides 'arithmetic'" in reason
    assert verdict(catalog, reuse())[0] == "refused_reuse"


@pytest.mark.parametrize(
    "steps",
    [
        [reuse("teleport"), new()],
        [new("WordCount")],
        [new("x" * 41)],
        [new("dup", "a"), new("dup", "b")],
    ],
)
def test_malformed_plans(catalog: ToolCatalog, steps: list[JSONObject]) -> None:
    assert verdict(catalog, *steps)[0] == "malformed"


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"steps": [new()]}, "'goal'"),
        ({"goal": "g", "steps": []}, "'steps'"),
        ({"goal": "g", "steps": [{**reuse(), "new_tool": new()["new_tool"]}]}, "exactly one"),
        ({"goal": "g", "steps": [{**new(), "new_tool": {"name": "x"}}]}, "non-empty name"),
    ],
)
def test_parse_errors(arguments: JSONObject, message: str) -> None:
    with pytest.raises(PlanError, match=message):
        parse_plan(arguments)


def test_describe_is_readable() -> None:
    plan = parse_plan({"goal": "Count words", "steps": [reuse(), new()]})
    assert plan.describe() == (
        "Goal: Count words\n"
        "  1. use calculator [arithmetic] -> reuse existing tool 'calculator'\n"
        "  2. build word_count [text_statistics] -> build new tool 'word_count': p"
    )
