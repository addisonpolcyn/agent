"""Plans are checked by deterministic rules before anyone is asked to approve them."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from agentlab.learning.plan import MAX_STEPS, PlanError, parse_plan, validate_plan

if TYPE_CHECKING:
    from agentlab.models import JSONObject
    from agentlab.skills.catalog import SkillCatalog


def new(name: str = "word_count", capability: str = "text_statistics", **flags: Any) -> JSONObject:
    return {
        "summary": f"build {name}",
        "capability": capability,
        "new_skill": {"name": name, "purpose": "p", "inputs": "i", "outputs": "o"},
        "needs_network": flags.get("network", False),
        "has_side_effects": flags.get("side_effects", False),
    }


def reuse(skill: str = "calculator", **flags: Any) -> JSONObject:
    return {
        "summary": f"use {skill}",
        "capability": "arithmetic",
        "reuse": skill,
        "needs_network": flags.get("network", False),
        "has_side_effects": False,
    }


def verdict(catalog: SkillCatalog, *steps: JSONObject) -> tuple[str, str]:
    result = validate_plan(parse_plan({"goal": "g", "steps": list(steps)}), catalog)
    return result.outcome, result.reason


def test_accepts_a_small_generic_plan(catalog: SkillCatalog) -> None:
    assert verdict(catalog, new()) == ("accepted", "")
    assert verdict(catalog, *[reuse()] * (MAX_STEPS - 1), new())[0] == "accepted"


def test_refuses_too_many_steps(catalog: SkillCatalog) -> None:
    outcome, reason = verdict(catalog, *[reuse()] * MAX_STEPS, new())
    assert outcome == "refused_too_large"
    assert "Split the task" in reason


def test_refuses_too_many_new_skills(catalog: SkillCatalog) -> None:
    steps = [new(f"skill_{i}", f"cap_{i}") for i in range(3)]
    assert verdict(catalog, *steps)[0] == "refused_too_large"


@pytest.mark.parametrize("flags", [{"network": True}, {"side_effects": True}])
def test_refuses_learning_network_or_side_effects(catalog: SkillCatalog, flags: Any) -> None:
    outcome, reason = verdict(catalog, new("fetch_page", "web_fetch", **flags))
    assert outcome == "refused_not_learnable"
    assert "web_fetch" in reason


def test_reusing_an_existing_network_step_is_fine(catalog: SkillCatalog) -> None:
    # Only learned code is limited; a trusted skill may do what learned code can't.
    assert verdict(catalog, reuse(network=True), new())[0] == "accepted"


def test_prefers_existing_skills(catalog: SkillCatalog) -> None:
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
def test_malformed_plans(catalog: SkillCatalog, steps: list[JSONObject]) -> None:
    assert verdict(catalog, *steps)[0] == "malformed"


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"steps": [new()]}, "'goal'"),
        ({"goal": "g", "steps": []}, "'steps'"),
        ({"goal": "g", "steps": [{**reuse(), "new_skill": new()["new_skill"]}]}, "exactly one"),
        ({"goal": "g", "steps": [{**new(), "new_skill": {"name": "x"}}]}, "non-empty name"),
    ],
)
def test_parse_errors(arguments: JSONObject, message: str) -> None:
    with pytest.raises(PlanError, match=message):
        parse_plan(arguments)


def test_describe_is_readable() -> None:
    plan = parse_plan({"goal": "Count words", "steps": [reuse(), new()]})
    assert plan.describe() == (
        "Goal: Count words\n"
        "  1. use calculator [arithmetic] -> reuse existing skill 'calculator'\n"
        "  2. build word_count [text_statistics] -> build new skill 'word_count': p"
    )
