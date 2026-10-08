"""State transitions of the agent loop, driven by scripted model responses."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from agentlab.agent.loop import REQUEST_CAPABILITY, SYSTEM_PROMPT, Agent
from agentlab.llm.fake import ScriptedLLM
from agentlab.models import AssistantMessage, LLMResponse, ToolCall, ToolResultMessage, UserMessage
from agentlab.skills.catalog import SkillCatalog

if TYPE_CHECKING:
    from agentlab.models import JSONObject


def call(name: str, arguments: JSONObject, call_id: str = "c1") -> LLMResponse:
    return LLMResponse(text=None, tool_calls=(ToolCall(call_id, name, arguments),))


def answer(text: str) -> LLMResponse:
    return LLMResponse(text=text)


def test_direct_answer_takes_one_step(catalog: SkillCatalog) -> None:
    llm = ScriptedLLM([answer("hello")])
    run = Agent(llm, catalog).run("Say hello")

    assert run.answer == "hello"
    assert run.stop_reason == "answered"
    assert run.steps == 1
    assert run.skills_used == ()
    (first,) = llm.calls
    assert first.system == SYSTEM_PROMPT
    assert first.messages == (UserMessage("Say hello"),)


def test_offers_catalog_skills_plus_request_capability(catalog: SkillCatalog) -> None:
    llm = ScriptedLLM([answer("ok")])
    Agent(llm, catalog).run("anything")
    assert [t.name for t in llm.calls[0].tools] == ["calculator", REQUEST_CAPABILITY.name]


def test_tool_call_then_answer(catalog: SkillCatalog) -> None:
    llm = ScriptedLLM([call("calculator", {"expression": "120 * 3"}), answer("360")])
    run = Agent(llm, catalog).run("What is 120 * 3?")

    assert run.answer == "360"
    assert run.steps == 2
    assert run.skills_used == ("calculator",)
    assert run.invocations[0].result.output == {"result": 360}
    second = llm.calls[1].messages
    assert isinstance(second[1], AssistantMessage)
    assert second[2] == ToolResultMessage("c1", {"result": 360}, is_error=False)


def test_skill_error_is_fed_back_as_error_observation(catalog: SkillCatalog) -> None:
    llm = ScriptedLLM([call("calculator", {"expression": "1 / 0"}), answer("can't")])
    run = Agent(llm, catalog).run("1/0?")

    assert run.invocations[0].result.error == "division by zero"
    assert llm.calls[1].messages[-1] == ToolResultMessage(
        "c1", {"error": "division by zero"}, is_error=True
    )


def test_unknown_tool_is_reported_not_raised(catalog: SkillCatalog) -> None:
    llm = ScriptedLLM([call("teleport", {}), answer("sorry")])
    run = Agent(llm, catalog).run("beam me up")

    assert run.answer == "sorry"
    observation = llm.calls[1].messages[-1]
    assert isinstance(observation, ToolResultMessage)
    assert observation.is_error


def test_capability_gap_is_recorded(catalog: SkillCatalog) -> None:
    gap = {"capability": "current_information", "reason": "needs live flights"}
    llm = ScriptedLLM([call(REQUEST_CAPABILITY.name, gap), answer("I can't search the web.")])
    run = Agent(llm, catalog).run("Find flights")

    assert run.skills_used == ()
    assert [(g.capability, g.reason) for g in run.capability_gaps] == [
        ("current_information", "needs live flights")
    ]
    observation = llm.calls[1].messages[-1]
    assert isinstance(observation, ToolResultMessage)
    assert observation.content["available"] is False


def test_multiple_tool_calls_in_one_turn(catalog: SkillCatalog) -> None:
    turn = LLMResponse(
        text=None,
        tool_calls=(
            ToolCall("a", "calculator", {"expression": "1 + 1"}),
            ToolCall("b", "calculator", {"expression": "2 + 2"}),
        ),
    )
    llm = ScriptedLLM([turn, answer("2 and 4")])
    run = Agent(llm, catalog).run("two sums")

    assert [i.result.output for i in run.invocations] == [{"result": 2}, {"result": 4}]
    assert [m.call_id for m in llm.calls[1].messages if isinstance(m, ToolResultMessage)] == [
        "a",
        "b",
    ]


def test_stops_after_max_steps(catalog: SkillCatalog) -> None:
    llm = ScriptedLLM([call("calculator", {"expression": "1 + 1"}, f"c{i}") for i in range(3)])
    run = Agent(llm, catalog, max_steps=3).run("loop forever")

    assert run.stop_reason == "max_steps"
    assert run.answer is None
    assert run.steps == 3
    assert len(run.invocations) == 3


def test_reserved_skill_name_is_rejected(catalog: SkillCatalog) -> None:
    class Reserved(SkillCatalog):
        def __contains__(self, name: object) -> bool:
            return name == REQUEST_CAPABILITY.name

    with pytest.raises(ValueError, match="reserved"):
        Agent(ScriptedLLM([]), Reserved({}))
