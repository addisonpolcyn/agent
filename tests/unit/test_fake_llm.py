from __future__ import annotations

import pytest

from agentlab.agent.loop import REQUEST_CAPABILITY
from agentlab.llm.client import LLMError
from agentlab.llm.fake import OfflineLLM, ScriptedLLM
from agentlab.models import (
    AssistantMessage,
    LLMResponse,
    ToolCall,
    ToolResultMessage,
    ToolSpec,
    UserMessage,
)

CALC = ToolSpec("calc", "Adds things.\nProvides: arithmetic", {"type": "object"})


def plan(task: str, tools: list[ToolSpec]) -> LLMResponse:
    return OfflineLLM().generate(system="", messages=[UserMessage(task)], tools=tools)


def test_scripted_llm_raises_when_exhausted() -> None:
    llm = ScriptedLLM([LLMResponse(text="one")])
    llm.generate(system="", messages=[], tools=[])
    with pytest.raises(LLMError, match="only 1 scripted"):
        llm.generate(system="", messages=[], tools=[])


def test_offline_selects_tool_by_declared_capability_not_name() -> None:
    response = plan("What is (2 + 3) * 4?", [CALC, REQUEST_CAPABILITY])
    (tool_call,) = response.tool_calls
    assert tool_call.name == "calc"
    assert tool_call.arguments == {"expression": "(2 + 3) * 4"}


def test_offline_requests_capability_when_no_tool_provides_it() -> None:
    response = plan("What is 2 + 2?", [REQUEST_CAPABILITY])
    (tool_call,) = response.tool_calls
    assert tool_call.name == REQUEST_CAPABILITY.name
    assert tool_call.arguments["capability"] == "arithmetic"


def test_offline_recognizes_need_for_current_information() -> None:
    response = plan("Find the latest flights to Tokyo", [CALC, REQUEST_CAPABILITY])
    (tool_call,) = response.tool_calls
    assert tool_call.arguments["capability"] == "current_information"


def test_offline_answers_without_tools_when_none_needed() -> None:
    response = plan("Say hello", [CALC, REQUEST_CAPABILITY])
    assert response.tool_calls == ()
    assert response.text is not None
    assert "[offline]" in response.text


def test_offline_answers_from_observation() -> None:
    messages = [
        UserMessage("What is 2 + 2?"),
        AssistantMessage(None, (ToolCall("c1", "calc", {"expression": "2 + 2"}),)),
        ToolResultMessage("c1", {"result": 4}),
    ]
    response = OfflineLLM().generate(system="", messages=messages, tools=[CALC])
    assert response.text == "The result is 4."


def test_offline_reports_tool_errors() -> None:
    messages = [
        UserMessage("1 / 0"),
        AssistantMessage(None, (ToolCall("c1", "calc", {"expression": "1 / 0"}),)),
        ToolResultMessage("c1", {"error": "division by zero"}, is_error=True),
    ]
    response = OfflineLLM().generate(system="", messages=messages, tools=[CALC])
    assert response.text == "The calc tool reported an error: division by zero"
