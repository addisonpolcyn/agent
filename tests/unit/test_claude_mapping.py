"""The Claude adapter's conversions, tested without the network."""

from __future__ import annotations

from typing import Any

import pytest
from anthropic.types.beta import BetaMessage

from agentlab.llm.claude import (
    from_anthropic_response,
    to_anthropic_messages,
    to_anthropic_tool,
)
from agentlab.llm.client import LLMError
from agentlab.models import AssistantMessage, ToolCall, ToolResultMessage, ToolSpec, UserMessage


def message(content: list[dict[str, Any]], stop_reason: str = "end_turn") -> BetaMessage:
    return BetaMessage.model_validate(
        {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-opus-5-5",
            "content": content,
            "stop_reason": stop_reason,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
    )


def test_tool_spec_maps_to_tool_param() -> None:
    spec = ToolSpec("calc", "adds", {"type": "object"})
    assert to_anthropic_tool(spec) == {
        "name": "calc",
        "description": "adds",
        "input_schema": {"type": "object"},
    }


def test_response_maps_text_and_tool_calls() -> None:
    response = from_anthropic_response(
        message(
            [
                {"type": "thinking", "thinking": "", "signature": "sig"},
                {"type": "text", "text": "Let me compute."},
                {"type": "tool_use", "id": "tu_1", "name": "calc", "input": {"expression": "1+1"}},
            ],
            stop_reason="tool_use",
        )
    )
    assert response.text == "Let me compute."
    assert response.tool_calls == (ToolCall("tu_1", "calc", {"expression": "1+1"}),)


@pytest.mark.parametrize(
    ("stop_reason", "error"), [("refusal", "declined"), ("max_tokens", "cut off")]
)
def test_unusable_stop_reasons_raise(stop_reason: str, error: str) -> None:
    with pytest.raises(LLMError, match=error):
        from_anthropic_response(message([{"type": "text", "text": "..."}], stop_reason))


def test_assistant_turn_replays_raw_content_including_thinking() -> None:
    response = from_anthropic_response(
        message(
            [
                {"type": "thinking", "thinking": "", "signature": "sig"},
                {"type": "tool_use", "id": "tu_1", "name": "calc", "input": {}},
            ],
            stop_reason="tool_use",
        )
    )
    converted = to_anthropic_messages([UserMessage("hi"), response.as_message()])
    content = list(converted[1]["content"])
    assert [getattr(block, "type", None) for block in content] == ["thinking", "tool_use"]


def test_consecutive_tool_results_share_one_user_message() -> None:
    converted = to_anthropic_messages(
        [
            UserMessage("two sums"),
            AssistantMessage(
                None, (ToolCall("a", "calc", {"e": "1+1"}), ToolCall("b", "calc", {"e": "2+2"}))
            ),
            ToolResultMessage("a", {"result": 2}),
            ToolResultMessage("b", {"error": "boom"}, is_error=True),
            UserMessage("thanks"),
        ]
    )
    assert [m["role"] for m in converted] == ["user", "assistant", "user", "user"]
    assert converted[1]["content"] == [
        {"type": "tool_use", "id": "a", "name": "calc", "input": {"e": "1+1"}},
        {"type": "tool_use", "id": "b", "name": "calc", "input": {"e": "2+2"}},
    ]
    assert converted[2]["content"] == [
        {"type": "tool_result", "tool_use_id": "a", "content": '{"result": 2}', "is_error": False},
        {
            "type": "tool_result",
            "tool_use_id": "b",
            "content": '{"error": "boom"}',
            "is_error": True,
        },
    ]
