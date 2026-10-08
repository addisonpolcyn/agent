"""Test doubles for the LLM boundary. Neither of these is a model, and neither pretends to be.

- ``ScriptedLLM`` replays canned responses, for precise unit tests of the agent loop.
- ``OfflineLLM`` is a deterministic stand-in that lets the CLI, evals, and flywheel run with
  no network. It mimics the *shape* of model reasoning (what capability does this task need?
  which offered tool provides it?) using fixed rules, so offline runs exercise the real
  discovery -> selection -> execution -> observation plumbing.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from agentlab.agent.loop import REQUEST_CAPABILITY
from agentlab.llm.client import LLMError
from agentlab.models import (
    AssistantMessage,
    LLMResponse,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from agentlab.skills.catalog import PROVIDES_PREFIX

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from agentlab.models import Message, ToolSpec


@dataclass(frozen=True)
class GenerateCall:
    system: str
    messages: tuple[Message, ...]
    tools: tuple[ToolSpec, ...]


class ScriptedLLM:
    """Returns the given responses in order and records every call it receives."""

    def __init__(self, responses: Iterable[LLMResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[GenerateCall] = []

    def generate(
        self, *, system: str, messages: Sequence[Message], tools: Sequence[ToolSpec]
    ) -> LLMResponse:
        self.calls.append(GenerateCall(system, tuple(messages), tuple(tools)))
        if len(self.calls) > len(self._responses):
            raise LLMError(f"ScriptedLLM has only {len(self._responses)} scripted responses")
        return self._responses[len(self.calls) - 1]


ARITHMETIC = "arithmetic"
CURRENT_INFORMATION = "current_information"

_EXPRESSION = re.compile(r"[\d(][\d\s.()+\-*/%]*[\d)]")
_OPERATOR = re.compile(r"[+\-*/%]")
_FRESHNESS = re.compile(
    r"\b(latest|current(ly)?|today|tonight|tomorrow|now|recent|upcoming|live|"
    r"this (week|month|year)|next (week|month|year))\b",
    re.IGNORECASE,
)


class OfflineLLM:
    """Deterministic, rule-based stand-in for a model. Not intelligence; see module docstring."""

    def generate(
        self, *, system: str, messages: Sequence[Message], tools: Sequence[ToolSpec]
    ) -> LLMResponse:
        last = messages[-1]
        if isinstance(last, ToolResultMessage):
            return LLMResponse(text=_answer_from_observation(messages, last))
        if not isinstance(last, UserMessage):
            raise LLMError("OfflineLLM expects the conversation to end with a user or tool turn")
        return _plan(last.text, tools, call_id=f"offline-{len(messages)}")


def _plan(task: str, tools: Sequence[ToolSpec], call_id: str) -> LLMResponse:
    expression = _find_expression(task)
    if expression is not None:
        capability, arguments = ARITHMETIC, {"expression": expression}
    elif _FRESHNESS.search(task):
        capability, arguments = CURRENT_INFORMATION, {}
    else:
        return LLMResponse(
            text="[offline] This task needs no tools, and the offline stand-in model "
            "cannot write free-form answers. Run without --offline to use Claude."
        )

    tool = _tool_providing(capability, tools)
    if tool is not None:
        return LLMResponse(text=None, tool_calls=(ToolCall(call_id, tool.name, arguments),))
    gap = {"capability": capability, "reason": f"The task requires {capability}."}
    return LLMResponse(text=None, tool_calls=(ToolCall(call_id, REQUEST_CAPABILITY.name, gap),))


def _find_expression(task: str) -> str | None:
    for candidate in _EXPRESSION.findall(task):
        if _OPERATOR.search(candidate):
            return candidate.strip()
    return None


def _tool_providing(capability: str, tools: Sequence[ToolSpec]) -> ToolSpec | None:
    for tool in tools:
        for line in tool.description.splitlines():
            if line.startswith(PROVIDES_PREFIX):
                provided = {c.strip() for c in line.removeprefix(PROVIDES_PREFIX).split(",")}
                if capability in provided:
                    return tool
    return None


def _answer_from_observation(messages: Sequence[Message], result: ToolResultMessage) -> str:
    tool_name = _tool_name_for(messages, result.call_id)
    content = result.content
    if tool_name == REQUEST_CAPABILITY.name:
        return (
            "I can't do this yet: it needs current information from the web, and none of my "
            "available skills provide it. I won't guess at details I can't verify."
        )
    if result.is_error:
        return f"The {tool_name} skill reported an error: {content.get('error')}"
    if "result" in content:
        return f"The result is {content['result']}."
    return f"{tool_name} returned {json.dumps(content, sort_keys=True)}"


def _tool_name_for(messages: Sequence[Message], call_id: str) -> str:
    for message in reversed(messages):
        if isinstance(message, AssistantMessage):
            for call in message.tool_calls:
                if call.id == call_id:
                    return call.name
    raise LLMError(f"no tool call with id {call_id!r} in the conversation")
