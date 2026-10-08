"""Test doubles for the LLM boundary. Neither of these is a model, and neither pretends to be.

- ``ScriptedLLM`` replays canned responses, for precise unit tests of the agent loop.
- ``OfflineLLM`` is a deterministic stand-in that lets the CLI, evals, and flywheel run with
  no network. It mimics the *shape* of model reasoning (what capability does this task need?
  which offered tool provides it? if none, can I propose learning one?) using fixed rules, so
  offline runs exercise the real discovery -> selection -> learning -> execution ->
  observation plumbing.
- ``FixtureAuthorLLM`` stands in for a model writing a skill. It returns a canned contract and
  implementation for the one fixture skill (``word_count``) so offline runs exercise the real
  sandbox and harness. It cannot write any other skill.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from agentlab.agent.loop import REQUEST_CAPABILITY
from agentlab.learning.plan import PROPOSE_SKILL_PLAN
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

    from agentlab.models import JSONObject, Message, ToolSpec


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
TEXT_STATISTICS = "text_statistics"

_WORD_COUNT = re.compile(r"\bcount (?:the )?words in ['\"](.*)['\"]", re.IGNORECASE)
# Capabilities the offline stand-in knows how to propose learning, as one generic step each.
_LEARNABLE: dict[str, JSONObject] = {
    TEXT_STATISTICS: {
        "summary": "Count the words in a text",
        "capability": TEXT_STATISTICS,
        "new_skill": {
            "name": "word_count",
            "purpose": "Count the whitespace-separated words in a text.",
            "inputs": "text: the text to count",
            "outputs": "count: the number of words",
        },
        "needs_network": False,
        "has_side_effects": False,
    }
}

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
    words = _WORD_COUNT.search(task)
    arguments: JSONObject
    if expression is not None:
        capability, arguments = ARITHMETIC, {"expression": expression}
    elif words is not None:
        capability, arguments = TEXT_STATISTICS, {"text": words.group(1)}
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
    if capability in _LEARNABLE and any(t.name == PROPOSE_SKILL_PLAN.name for t in tools):
        plan = {"goal": f"Provide {capability}", "steps": [_LEARNABLE[capability]]}
        return LLMResponse(
            text=None, tool_calls=(ToolCall(call_id, PROPOSE_SKILL_PLAN.name, plan),)
        )
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
        capability = _call_arguments(messages, result.call_id).get("capability")
        if capability == CURRENT_INFORMATION:
            return (
                "I can't do this yet: it needs current information from the web, and none of my "
                "available skills provide it. I won't guess at details I can't verify."
            )
        return f"I can't do this yet: it needs {capability}, and none of my skills provide it."
    if tool_name == PROPOSE_SKILL_PLAN.name:
        return (
            f"I can't do this: I don't have a skill for it, and learning one didn't work out "
            f"({content.get('outcome')}: {content.get('detail')})."
        )
    if result.is_error:
        return f"The {tool_name} skill reported an error: {content.get('error')}"
    if "result" in content:
        return f"The result is {content['result']}."
    return f"{tool_name} returned {json.dumps(content, sort_keys=True)}"


def _tool_name_for(messages: Sequence[Message], call_id: str) -> str:
    return _call(messages, call_id).name


def _call_arguments(messages: Sequence[Message], call_id: str) -> JSONObject:
    return _call(messages, call_id).arguments


def _call(messages: Sequence[Message], call_id: str) -> ToolCall:
    for message in reversed(messages):
        if isinstance(message, AssistantMessage):
            for call in message.tool_calls:
                if call.id == call_id:
                    return call
    raise LLMError(f"no tool call with id {call_id!r} in the conversation")


_WORD_COUNT_FIXTURE: dict[str, JSONObject] = {
    "submit_contract": {
        "input_schema": {
            "type": "object",
            "required": ["text"],
            "properties": {"text": {"type": "string"}},
        },
        "output_schema": {
            "type": "object",
            "required": ["count"],
            "properties": {"count": {"type": "integer"}},
        },
        "tests": [
            {"name": "two_words", "kind": "normal", "arguments": {"text": "hello world"},
             "expect_output": {"count": 2}},
            {"name": "punctuation", "kind": "normal", "arguments": {"text": "one, two; three"},
             "expect_output": {"count": 3}},
            {"name": "extra_spaces", "kind": "edge", "arguments": {"text": "  spaced   out  "},
             "expect_output": {"count": 2}},
            {"name": "empty", "kind": "edge", "arguments": {"text": ""},
             "expect_output": {"count": 0}},
            {"name": "not_a_string", "kind": "error", "arguments": {"text": 5},
             "expect_error": "string"},
            {"name": "letters", "kind": "normal", "arguments": {"text": "a b c d e f g"},
             "expect_output": {"count": 7}},
        ],
    },
    "submit_skill": {
        "description": "Counts the whitespace-separated words in a text. [offline fixture]",
        "when_to_use": "The task needs the number of words in a given text.",
        "limitations": "Splits on whitespace only; punctuation stays attached to words.",
        "code": (
            "def run(arguments):\n"
            "    text = arguments.get('text')\n"
            "    if not isinstance(text, str):\n"
            "        raise SkillError('text must be a string')\n"
            "    return {'count': len(text.split())}\n"
        ),
    },
}  # fmt: skip
_SKILL_FIXTURES = {"word_count": _WORD_COUNT_FIXTURE}


class FixtureAuthorLLM:
    """Stand-in for a model writing a skill. Fixture data, not intelligence (see module doc).

    It answers whichever submit tool it is offered with the canned data for the requested
    skill, and declines (no tool call) for any skill it has no fixture for.
    """

    def generate(
        self, *, system: str, messages: Sequence[Message], tools: Sequence[ToolSpec]
    ) -> LLMResponse:
        brief = messages[-1]
        if not isinstance(brief, UserMessage) or len(tools) != 1:
            raise LLMError("FixtureAuthorLLM expects one brief and one submit tool")
        name = str(json.loads(brief.text).get("skill", {}).get("name"))
        fixture = _SKILL_FIXTURES.get(name, {}).get(tools[0].name)
        if fixture is None:
            return LLMResponse(text=f"[offline fixture] no canned skill named {name!r}")
        return LLMResponse(text=None, tool_calls=(ToolCall("fixture", tools[0].name, fixture),))
