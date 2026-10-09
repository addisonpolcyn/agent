"""Test doubles for the LLM boundary. Neither of these is a model, and neither pretends to be.

- ``ScriptedLLM`` replays canned responses, for precise unit tests of the agent loop.
- ``OfflineLLM`` is a deterministic stand-in that lets the CLI, evals, and flywheel run with
  no network. It mimics the *shape* of model reasoning (what capability does this task need?
  which offered tool provides it? if none, can I propose learning one?) using fixed rules, so
  offline runs exercise the real discovery -> selection -> learning -> execution ->
  observation plumbing.
- ``FixtureAuthorLLM`` stands in for a model writing a skill. It returns a canned contract and
  implementation for a few fixture skills (``word_count``, ``list_files``, ``sha256_hex``) so
  offline runs exercise the real skill process and harness. It cannot write any other skill.
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
DIRECTORY_LISTING = "directory_listing"
HASHING = "hashing"

_LIST_FILES = re.compile(
    r"\blist (?:the )?files in (?:the )?([^\s?]+?)[.?]?(?:\s|$)", re.IGNORECASE
)

_WORD_COUNT = re.compile(r"\bcount (?:the )?words in ['\"](.*)['\"]", re.IGNORECASE)
_SHA256 = re.compile(r"\bsha-?256 (?:hash )?of ['\"](.*)['\"]", re.IGNORECASE)
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
    },
    DIRECTORY_LISTING: {
        "summary": "List the files in a local folder",
        "capability": DIRECTORY_LISTING,
        "new_skill": {
            "name": "list_files",
            "purpose": "List the names and types of the entries in a local folder.",
            "inputs": "path: the folder",
            "outputs": "entries: name and type of each entry",
        },
        "reads_files": True,
    },
    HASHING: {
        "summary": "Hash a text with SHA-256",
        "capability": HASHING,
        "new_skill": {
            "name": "sha256_hex",
            "purpose": "Compute the SHA-256 digest of a text (UTF-8), as hex.",
            "inputs": "text: the text to hash",
            "outputs": "hex: the digest as 64 lowercase hex characters",
        },
    },
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
    listing = _LIST_FILES.search(task)
    digest = _SHA256.search(task)
    arguments: JSONObject
    if expression is not None:
        capability, arguments = ARITHMETIC, {"expression": expression}
    elif listing is not None:
        capability, arguments = DIRECTORY_LISTING, {"path": listing.group(1)}
    elif words is not None:
        capability, arguments = TEXT_STATISTICS, {"text": words.group(1)}
    elif digest is not None:
        capability, arguments = HASHING, {"text": digest.group(1)}
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
_ENTRIES_SCHEMA: JSONObject = {
    "type": "object",
    "required": ["entries"],
    "properties": {
        "entries": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["name", "type"],
                "properties": {"name": {"type": "string"}, "type": {"type": "string"}},
            },
        }
    },
}
_LIST_FILES_FIXTURE: dict[str, JSONObject] = {
    "submit_contract": {
        "input_schema": {
            "type": "object",
            "required": ["path"],
            "properties": {"path": {"type": "string"}},
        },
        "output_schema": _ENTRIES_SCHEMA,
        "tests": [
            {"name": "two_files", "kind": "normal", "arguments": {"path": "{root}"},
             "files": {"a.txt": "x", "b.txt": "y"},
             "expect_output": {"entries": [{"name": "a.txt", "type": "file"},
                                           {"name": "b.txt", "type": "file"}]}},
            {"name": "folder_and_file", "kind": "normal", "arguments": {"path": "{root}"},
             "files": {"docs/readme.md": "hi", "top.txt": "t"},
             "expect_output": {"entries": [{"name": "docs", "type": "dir"},
                                           {"name": "top.txt", "type": "file"}]}},
            {"name": "subfolder", "kind": "edge", "arguments": {"path": "{root}/docs"},
             "files": {"docs/readme.md": "hi"},
             "expect_output": {"entries": [{"name": "readme.md", "type": "file"}]}},
            {"name": "empty_folder", "kind": "edge", "arguments": {"path": "{root}"},
             "expect_output": {"entries": []}},
            {"name": "missing", "kind": "error", "arguments": {"path": "{root}/nope"},
             "expect_error": "no such file"},
            {"name": "not_a_string", "kind": "error", "arguments": {"path": 5},
             "expect_error": "path"},
        ],
    },
    "submit_skill": {
        "description": "Lists the entries of a local folder. [offline fixture]",
        "when_to_use": "The user asks what is in a local folder.",
        "limitations": "One level only; names and types, no contents.",
        "code": (
            "def run(arguments):\n"
            "    path = arguments.get('path')\n"
            "    if not isinstance(path, str) or not path:\n"
            "        raise SkillError('path must be a non-empty string')\n"
            "    entries = list_dir(path)\n"
            "    return {'entries': [{'name': e['name'], 'type': e['type']} for e in entries]}\n"
        ),
    },
}  # fmt: skip
# Uses hashlib, outside the old pure-data import allowlist: learned skills may import anything.
_SHA256_FIXTURE: dict[str, JSONObject] = {
    "submit_contract": {
        "input_schema": {
            "type": "object",
            "required": ["text"],
            "properties": {"text": {"type": "string"}},
        },
        "output_schema": {
            "type": "object",
            "required": ["hex"],
            "properties": {"hex": {"type": "string"}},
        },
        "tests": [
            {"name": "abc", "kind": "normal", "arguments": {"text": "abc"},
             "expect_output": {"hex": "ba7816bf8f01cfea414140de5dae2223"
                                      "b00361a396177a9cb410ff61f20015ad"}},
            {"name": "sentence", "kind": "normal", "arguments": {"text": "The quick brown fox"},
             "expect_output": {"hex": "5cac4f980fedc3d3f1f99b4be3472c9b"
                                      "30d56523e632d151237ec9309048bda9"}},
            {"name": "unicode", "kind": "normal", "arguments": {"text": "été"},
             "expect_output": {"hex": "bd010c64132bf5cae8aea89f67625157"
                                      "27dcf68a5dd1de813c87f50a16c4513c"}},
            {"name": "empty", "kind": "edge", "arguments": {"text": ""},
             "expect_output": {"hex": "e3b0c44298fc1c149afbf4c8996fb924"
                                      "27ae41e4649b934ca495991b7852b855"}},
            {"name": "spaces", "kind": "edge", "arguments": {"text": "  "},
             "expect_output": {"hex": "6c179f21e6f62b629055d8ab40f454ed"
                                      "02e48b68563913473b857d3638e23b28"}},
            {"name": "not_a_string", "kind": "error", "arguments": {"text": 5},
             "expect_error": "string"},
        ],
    },
    "submit_skill": {
        "description": "Computes the SHA-256 digest of a text as hex. [offline fixture]",
        "when_to_use": "The task needs a SHA-256 hash or checksum of a given text.",
        "limitations": "Text only, encoded as UTF-8; no files or other algorithms.",
        "code": (
            "import hashlib\n\n"
            "def run(arguments):\n"
            "    text = arguments.get('text')\n"
            "    if not isinstance(text, str):\n"
            "        raise SkillError('text must be a string')\n"
            "    return {'hex': hashlib.sha256(text.encode('utf-8')).hexdigest()}\n"
        ),
    },
}  # fmt: skip
_SKILL_FIXTURES = {
    "word_count": _WORD_COUNT_FIXTURE,
    "list_files": _LIST_FILES_FIXTURE,
    "sha256_hex": _SHA256_FIXTURE,
}


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
