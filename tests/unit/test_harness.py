"""The runtime eval harness accepts working skills and rejects junk."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from agentlab.learning.author import SkillAuthor
from agentlab.learning.harness import (
    build_skill,
    contract_problems,
    evaluate_candidate,
    schema_errors,
    split_tests,
)
from agentlab.learning.models import SkillSpec
from agentlab.learning.sandbox import SandboxResult
from agentlab.llm.fake import ScriptedLLM
from agentlab.models import LLMResponse, ToolCall, UserMessage

if TYPE_CHECKING:
    from agentlab.learning.models import SkillContract
    from agentlab.models import JSONObject

SPEC = SkillSpec(
    name="word_count",
    capability="text_statistics",
    purpose="Count the words in a piece of text.",
    inputs="text: the text",
    outputs="count: number of whitespace-separated words",
)
MANIFEST_PATH = Path("word_count/skill.json")

TESTS: list[JSONObject] = [
    {"name": "two_words", "kind": "normal", "arguments": {"text": "hello world"},
     "expect_output": {"count": 2}},
    {"name": "punctuation", "kind": "normal", "arguments": {"text": "one, two; three"},
     "expect_output": {"count": 3}},
    {"name": "extra_spaces", "kind": "edge", "arguments": {"text": "  spaced   out  "},
     "expect_output": {"count": 2}},
    {"name": "empty", "kind": "edge", "arguments": {"text": ""}, "expect_output": {"count": 0}},
    {"name": "not_a_string", "kind": "error", "arguments": {"text": 5},
     "expect_error": "string"},
    {"name": "letters", "kind": "normal", "arguments": {"text": "a b c d e f g"},
     "expect_output": {"count": 7}},
]  # fmt: skip
CONTRACT: JSONObject = {
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
    "tests": TESTS,
}

GOOD = """\
def run(arguments):
    text = arguments.get("text")
    if not isinstance(text, str):
        raise SkillError("text must be a string")
    return {"count": len(text.split())}
"""
CONSTANT = "def run(arguments):\n    return {'count': 2}\n"
CRASHES = "def run(arguments):\n    return {'count': len(arguments['text'].split())}\n"
WRONG_SCHEMA = """\
def run(arguments):
    text = arguments.get("text")
    if not isinstance(text, str):
        raise SkillError("text must be a string")
    return {"words": len(text.split())}
"""
HARDCODED = """\
def run(arguments):
    text = arguments.get("text")
    if not isinstance(text, str):
        raise SkillError("text must be a string")
    answers = {"hello world": 2, "one, two; three": 3, "": 0}
    return {"count": answers.get(text, 1)}
"""
UNUSABLE = "def count(arguments):\n    return {'count': 0}\n"


def contract_call(contract: JSONObject = CONTRACT) -> LLMResponse:
    return LLMResponse(text=None, tool_calls=(ToolCall("t", "submit_contract", contract),))


def code_call(code: str) -> LLMResponse:
    skill = {
        "description": "Counts words.",
        "when_to_use": "The task needs the number of words in a text.",
        "limitations": "Splits on whitespace only.",
        "code": code,
    }
    return LLMResponse(text=None, tool_calls=(ToolCall("c", "submit_skill", skill),))


def build(*responses: LLMResponse) -> tuple[ScriptedLLM, SkillAuthor]:
    llm = ScriptedLLM(responses)
    return llm, SkillAuthor(llm)


def contract() -> SkillContract:
    return SkillAuthor(ScriptedLLM([contract_call()])).write_contract(SPEC)


def failed_checks(code: str) -> set[str]:
    _, author = build(code_call(code))
    the_contract = contract()
    visible, holdout = split_tests(the_contract.tests)
    candidate = author.write_code(SPEC, the_contract, visible, [], MANIFEST_PATH)
    report = evaluate_candidate(candidate, the_contract, frozenset(c.id for c in holdout))
    return {name for name, _ in report.failure_signature}


def test_working_skill_is_ready_on_first_attempt() -> None:
    _, author = build(contract_call(), code_call(GOOD))
    report = build_skill(SPEC, author, manifest_path=MANIFEST_PATH)

    assert report.verdict == "ready"
    assert report.attempts == 1
    assert report.candidate is not None
    assert report.candidate.manifest.capabilities == ("text_statistics",)
    assert "6/6 tests passed (2 held out)" in report.summary()


def test_retry_gets_failures_as_feedback_and_can_recover() -> None:
    llm, author = build(contract_call(), code_call(CONSTANT), code_call(GOOD))
    report = build_skill(SPEC, author, manifest_path=MANIFEST_PATH)

    assert report.verdict == "ready"
    assert report.attempts == 2
    retry = llm.calls[2].messages[0]
    assert isinstance(retry, UserMessage)
    feedback = json.loads(retry.text)["previous_attempt_failed"]
    assert any(line.startswith("non_constant:") for line in feedback)


def test_held_out_inputs_never_reach_the_code_writer() -> None:
    llm, author = build(contract_call(), code_call(HARDCODED), code_call(GOOD))
    build_skill(SPEC, author, manifest_path=MANIFEST_PATH)

    for call in llm.calls[1:]:
        brief = call.messages[0]
        assert isinstance(brief, UserMessage)
        assert "spaced   out" not in brief.text
        assert "a b c d e f g" not in brief.text


def test_gives_up_after_max_attempts() -> None:
    _, author = build(
        contract_call(), code_call(CONSTANT), code_call(CRASHES), code_call(WRONG_SCHEMA)
    )
    report = build_skill(SPEC, author, manifest_path=MANIFEST_PATH)

    assert report.verdict == "failed"
    assert report.attempts == 3
    assert report.candidate is None
    assert "still failing after 3 attempts" in report.summary()


def test_stops_early_when_stuck() -> None:
    _, author = build(contract_call(), code_call(CONSTANT), code_call(CONSTANT))
    report = build_skill(SPEC, author, manifest_path=MANIFEST_PATH)

    assert report.verdict == "failed"
    assert report.attempts == 2
    assert "stuck" in report.reason


def test_author_that_does_not_submit_is_a_failed_attempt() -> None:
    no_tool = LLMResponse(text="here is some code...")
    _, author = build(contract_call(), no_tool, no_tool)
    report = build_skill(SPEC, author, manifest_path=MANIFEST_PATH)

    assert report.verdict == "failed"
    assert report.reports[0].checks[0].name == "author"


def test_model_failure_while_authoring_is_a_failed_attempt() -> None:
    _, author = build(contract_call())  # the code call finds the script exhausted: LLMError
    report = build_skill(SPEC, author, manifest_path=MANIFEST_PATH)

    assert report.verdict == "failed"
    assert "the model call failed" in report.reports[0].checks[0].detail


def test_weak_test_suite_is_rejected_before_any_code_is_written() -> None:
    weak = {**CONTRACT, "tests": TESTS[:2]}
    llm, author = build(contract_call(weak), contract_call(weak))
    report = build_skill(SPEC, author, manifest_path=MANIFEST_PATH)

    assert report.verdict == "failed"
    assert "could not write a usable test suite" in report.reason
    assert len(llm.calls) == 2


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        (CONSTANT, {"non_constant", "visible_tests", "holdout_tests"}),
        (CRASHES, {"no_crashes", "visible_tests"}),
        (WRONG_SCHEMA, {"output_schema", "visible_tests", "holdout_tests"}),
        (HARDCODED, {"hardcoded", "holdout_tests"}),
        (UNUSABLE, {"static"}),
    ],
)
def test_junk_is_caught_by_the_right_checks(code: str, expected: set[str]) -> None:
    assert failed_checks(code) == expected


def test_overfit_code_is_called_out() -> None:
    _, author = build(code_call(HARDCODED))
    the_contract = contract()
    visible, holdout = split_tests(the_contract.tests)
    candidate = author.write_code(SPEC, the_contract, visible, [], MANIFEST_PATH)
    report = evaluate_candidate(candidate, the_contract, frozenset(c.id for c in holdout))
    (holdout_check,) = [c for c in report.checks if c.name == "holdout_tests"]
    assert "fitted to the examples" in holdout_check.detail


def test_nondeterminism_is_caught() -> None:
    _, author = build(code_call(GOOD))
    the_contract = contract()
    candidate = author.write_code(SPEC, the_contract, the_contract.tests, [], MANIFEST_PATH)
    outputs = iter(range(1000))

    def flaky(code: str, arguments: JSONObject) -> SandboxResult:
        return SandboxResult(output={"count": next(outputs)})

    report = evaluate_candidate(candidate, the_contract, frozenset(), run=flaky)
    assert ("deterministic", "different results on a second run: two_words, punctuation, "
            "extra_spaces") in report.failure_signature  # fmt: skip


def test_contract_problems() -> None:
    assert contract_problems(contract()) == []
    no_error_case = SkillAuthor(
        ScriptedLLM([contract_call({**CONTRACT, "tests": [*TESTS[:4], TESTS[5], TESTS[1]]})])
    ).write_contract(SPEC)
    problems = contract_problems(no_error_case)
    assert "needs at least one 'error' test" in problems
    assert "test names must be unique" in problems


def test_schema_errors() -> None:
    schema: JSONObject = {
        "type": "object",
        "required": ["n"],
        "additionalProperties": False,
        "properties": {
            "n": {"type": "integer"},
            "tags": {"type": "array", "items": {"type": "string", "enum": ["a", "b"]}},
        },
    }
    assert schema_errors({"n": 1, "tags": ["a"]}, schema) == []
    assert schema_errors({"n": True}, schema) == ["at $.n: expected integer, got boolean"]
    assert schema_errors({"tags": ["c"], "x": 1}, schema) == [
        "at $: missing required 'n'",
        "at $.tags[0]: 'c' is not one of ['a', 'b']",
        "at $: unexpected property 'x'",
    ]
