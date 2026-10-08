"""Ask the model to write a skill: first its contract and tests, then, separately, its code.

The two calls are deliberately independent. The test writer never sees the code, and the
code writer never sees the held-out tests, so a candidate can't pass by mirroring its tests.
Both go through ``LLMClient``, so the author works with Claude or a scripted fake.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, cast

from agentlab.evals.models import EvalCase, Expectations
from agentlab.learning.models import SkillCandidate, SkillContract
from agentlab.learning.sandbox import ALLOWED_MODULES
from agentlab.models import ToolSpec, UserMessage
from agentlab.skills.catalog import manifest_from_data
from agentlab.skills.models import SkillManifestError

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from agentlab.learning.models import SkillSpec
    from agentlab.llm.client import LLMClient
    from agentlab.models import JSONObject, LLMResponse

LEARNED_IMPLEMENTATION = "sandbox:skill.py"
TEST_KINDS = ("normal", "edge", "error")


class AuthorError(Exception):
    """The model did not return a usable contract or candidate. Counts as a failed attempt."""


CONTRACT_PROMPT = """\
You write the contract for a small, generic, pure-Python skill BEFORE anyone implements it.

Return, through submit_contract:
- input_schema and output_schema: JSON Schema objects (type "object"). Keep them small.
- tests: at least 6 cases. Each has a unique snake_case name, a kind ("normal", "edge" or \
"error"), arguments, and exactly one of expect_output (the complete expected output object) \
or expect_error (a short substring of the error message the skill should raise).
  Include at least one "edge" case (empty or boundary input) and at least one "error" case \
(invalid input). Use varied inputs so a hard-coded implementation would fail.
  Only write expected outputs you are certain of. Prefer cases whose answer is unambiguous.
The skill must be generic and reusable: it transforms data. No network, files or side effects."""

CODE_PROMPT = f"""\
You implement a small, generic skill as pure Python that runs in a strict sandbox.

Rules:
- Define a top-level `def run(arguments):` that takes a dict matching the input schema and \
returns a dict matching the output schema exactly.
- For invalid input, `raise SkillError("message")`. SkillError is predefined; don't import it.
- Imports allowed: {", ".join(sorted(ALLOWED_MODULES))}. Nothing else.
- Not allowed: open, eval, exec, getattr, type, dir, str.format (use f-strings), any name or \
attribute starting with "_" (except defining or calling __init__), async, global.
- Solve the general problem. Never special-case the example inputs.

Return, through submit_skill, the code plus a one-sentence description, when_to_use (when a \
model should pick this tool) and limitations (what it cannot do)."""

SUBMIT_CONTRACT = ToolSpec(
    name="submit_contract",
    description="Submit the schemas and tests for the skill.",
    input_schema={
        "type": "object",
        "required": ["input_schema", "output_schema", "tests"],
        "properties": {
            "input_schema": {"type": "object"},
            "output_schema": {"type": "object"},
            "tests": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["name", "kind", "arguments"],
                    "properties": {
                        "name": {"type": "string"},
                        "kind": {"type": "string", "enum": list(TEST_KINDS)},
                        "arguments": {"type": "object"},
                        "expect_output": {"type": "object"},
                        "expect_error": {"type": "string"},
                    },
                },
            },
        },
    },
)

SUBMIT_SKILL = ToolSpec(
    name="submit_skill",
    description="Submit the implementation of the skill.",
    input_schema={
        "type": "object",
        "required": ["description", "when_to_use", "limitations", "code"],
        "properties": {
            "description": {"type": "string"},
            "when_to_use": {"type": "string"},
            "limitations": {"type": "string"},
            "code": {"type": "string"},
        },
    },
)


class SkillAuthor:
    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    def write_contract(self, spec: SkillSpec, feedback: Sequence[str] = ()) -> SkillContract:
        brief: JSONObject = {"skill": spec.to_json()}
        if feedback:
            brief["previous_contract_rejected"] = list(feedback)
        submitted = self._ask(CONTRACT_PROMPT, brief, SUBMIT_CONTRACT)
        return SkillContract(
            input_schema=_object(submitted, "input_schema"),
            output_schema=_object(submitted, "output_schema"),
            tests=tuple(_test_case(spec.name, raw) for raw in _list(submitted, "tests")),
        )

    def write_code(
        self,
        spec: SkillSpec,
        contract: SkillContract,
        visible_tests: Sequence[EvalCase],
        feedback: Sequence[str],
        manifest_path: Path,
    ) -> SkillCandidate:
        brief: JSONObject = {
            "skill": spec.to_json(),
            "input_schema": contract.input_schema,
            "output_schema": contract.output_schema,
            "example_tests": [_test_json(case) for case in visible_tests],
        }
        if feedback:
            brief["previous_attempt_failed"] = list(feedback)
        submitted = self._ask(CODE_PROMPT, brief, SUBMIT_SKILL)
        data: JSONObject = {
            "name": spec.name,
            "description": submitted.get("description"),
            "when_to_use": submitted.get("when_to_use"),
            "limitations": submitted.get("limitations"),
            "capabilities": [spec.capability],
            "input_schema": contract.input_schema,
            "output_schema": contract.output_schema,
            "implementation": LEARNED_IMPLEMENTATION,
        }
        try:
            manifest = manifest_from_data(data, manifest_path)
        except SkillManifestError as exc:
            raise AuthorError(str(exc)) from exc
        code = submitted.get("code")
        if not isinstance(code, str) or not code.strip():
            raise AuthorError("submit_skill had no code")
        return SkillCandidate(manifest, code)

    def _ask(self, system: str, brief: JSONObject, tool: ToolSpec) -> JSONObject:
        response = self._llm.generate(
            system=system,
            messages=[UserMessage(json.dumps(brief, indent=2))],
            tools=[tool],
        )
        return _submitted(response, tool)


def _submitted(response: LLMResponse, tool: ToolSpec) -> JSONObject:
    for call in response.tool_calls:
        if call.name == tool.name:
            return call.arguments
    raise AuthorError(f"the model did not call {tool.name}")


def _test_case(skill: str, raw: object) -> EvalCase:
    if not isinstance(raw, dict):
        raise AuthorError("each test must be an object")
    test = cast("dict[str, Any]", raw)
    name, kind, arguments = test.get("name"), test.get("kind"), test.get("arguments")
    output, error = test.get("expect_output"), test.get("expect_error")
    if not isinstance(name, str) or not name or kind not in TEST_KINDS:
        raise AuthorError(f"test {name!r}: needs a name and a kind in {TEST_KINDS}")
    if not isinstance(arguments, dict):
        raise AuthorError(f"test {name!r}: arguments must be an object")
    if (output is None) == (error is None):
        raise AuthorError(f"test {name!r}: needs exactly one of expect_output or expect_error")
    if output is not None and not isinstance(output, dict):
        raise AuthorError(f"test {name!r}: expect_output must be an object")
    if error is not None and (not isinstance(error, str) or not error):
        raise AuthorError(f"test {name!r}: expect_error must be a non-empty string")
    return EvalCase(
        id=name,
        kind="skill",
        description=f"generated {kind} test",
        expect=Expectations(output_equals=cast("JSONObject | None", output), error_contains=error),
        tags=(str(kind),),
        skill=skill,
        arguments=cast("JSONObject", arguments),
    )


def _test_json(case: EvalCase) -> JSONObject:
    test: JSONObject = {"name": case.id, "arguments": case.arguments}
    if case.expect.output_equals is not None:
        test["expect_output"] = case.expect.output_equals
    if case.expect.error_contains is not None:
        test["expect_error"] = case.expect.error_contains
    return test


def _object(data: JSONObject, key: str) -> JSONObject:
    value = data.get(key)
    if not isinstance(value, dict):
        raise AuthorError(f"'{key}' must be an object")
    return cast("JSONObject", value)


def _list(data: JSONObject, key: str) -> list[object]:
    value = data.get(key)
    if not isinstance(value, list):
        raise AuthorError(f"'{key}' must be a list")
    return cast("list[object]", value)
