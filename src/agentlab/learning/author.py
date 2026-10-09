"""Ask the model to write a tool: first its contract and tests, then, separately, its code.

The two calls are deliberately independent. The test writer never sees the code, and the
code writer never sees the held-out tests, so a candidate can't pass by mirroring its tests.
Both go through ``LLMClient``, so the author works with Claude or a scripted fake.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, cast

from agentlab.evals.models import EvalCase, Expectations
from agentlab.learning.models import ToolCandidate, ToolContract
from agentlab.llm.client import LLMError
from agentlab.models import ToolSpec, UserMessage
from agentlab.tools.catalog import manifest_from_data
from agentlab.tools.models import ToolManifestError

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from agentlab.learning.models import NewToolSpec
    from agentlab.llm.client import LLMClient
    from agentlab.models import JSONObject, LLMResponse

LEARNED_IMPLEMENTATION = "sandbox:tool.py"
TEST_KINDS = ("normal", "edge", "error")


class AuthorError(Exception):
    """The model did not return a usable contract or candidate. Counts as a failed attempt."""


CONTRACT_PROMPT = """\
You write the contract for a small, generic Python tool BEFORE anyone implements it.

Return, through submit_contract:
- input_schema and output_schema: JSON Schema objects (type "object"). Keep them small.
- tests: 6 to 12 cases with short inputs. Each has a unique snake_case name, a kind \
("normal", "edge" or "error"), arguments, and exactly one of expect_output (the complete \
expected output object) or expect_error (a short substring of the error message the tool \
should raise).
  Include at least one "edge" case (empty or boundary input) and at least one "error" case \
(invalid input). Use varied inputs so a hard-coded implementation would fail.
  Only write expected outputs you are certain of. Prefer cases whose answer is unambiguous.
The tool must be generic and reusable. Values particular to the user's request (places, \
dates, accounts, hosts, URLs, paths, search terms) are inputs, never constants or defaults; \
vary them across tests. It may use any library, the network, files or \
commands, but tests must be deterministic: if the real output depends on this machine, the \
clock or live data, give the tool an input that supplies that data (raw text to parse, or a \
path to a file the test provides, with the real source as the default) and test through it.

If the tool reads files ("reads_files": true), each test also has "files": an object mapping \
relative paths to text contents (e.g. {"notes/a.txt": "hello"}). The harness creates them in a \
fresh folder and replaces the literal "{root}" in argument strings with that folder's path, so \
arguments look like {"path": "{root}/notes"}. Expected outputs must not contain absolute \
paths: use names or paths relative to the input. The tool reads through functions whose \
errors contain these phrases, so expected errors for such cases should use them: \
"no such file or directory", "not a directory", "not a file"."""

CODE_PROMPT = """\
You implement a small, generic tool in Python. It runs in its own process on the user's \
machine, with the user's environment and working directory.

Rules:
- Define a top-level `def run(arguments):` that takes a dict matching the input schema and \
returns a dict matching the output schema exactly.
- For invalid input, `raise ToolError("message")`. ToolError is predefined; don't import it.
- Any import, builtin or library installed in this Python is available, including os, \
subprocess and urllib. Prefer the standard library: other packages may not be installed.
- Two file helpers are predefined (don't import anything for them): \
`list_dir(path)` returns a sorted list of entries, each with keys name, type ("file", \
"dir", "symlink" or "other") and size (files only), \
and `read_text(path)` returns a file's text (first 1 MB). Both raise \
ToolError for missing paths, list_dir also when the path is not a folder, and read_text \
when it is not a file; let those propagate.
- Solve the general problem. Never special-case the example inputs.
- Never embed request-specific values (hosts, accounts, URLs, paths, dates, search terms) \
as constants or defaults: take them as arguments.
- Keep it compact: well under 150 lines. A small tool that handles the common cases well \
beats a large one.

Return, through submit_tool, the code plus a one-sentence description, when_to_use (when a \
model should pick this tool) and limitations (what it cannot do)."""

SUBMIT_CONTRACT = ToolSpec(
    name="submit_contract",
    description="Submit the schemas and tests for the tool.",
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
                        "files": {
                            "type": "object",
                            "additionalProperties": {"type": "string"},
                        },
                    },
                },
            },
        },
    },
)

SUBMIT_TOOL = ToolSpec(
    name="submit_tool",
    description="Submit the implementation of the tool.",
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


class ToolAuthor:
    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    def write_contract(self, spec: NewToolSpec, feedback: Sequence[str] = ()) -> ToolContract:
        brief: JSONObject = {"tool": spec.to_json()}
        if feedback:
            brief["previous_contract_rejected"] = list(feedback)
        submitted = self._ask(CONTRACT_PROMPT, brief, SUBMIT_CONTRACT)
        return ToolContract(
            input_schema=_object(submitted, "input_schema"),
            output_schema=_object(submitted, "output_schema"),
            tests=tuple(_test_case(spec.name, raw) for raw in _list(submitted, "tests")),
        )

    def write_code(
        self,
        spec: NewToolSpec,
        contract: ToolContract,
        visible_tests: Sequence[EvalCase],
        feedback: Sequence[str],
        manifest_path: Path,
    ) -> ToolCandidate:
        brief: JSONObject = {
            "tool": spec.to_json(),
            "input_schema": contract.input_schema,
            "output_schema": contract.output_schema,
            "example_tests": [_test_json(case) for case in visible_tests],
        }
        if feedback:
            brief["previous_attempt_failed"] = list(feedback)
        submitted = self._ask(CODE_PROMPT, brief, SUBMIT_TOOL)
        data: JSONObject = {
            "name": spec.name,
            "description": submitted.get("description"),
            "when_to_use": submitted.get("when_to_use"),
            "limitations": submitted.get("limitations"),
            "capabilities": [spec.capability],
            "input_schema": contract.input_schema,
            "output_schema": contract.output_schema,
            "implementation": LEARNED_IMPLEMENTATION,
            "reads_files": spec.reads_files,
        }
        try:
            manifest = manifest_from_data(data, manifest_path)
        except ToolManifestError as exc:
            raise AuthorError(str(exc)) from exc
        code = submitted.get("code")
        if not isinstance(code, str) or not code.strip():
            raise AuthorError("submit_tool had no code")
        return ToolCandidate(manifest, code)

    def _ask(self, system: str, brief: JSONObject, tool: ToolSpec) -> JSONObject:
        try:
            response = self._llm.generate(
                system=system,
                messages=[UserMessage(json.dumps(brief, indent=2))],
                tools=[tool],
            )
        except LLMError as exc:
            # A failed authoring call is a failed attempt, not a failed run.
            raise AuthorError(f"the model call failed: {exc}") from exc
        return _submitted(response, tool)


def _submitted(response: LLMResponse, tool: ToolSpec) -> JSONObject:
    for call in response.tool_calls:
        if call.name == tool.name:
            return call.arguments
    raise AuthorError(f"the model did not call {tool.name}")


def _test_case(tool: str, raw: object) -> EvalCase:
    if not isinstance(raw, dict):
        raise AuthorError("each test must be an object")
    test = cast("dict[str, Any]", raw)
    name, kind, arguments = test.get("name"), test.get("kind"), test.get("arguments")
    output, error = test.get("expect_output"), test.get("expect_error")
    files = test.get("files", {})
    if not isinstance(files, dict) or not all(
        isinstance(k, str) and isinstance(v, str)
        for k, v in cast("dict[object, object]", files).items()
    ):
        raise AuthorError(f"test {name!r}: files must map relative paths to text")
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
        kind="tool",
        description=f"generated {kind} test",
        expect=Expectations(output_equals=cast("JSONObject | None", output), error_contains=error),
        tags=(str(kind),),
        tool=tool,
        arguments=cast("JSONObject", arguments),
        files=cast("dict[str, str]", files),
    )


def _test_json(case: EvalCase) -> JSONObject:
    test: JSONObject = {"name": case.id, "arguments": case.arguments}
    if case.files:
        test["files"] = case.files
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
