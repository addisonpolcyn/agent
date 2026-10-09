"""Learn tools on demand: validate the plan, ask the user, build and evaluate, store.

    plan -> validate_plan -> user approves the plan -> build_tool (harness)
         -> saved under the local learned dir -> added to the catalog

Learned tools live only in the local learned directory (``.agentlab/learned/`` by default,
git-ignored). They are reloaded on later runs and run in a separate process, unrestricted
(see ``sandbox.py``). The user's one approval, of the plan, covers building and using them.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from agentlab.learning.author import LEARNED_IMPLEMENTATION
from agentlab.learning.harness import MAX_ATTEMPTS, build_tool
from agentlab.learning.plan import MAX_NEW_TOOLS, PlanError, parse_plan, validate_plan
from agentlab.learning.sandbox import check_source, run_sandboxed
from agentlab.tools.catalog import manifest_from_data
from agentlab.tools.models import ToolError, ToolManifestError

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Any

    from agentlab.learning.approval import Approver
    from agentlab.learning.author import ToolAuthor
    from agentlab.learning.harness import Runner
    from agentlab.learning.models import HarnessReport
    from agentlab.models import JSONObject
    from agentlab.tools.catalog import ToolCatalog, ToolFunction
    from agentlab.tools.models import ToolManifest

TOOL_FILE = "tool.json"
CODE_FILE = "tool.py"
TESTS_FILE = "tests.json"
REPORT_FILE = "report.json"
# Learned-tool code that ships with agentlab; see ``install_starter_tools``.
STARTER_DIR = Path(__file__).parent / "starter"

# What the model is told to do next, per outcome. It explains the result to the user.
_GUIDANCE = {
    "malformed": "Fix the plan and propose it again, or tell the user what you cannot do.",
    "refused_too_large": "Tell the user this is too large to learn in one go and suggest "
    "splitting it into smaller requests.",
    "refused_reuse": "Use the existing tools instead of building new ones.",
    "declined_plan": "The user said no. Tell them plainly what you cannot do without it.",
    "failed": "Tell the user the tool isn't working after a reasonable number of attempts, "
    "and why. Do not guess the answer instead.",
    "ready": "The new tools are now available as tools.",
}


@dataclass(frozen=True)
class LearningOutcome:
    """What happened to one ``propose_tool_plan`` call. Recorded in the ``AgentRun``."""

    outcome: str
    reason: str
    learned: tuple[str, ...] = ()
    reports: tuple[HarnessReport, ...] = ()

    @property
    def ok(self) -> bool:
        return self.outcome == "ready"

    def observation(self) -> JSONObject:
        return {
            "outcome": self.outcome,
            "learned_tools": list(self.learned),
            "detail": self.reason,
            "next": _GUIDANCE[self.outcome],
        }


class ToolLearner:
    def __init__(
        self,
        author: ToolAuthor,
        store_dir: Path,
        *,
        run: Runner = run_sandboxed,
        max_attempts: int = MAX_ATTEMPTS,
    ) -> None:
        self._author = author
        self._store_dir = store_dir
        self._run = run
        self._max_attempts = max_attempts

    def load_learned(self) -> list[tuple[ToolManifest, ToolFunction]]:
        return load_learned(self._store_dir, self._run)

    def learn(
        self,
        arguments: JSONObject,
        catalog: ToolCatalog,
        approver: Approver | None,
        *,
        already_learned: int = 0,
    ) -> tuple[LearningOutcome, ToolCatalog]:
        """Handle one plan. Returns the outcome and the catalog including any new tools.

        ``already_learned`` counts tools learned earlier for the same request: the limit on
        new tools is per request, so several small plans can't add up past it.
        """
        try:
            plan = parse_plan(arguments)
        except PlanError as exc:
            return LearningOutcome("malformed", str(exc)), catalog
        verdict = validate_plan(plan, catalog)
        if verdict.outcome != "accepted":
            return LearningOutcome(verdict.outcome, verdict.reason), catalog
        if already_learned + len(plan.new_tools) > MAX_NEW_TOOLS:
            reason = (
                f"{already_learned} tool(s) already learned for this request; "
                f"{len(plan.new_tools)} more is past the limit of {MAX_NEW_TOOLS} per request. "
                "Use what you learned, and suggest the rest as separate requests."
            )
            return LearningOutcome("refused_too_large", reason), catalog
        if approver is None or not approver.approve_plan(plan):
            return LearningOutcome("declined_plan", "the user did not approve the plan"), catalog

        learned: list[str] = []
        reports: list[HarnessReport] = []
        for spec in plan.new_tools:
            report = build_tool(
                spec,
                self._author,
                manifest_path=self._store_dir / spec.name / TOOL_FILE,
                run=self._run,
                max_attempts=self._max_attempts,
            )
            reports.append(report)
            _save_report(self._store_dir, report)
            if report.verdict == "failed":
                return LearningOutcome(
                    "failed", report.summary(), tuple(learned), tuple(reports)
                ), catalog
            saved = _save_tool(self._store_dir, report, self._run)
            catalog = catalog.with_tools([saved])
            learned.append(spec.name)
        summary = "; ".join(report.summary() for report in reports)
        return LearningOutcome("ready", summary, tuple(learned), tuple(reports)), catalog


def install_starter_tools(store_dir: Path) -> list[str]:
    """Copy the starter tools into ``store_dir``, skipping any name already there.

    Starter tools are learned tools that ship with agentlab, so they run in the tool process
    like any other. An existing directory is never overwritten, so local edits survive.
    Returns the names installed.
    """
    installed: list[str] = []
    for source in sorted(STARTER_DIR.glob(f"*/{TOOL_FILE}")):
        target = store_dir / source.parent.name
        if target.exists():
            continue
        target.mkdir(parents=True)
        shutil.copyfile(source.parent / CODE_FILE, target / CODE_FILE)
        # Written last, as in ``_save_tool``: a directory without tool.json is never loaded.
        shutil.copyfile(source, target / TOOL_FILE)
        installed.append(target.name)
    return installed


def load_learned(
    store_dir: Path, run: Runner = run_sandboxed
) -> list[tuple[ToolManifest, ToolFunction]]:
    """Every saved tool in ``store_dir``. Malformed ones fail loudly, like
    ``discover_catalog``: the files may have been edited by hand."""
    tools: list[tuple[ToolManifest, ToolFunction]] = []
    for path in sorted(store_dir.glob(f"*/{TOOL_FILE}")):
        try:
            manifest = manifest_from_data(json.loads(path.read_text(encoding="utf-8")), path)
        except json.JSONDecodeError as exc:
            raise ToolManifestError(f"{path}: invalid JSON: {exc}") from exc
        if manifest.implementation != LEARNED_IMPLEMENTATION:
            raise ToolManifestError(f"{path}: learned tools must use {LEARNED_IMPLEMENTATION!r}")
        code = (path.parent / CODE_FILE).read_text(encoding="utf-8")
        problems = check_source(code)
        if problems:
            raise ToolManifestError(f"{path.parent / CODE_FILE}: {'; '.join(problems)}")
        tools.append((manifest, sandboxed_tool(code, run)))
    return tools


def sandboxed_tool(code: str, run: Runner) -> ToolFunction:
    """Adapt generated code to the catalog's tool contract.

    A crash in generated code is the tool's failure, not a bug of ours, so it becomes a
    ``ToolError`` like any other.
    """

    def execute(arguments: Mapping[str, Any]) -> JSONObject:
        result = run(code, dict(arguments))
        if result.output is not None:
            return result.output
        raise ToolError(result.error or f"learned tool crashed: {result.crash}")

    return execute


def _save_report(store_dir: Path, report: HarnessReport) -> None:
    directory = store_dir / report.spec.name
    directory.mkdir(parents=True, exist_ok=True)
    _write_json(directory / REPORT_FILE, report.to_json())


def _save_tool(
    store_dir: Path, report: HarnessReport, run: Runner
) -> tuple[ToolManifest, ToolFunction]:
    assert report.candidate is not None
    assert report.contract is not None
    manifest, code = report.candidate.manifest, report.candidate.code
    directory = store_dir / manifest.name
    (directory / CODE_FILE).write_text(code, encoding="utf-8")
    _write_json(
        directory / TESTS_FILE,
        {
            "input_schema": report.contract.input_schema,
            "output_schema": report.contract.output_schema,
            "tests": [
                {
                    "name": case.id,
                    "kind": case.tags[0],
                    "arguments": case.arguments,
                    "expect_output": case.expect.output_equals,
                    "expect_error": case.expect.error_contains,
                    "files": case.files,
                }
                for case in report.contract.tests
            ],
        },
    )
    # Written last: a directory without tool.json is never loaded.
    _write_json(
        directory / TOOL_FILE,
        {
            "name": manifest.name,
            "description": manifest.description,
            "when_to_use": manifest.when_to_use,
            "limitations": manifest.limitations,
            "capabilities": list(manifest.capabilities),
            "input_schema": manifest.input_schema,
            "output_schema": manifest.output_schema,
            "implementation": manifest.implementation,
            "reads_files": manifest.reads_files,
        },
    )
    return manifest, sandboxed_tool(code, run)


def _write_json(path: Path, data: JSONObject) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
