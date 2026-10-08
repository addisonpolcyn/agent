"""The learner: rules, two approval gates, the harness, and the local store."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pytest

from agentlab.learning.approval import ConsoleApprover, FileGrants, FixedApprover
from agentlab.learning.author import SkillAuthor
from agentlab.learning.learner import SkillLearner, load_learned
from agentlab.learning.plan import parse_plan
from agentlab.llm.fake import FixtureAuthorLLM, ScriptedLLM
from agentlab.models import LLMResponse, ToolCall, ToolSpec, UserMessage
from agentlab.skills.models import SkillManifestError

if TYPE_CHECKING:
    from pathlib import Path

    from agentlab.learning.models import HarnessReport
    from agentlab.learning.plan import SkillPlan
    from agentlab.models import JSONObject
    from agentlab.skills.catalog import SkillCatalog

PLAN: JSONObject = {
    "goal": "Count words",
    "steps": [
        {
            "summary": "Count the words in a text",
            "capability": "text_statistics",
            "new_skill": {
                "name": "word_count",
                "purpose": "Count words.",
                "inputs": "text",
                "outputs": "count",
            },
            "needs_network": False,
            "has_side_effects": False,
        }
    ],
}


@dataclass
class SpyApprover:
    plan: bool = True
    skill: bool = True
    asked: list[str] = field(default_factory=list[str])

    def approve_plan(self, plan: SkillPlan) -> bool:
        self.asked.append("plan")
        return self.plan

    def approve_skill(self, report: HarnessReport) -> bool:
        self.asked.append(f"skill:{report.verdict}")
        return self.skill


def learner(store: Path) -> SkillLearner:
    return SkillLearner(SkillAuthor(FixtureAuthorLLM()), store)


def test_approved_skill_is_built_saved_and_usable(catalog: SkillCatalog, tmp_path: Path) -> None:
    approver = SpyApprover()
    outcome, updated = learner(tmp_path).learn(PLAN, catalog, approver)

    assert outcome.outcome == "ready"
    assert outcome.learned == ("word_count",)
    assert approver.asked == ["plan", "skill:ready"]
    assert updated.execute("word_count", {"text": "a b c"}).output == {"count": 3}
    assert "word_count" not in catalog, "the input catalog is never mutated"
    files = sorted(p.name for p in (tmp_path / "word_count").iterdir())
    assert files == ["report.json", "skill.json", "skill.py", "tests.json"]


def test_learned_skills_reload_and_stay_sandboxed(catalog: SkillCatalog, tmp_path: Path) -> None:
    learner(tmp_path).learn(PLAN, catalog, SpyApprover())
    reloaded = catalog.with_skills(learner(tmp_path).load_learned())

    assert reloaded.execute("word_count", {"text": "x y"}).output == {"count": 2}
    assert reloaded.execute("word_count", {"text": 5}).error == "text must be a string"


def test_without_an_approver_nothing_is_built(catalog: SkillCatalog, tmp_path: Path) -> None:
    outcome, updated = learner(tmp_path).learn(PLAN, catalog, None)

    assert outcome.outcome == "declined_plan"
    assert updated is catalog
    assert not tmp_path.exists() or not any(tmp_path.iterdir())


def test_rejected_code_is_never_saved_as_a_skill(catalog: SkillCatalog, tmp_path: Path) -> None:
    outcome, updated = learner(tmp_path).learn(PLAN, catalog, FixedApprover(True, False))

    assert outcome.outcome == "declined_skill"
    assert "word_count" not in updated
    assert (tmp_path / "word_count" / "report.json").exists()
    assert load_learned(tmp_path) == []


def test_refused_plans_never_reach_the_user(catalog: SkillCatalog, tmp_path: Path) -> None:
    approver = SpyApprover()
    step = {**PLAN["steps"][0], "needs_network": True}
    outcome, _ = learner(tmp_path).learn({**PLAN, "steps": [step]}, catalog, approver)

    assert outcome.outcome == "refused_not_learnable"
    assert approver.asked == []
    assert "request_capability" in outcome.observation()["next"]


def test_malformed_plan_is_reported(catalog: SkillCatalog, tmp_path: Path) -> None:
    outcome, _ = learner(tmp_path).learn({"goal": "g"}, catalog, SpyApprover())
    assert outcome.outcome == "malformed"


def test_failed_build_is_reported_honestly(catalog: SkillCatalog, tmp_path: Path) -> None:
    fixture = FixtureAuthorLLM()
    junk = {"description": "d", "when_to_use": "w", "limitations": "l",
            "code": "def run(arguments):\n    return {'count': 2}\n"}  # fmt: skip
    contract = fixture.generate(
        system="",
        messages=[UserMessage(json.dumps({"skill": {"name": "word_count"}}))],
        tools=[ToolSpec("submit_contract", "", {"type": "object"})],
    )
    llm = ScriptedLLM([contract, *[LLMResponse(None, (ToolCall("c", "submit_skill", junk),))] * 3])
    approver = SpyApprover()
    outcome, updated = SkillLearner(SkillAuthor(llm), tmp_path).learn(PLAN, catalog, approver)

    assert outcome.outcome == "failed"
    assert "not ready after 2 attempt(s)" in outcome.reason
    assert approver.asked == ["plan"], "a failed skill is never offered for approval"
    assert "word_count" not in updated
    report = json.loads((tmp_path / "word_count" / "report.json").read_text())
    assert report["verdict"] == "failed"


def test_tampered_learned_code_fails_loudly(catalog: SkillCatalog, tmp_path: Path) -> None:
    learner(tmp_path).learn(PLAN, catalog, SpyApprover())
    (tmp_path / "word_count" / "skill.py").write_text("import os\ndef run(a):\n    return {}\n")
    with pytest.raises(SkillManifestError, match="import of 'os'"):
        load_learned(tmp_path)


def test_learned_manifest_must_point_at_the_sandbox(catalog: SkillCatalog, tmp_path: Path) -> None:
    learner(tmp_path).learn(PLAN, catalog, SpyApprover())
    path = tmp_path / "word_count" / "skill.json"
    data = json.loads(path.read_text())
    path.write_text(
        json.dumps({**data, "implementation": "agentlab.skills.builtin.calculator:run"})
    )
    with pytest.raises(SkillManifestError, match=r"must use 'sandbox:skill\.py'"):
        load_learned(tmp_path)


@pytest.mark.parametrize(
    ("answer", "approved"), [("y", True), ("YES", True), ("n", False), ("", False)]
)
def test_console_approver_needs_an_explicit_yes(answer: str, approved: bool) -> None:
    shown: list[str] = []
    approver = ConsoleApprover(ask=lambda _: answer, show=shown.append, interactive=True)
    assert approver.approve_plan(_plan()) is approved
    assert "I don't have a skill for this yet" in shown[0]


def test_console_approver_says_no_without_a_terminal() -> None:
    def never(prompt: str) -> str:
        raise AssertionError("must not prompt without a terminal")

    shown: list[str] = []
    approver = ConsoleApprover(ask=never, show=shown.append, interactive=False)
    assert approver.approve_plan(_plan()) is False
    assert "not approved" in shown[-1]


def _plan() -> SkillPlan:
    return parse_plan(PLAN)


LIST_PLAN: JSONObject = {
    "goal": "List folders",
    "steps": [
        {
            "summary": "List a folder",
            "capability": "directory_listing",
            "new_skill": {"name": "list_files", "purpose": "p", "inputs": "i", "outputs": "o"},
            "reads_files": True,
            "needs_network": False,
            "has_side_effects": False,
        }
    ],
}


def file_learner(store: Path, answers: list[bool], asked: list[str]) -> SkillLearner:
    replies = iter(answers)

    def confirm(question: str) -> bool:
        asked.append(question)
        return next(replies)

    grants = FileGrants(confirm)
    return SkillLearner(SkillAuthor(FixtureAuthorLLM()), store, grants=grants)


def test_file_skill_is_tested_on_fixtures_then_asks_per_folder(
    catalog: SkillCatalog, tmp_path: Path
) -> None:
    data = tmp_path / "data"
    (data / "inner").mkdir(parents=True)
    (data / "notes.txt").write_text("x")
    asked: list[str] = []
    the_learner = file_learner(tmp_path / "store", [True], asked)
    outcome, updated = the_learner.learn(LIST_PLAN, catalog, SpyApprover())

    assert outcome.outcome == "ready", outcome.reason
    assert asked == [], "testing on fixture files needs no access to the user's folders"
    listed = updated.execute("list_files", {"path": str(data)})
    assert listed.output == {
        "entries": [{"name": "inner", "type": "dir"}, {"name": "notes.txt", "type": "file"}]
    }
    assert len(asked) == 1
    assert str(data.resolve()) in asked[0]
    updated.execute("list_files", {"path": str(data / "inner")})
    assert len(asked) == 1, "an approved folder covers its subfolders for the session"


def test_denied_folder_is_an_error_and_not_asked_again(
    catalog: SkillCatalog, tmp_path: Path
) -> None:
    asked: list[str] = []
    the_learner = file_learner(tmp_path / "store", [False], asked)
    _, updated = the_learner.learn(LIST_PLAN, catalog, SpyApprover())

    for _ in range(2):
        result = updated.execute("list_files", {"path": str(tmp_path)})
        assert result.error == f"the user did not allow reading {tmp_path.resolve()}"
    assert len(asked) == 1


def test_without_anyone_to_ask_folders_stay_closed(catalog: SkillCatalog, tmp_path: Path) -> None:
    _, updated = learner(tmp_path / "store").learn(LIST_PLAN, catalog, SpyApprover())
    result = updated.execute("list_files", {"path": str(tmp_path)})
    assert result.error is not None
    assert "did not allow" in result.error
