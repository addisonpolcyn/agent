"""Full offline loop: OfflineLLM + the real skill catalog + the real eval cases."""

from __future__ import annotations

from typing import TYPE_CHECKING

from agentlab.agent.loop import Agent
from agentlab.evals.cases import load_cases
from agentlab.evals.runner import run_suite
from agentlab.llm.fake import OfflineLLM

if TYPE_CHECKING:
    from pathlib import Path

    from agentlab.skills.catalog import SkillCatalog


def test_discovers_selects_and_uses_calculator(catalog: SkillCatalog) -> None:
    run = Agent(OfflineLLM(), catalog).run("What is 123 * 456?")

    assert run.skills_used == ("calculator",)
    assert run.invocations[0].arguments == {"expression": "123 * 456"}
    assert run.invocations[0].result.output == {"result": 56088}
    assert run.answer == "The result is 56088."
    assert run.stop_reason == "answered"


def test_flight_request_reports_missing_web_capability(catalog: SkillCatalog) -> None:
    run = Agent(OfflineLLM(), catalog).run("Find the latest flights from SFO to Tokyo")

    assert run.skills_used == ()
    assert [gap.capability for gap in run.capability_gaps] == ["current_information"]
    assert run.answer is not None
    assert "can't" in run.answer


def test_offline_suite_passes(catalog: SkillCatalog, cases_dir: Path) -> None:
    summary = run_suite(load_cases(cases_dir), Agent(OfflineLLM(), catalog), catalog)
    failures = {r.case_id: r.failed_checks for r in summary.results if not r.passed}
    assert failures == {}
