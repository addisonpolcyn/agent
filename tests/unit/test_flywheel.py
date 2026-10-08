from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from agentlab.evals.models import CheckResult, EvalResult, EvalSummary
from agentlab.flywheel.loop import FAILURES_FILE, RESULTS_FILE, record_iteration

if TYPE_CHECKING:
    from pathlib import Path

T0 = datetime(2026, 10, 8, 12, 0, tzinfo=UTC)


def summary(**passing: bool) -> EvalSummary:
    return EvalSummary(
        tuple(
            EvalResult(
                case_id, "agent", (), (CheckResult("answer_contains", ok, "" if ok else "missing"),)
            )
            for case_id, ok in passing.items()
        )
    )


def test_first_iteration_is_the_baseline(tmp_path: Path) -> None:
    iteration = record_iteration(summary(a=True, b=False), tmp_path, T0)

    assert iteration.comparison is None
    assert iteration.run_dir.name == "20261008T120000000000Z"
    results = json.loads((iteration.run_dir / RESULTS_FILE).read_text())
    assert results["passed"] == 1
    assert results["results"][1]["checks"][0]["detail"] == "missing"
    report = (iteration.run_dir / FAILURES_FILE).read_text()
    assert report == iteration.report
    assert "baseline" in report
    assert "### answer_contains (1)" in report
    assert "- `b`: missing" in report


def test_second_iteration_compares_with_previous(tmp_path: Path) -> None:
    record_iteration(summary(a=True, b=False), tmp_path, T0)
    iteration = record_iteration(summary(a=False, b=True), tmp_path, T0 + timedelta(minutes=1))

    assert iteration.comparison is not None
    assert iteration.comparison.previous_run == "20261008T120000000000Z"
    assert iteration.comparison.newly_failing == ("a",)
    assert iteration.comparison.newly_passing == ("b",)
    assert "- Newly failing: a" in iteration.report


def test_all_passing_report_asks_for_harder_cases(tmp_path: Path) -> None:
    iteration = record_iteration(summary(a=True), tmp_path, T0)
    assert "Add a harder case" in iteration.report
