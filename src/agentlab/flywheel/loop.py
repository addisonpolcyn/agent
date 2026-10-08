"""One turn of the flywheel: record an eval run, compare it to the last one, summarize failures.

    eval cases -> agent -> results -> evaluation -> failure summary -> (you) -> next iteration

Deliberately boring: it writes files and prints a report. A human (or a future, supervised
process) reads the failure summary and decides what to change. Nothing here modifies code.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from datetime import datetime
    from pathlib import Path

    from agentlab.evals.models import EvalSummary

RESULTS_FILE = "results.json"
FAILURES_FILE = "failures.md"


@dataclass(frozen=True)
class Comparison:
    previous_run: str
    previous_pass_rate: float
    newly_failing: tuple[str, ...]
    newly_passing: tuple[str, ...]


@dataclass(frozen=True)
class Iteration:
    run_dir: Path
    report: str
    comparison: Comparison | None


def record_iteration(summary: EvalSummary, runs_dir: Path, now: datetime) -> Iteration:
    run_dir = runs_dir / now.strftime("%Y%m%dT%H%M%S%fZ")
    previous = _latest_run(runs_dir)
    comparison = _compare(summary, previous) if previous is not None else None
    report = render_report(summary, run_dir.name, comparison)

    run_dir.mkdir(parents=True)
    payload = {"run": run_dir.name, **summary.to_json()}
    (run_dir / RESULTS_FILE).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (run_dir / FAILURES_FILE).write_text(report, encoding="utf-8")
    return Iteration(run_dir, report, comparison)


def render_report(summary: EvalSummary, run_name: str, comparison: Comparison | None) -> str:
    lines = [
        f"# Eval iteration {run_name}",
        "",
        f"Passed {summary.passed_count}/{len(summary.scored)} cases "
        f"({summary.pass_rate:.0%}), mean score {summary.mean_score:.2f}"
        f"{f', {summary.skipped_count} skipped' if summary.skipped_count else ''}.",
        "",
        "## Change since previous run",
        "",
    ]
    if comparison is None:
        lines.append("No previous run: this is the baseline.")
    else:
        lines += [
            f"Previous: {comparison.previous_run} "
            f"({comparison.previous_pass_rate:.0%} -> {summary.pass_rate:.0%})",
            f"- Newly failing: {', '.join(comparison.newly_failing) or 'none'}",
            f"- Newly passing: {', '.join(comparison.newly_passing) or 'none'}",
        ]
    lines += ["", "## Failures by mode", ""]
    modes = summary.failures_by_mode()
    if not modes:
        lines.append("None. Add a harder case to keep the flywheel turning.")
    for mode, failures in sorted(modes.items(), key=lambda item: -len(item[1])):
        lines.append(f"### {mode} ({len(failures)})")
        lines += [f"- `{case_id}`: {detail}" for case_id, detail in failures]
        lines.append("")
    lines += [
        "",
        "## Next step",
        "",
        "Pick the most frequent failure mode, make one targeted change, and run the flywheel "
        "again.",
    ]
    return "\n".join(lines) + "\n"


def _latest_run(runs_dir: Path) -> Path | None:
    if not runs_dir.is_dir():
        return None
    runs = sorted(p for p in runs_dir.iterdir() if (p / RESULTS_FILE).is_file())
    return runs[-1] if runs else None


def _compare(summary: EvalSummary, previous_dir: Path) -> Comparison:
    previous = json.loads((previous_dir / RESULTS_FILE).read_text(encoding="utf-8"))
    was_passing = {
        r["case_id"]: bool(r["passed"]) for r in cast("list[dict[str, Any]]", previous["results"])
    }
    now_passing = {r.case_id: r.passed for r in summary.scored}
    return Comparison(
        previous_run=previous_dir.name,
        previous_pass_rate=float(previous["pass_rate"]),
        newly_failing=tuple(
            c for c, ok in now_passing.items() if not ok and was_passing.get(c, False)
        ),
        newly_passing=tuple(c for c, ok in now_passing.items() if ok and not was_passing.get(c)),
    )
