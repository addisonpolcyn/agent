"""The human in the loop: nothing is built, or used, without two approvals.

Gate 1 shows the plan ("I don't have this skill, but I can try to build it"). Gate 2 shows
the generated code and the harness verdict before the skill is ever used.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Callable

    from agentlab.learning.models import HarnessReport
    from agentlab.learning.plan import SkillPlan


class Approver(Protocol):
    """The human in the loop. Nothing is built or used without both approvals."""

    def approve_plan(self, plan: SkillPlan) -> bool: ...

    def approve_skill(self, report: HarnessReport) -> bool: ...


@dataclass(frozen=True)
class FixedApprover:
    """Answers both gates with fixed values. For tests and evals, never for real users."""

    plan: bool
    skill: bool

    def approve_plan(self, plan: SkillPlan) -> bool:
        return self.plan

    def approve_skill(self, report: HarnessReport) -> bool:
        return self.skill


class ConsoleApprover:
    """Asks on the terminal. Anything but 'y' or 'yes' is a no, and so is having no terminal."""

    def __init__(
        self,
        *,
        ask: Callable[[str], str] = input,
        show: Callable[[str], None] = print,
        interactive: bool | None = None,
    ) -> None:
        self._ask = ask
        self._show = show
        self._interactive = sys.stdin.isatty() if interactive is None else interactive

    def approve_plan(self, plan: SkillPlan) -> bool:
        self._show(
            "\nI don't have a skill for this yet, but I can try to build one:\n"
            f"{plan.describe()}\n"
            "New skills are generated Python, tested in a sandbox (no network, no files) and "
            "saved only on this machine. You'll see the code before it is used."
        )
        return self._confirm("Build it? [y/N] ")

    def approve_skill(self, report: HarnessReport) -> bool:
        assert report.candidate is not None, "only ready skills reach the second gate"
        self._show(
            f"\n{report.summary()}\n--- code for {report.spec.name} ---\n"
            f"{report.candidate.code.rstrip()}\n---"
        )
        return self._confirm(f"Use '{report.spec.name}'? [y/N] ")

    def _confirm(self, prompt: str) -> bool:
        if not self._interactive:
            self._show("(no terminal to ask on, so not approved)")
            return False
        try:
            answer = self._ask(prompt)
        except EOFError, KeyboardInterrupt:
            return False
        return answer.strip().lower() in {"y", "yes"}
