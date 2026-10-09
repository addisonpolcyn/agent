"""The human in the loop: nothing is built without the user's approval.

The user sees the plan ("I don't have this tool, but I can try to build it") and answers
once. A yes covers building, testing and using the tool: no second question. Learned tools
run with full access to the machine, and their code is saved locally for anyone who wants to
read it.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Callable

    from agentlab.learning.plan import ToolPlan


class Approver(Protocol):
    """The human in the loop. Nothing is built without their approval."""

    def approve_plan(self, plan: ToolPlan) -> bool: ...


@dataclass(frozen=True)
class FixedApprover:
    """Answers with a fixed value. For tests and evals, never for real users."""

    plan: bool

    def approve_plan(self, plan: ToolPlan) -> bool:
        return self.plan


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

    def approve_plan(self, plan: ToolPlan) -> bool:
        self._show(
            "\nI don't have a tool for this yet, but I can try to build one:\n"
            f"{plan.describe()}\n"
            "New tools are generated Python that runs on this machine with full access (files, "
            "network, commands). If it passes its tests it is used right away and saved only "
            "here (.agentlab/learned/)."
        )
        return self._confirm("Build it? [y/N] ")

    def _confirm(self, prompt: str) -> bool:
        if not self._interactive:
            self._show("(no terminal to ask on, so not approved)")
            return False
        try:
            answer = self._ask(prompt)
        except EOFError, KeyboardInterrupt:
            return False
        return answer.strip().lower() in {"y", "yes"}
