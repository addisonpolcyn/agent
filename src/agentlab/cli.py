"""Command-line entry point: ``agentlab ask | chat | eval | flywheel | skills``."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from agentlab.agent.loop import Agent
from agentlab.config import ConfigError, Settings
from agentlab.evals.cases import EvalCaseError, load_cases
from agentlab.evals.runner import format_summary, run_suite
from agentlab.flywheel.loop import record_iteration
from agentlab.learning.approval import ConsoleApprover
from agentlab.learning.author import LEARNED_IMPLEMENTATION, SkillAuthor
from agentlab.learning.learner import SkillLearner, load_learned
from agentlab.llm.claude import ClaudeClient
from agentlab.llm.client import LLMError
from agentlab.llm.fake import FixtureAuthorLLM, OfflineLLM
from agentlab.skills.catalog import discover_catalog
from agentlab.skills.models import SkillManifestError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from agentlab.agent.loop import AgentRun
    from agentlab.evals.models import EvalSummary
    from agentlab.learning.approval import Approver
    from agentlab.llm.client import LLMClient
    from agentlab.skills.catalog import SkillCatalog

EXIT_OK = 0
EXIT_EVAL_FAILED = 1
EXIT_ERROR = 2


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        settings = Settings.load()
        catalog = discover_catalog(settings.skills_dir)
        match args.command:
            case "skills":
                return _skills(settings, catalog)
            case "ask" | "chat":
                learned_dir = None if args.no_learn else settings.learned_dir
                agent = _agent(settings, catalog, args.offline, learned_dir)
                if args.command == "ask":
                    return _ask(agent, args.task, ConsoleApprover())
                return _chat(agent, ConsoleApprover())
            case "eval":
                return _eval(settings, catalog, args.offline)
            case _:
                return _flywheel(settings, catalog, args.offline)
    except (ConfigError, LLMError, SkillManifestError, EvalCaseError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agentlab", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    offline_help = "use the deterministic offline stand-in instead of Claude (no network)"
    no_learn_help = "use only the repo's skills: no learning, and learned skills are ignored"

    ask = commands.add_parser("ask", help="run the agent on one task")
    ask.add_argument("task")
    chat = commands.add_parser(
        "chat", help="ask questions interactively (each is answered independently)"
    )
    for command in (ask, chat):
        command.add_argument("--offline", action="store_true", help=offline_help)
        command.add_argument("--no-learn", action="store_true", help=no_learn_help)
    for name, help_text in [
        ("eval", "run the eval suite"),
        ("flywheel", "run evals, record the iteration, and summarize failures"),
    ]:
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--offline", action="store_true", help=offline_help)
    commands.add_parser("skills", help="list discovered and learned skills")
    return parser


def _agent(
    settings: Settings, catalog: SkillCatalog, offline: bool, learned_dir: Path | None
) -> Agent:
    """An agent; with ``learned_dir`` it can learn skills and keeps them there."""
    llm: LLMClient
    author_llm: LLMClient
    if offline:
        llm, author_llm = OfflineLLM(), FixtureAuthorLLM()
    elif settings.anthropic_api_key is None:
        raise LLMError("ANTHROPIC_API_KEY is not set (or pass --offline)")
    else:
        llm = ClaudeClient(api_key=settings.anthropic_api_key.reveal(), model=settings.model)
        author_llm = llm
    if learned_dir is None:
        return Agent(llm, catalog)
    return Agent(llm, catalog, learner=SkillLearner(SkillAuthor(author_llm), learned_dir))


def _skills(settings: Settings, catalog: SkillCatalog) -> int:
    for manifest in catalog.with_skills(load_learned(settings.learned_dir)).manifests:
        learned = "  [learned]" if manifest.implementation == LEARNED_IMPLEMENTATION else ""
        print(f"{manifest.name}  [{', '.join(manifest.capabilities)}]{learned}")
        print(f"    {manifest.description}")
    return EXIT_OK


def _ask(agent: Agent, task: str, approver: Approver) -> int:
    _print_run(agent.run(task, approver=approver))
    return EXIT_OK


def _chat(agent: Agent, approver: Approver) -> int:
    """A REPL over ``Agent.run``. The agent has no memory yet, so turns don't see each other."""
    print("agentlab chat. Each question is answered independently. 'exit' or Ctrl-D to quit.")
    while True:
        try:
            task = input("\n> ").strip()
        except EOFError, KeyboardInterrupt:
            print()
            return EXIT_OK
        if task in {"exit", "quit"}:
            return EXIT_OK
        if not task:
            continue
        try:
            _print_run(agent.run(task, approver=approver))
        except LLMError as exc:
            # Report and keep the session alive; one failed call shouldn't end the chat.
            print(f"error: {exc}", file=sys.stderr)


def _print_run(run: AgentRun) -> None:
    print(run.answer or "(no answer)")
    print(_format_trace(run))


def _eval(settings: Settings, catalog: SkillCatalog, offline: bool) -> int:
    summary = _run_suite(settings, catalog, offline)
    print(format_summary(summary))
    return EXIT_OK if summary.all_passed else EXIT_EVAL_FAILED


def _flywheel(settings: Settings, catalog: SkillCatalog, offline: bool) -> int:
    summary = _run_suite(settings, catalog, offline)
    iteration = record_iteration(summary, settings.runs_dir, datetime.now(UTC))
    print(iteration.report)
    print(f"Recorded in {iteration.run_dir}")
    return EXIT_OK


def _run_suite(settings: Settings, catalog: SkillCatalog, offline: bool) -> EvalSummary:
    """Each case gets a fresh agent and its own empty learned-skills directory."""
    cases = load_cases(settings.cases_dir)
    with tempfile.TemporaryDirectory(prefix="agentlab-eval-") as root:

        def agent_for() -> Agent:
            return _agent(settings, catalog, offline, Path(tempfile.mkdtemp(dir=root)))

        return run_suite(cases, agent_for, catalog)


def _format_trace(run: AgentRun) -> str:
    lines = ["", f"trace ({run.steps} step(s), stop: {run.stop_reason})"]
    for invocation in run.invocations:
        lines.append(
            f"  skill {invocation.name}({json.dumps(invocation.arguments)}) "
            f"-> {json.dumps(invocation.result.as_content())}"
        )
    lines.extend(
        f"  capability gap: {gap.capability} ({gap.reason})" for gap in run.capability_gaps
    )
    lines.extend(f"  learning: {o.outcome} ({o.reason})" for o in run.learning)
    return "\n".join(lines)
