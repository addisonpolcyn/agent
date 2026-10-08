"""Command-line entry point: ``agentlab ask | chat | eval | flywheel | skills``."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from agentlab.agent.loop import Agent
from agentlab.config import Settings
from agentlab.evals.cases import EvalCaseError, load_cases
from agentlab.evals.runner import format_summary, run_suite
from agentlab.flywheel.loop import record_iteration
from agentlab.llm.claude import ClaudeClient
from agentlab.llm.client import LLMError
from agentlab.llm.fake import OfflineLLM
from agentlab.skills.catalog import discover_catalog
from agentlab.skills.models import SkillManifestError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from agentlab.agent.loop import AgentRun
    from agentlab.llm.client import LLMClient
    from agentlab.skills.catalog import SkillCatalog

EXIT_OK = 0
EXIT_EVAL_FAILED = 1
EXIT_ERROR = 2


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    settings = Settings.from_env(os.environ)
    try:
        catalog = discover_catalog(settings.skills_dir)
        match args.command:
            case "skills":
                return _skills(catalog)
            case "ask":
                return _ask(_agent(settings, catalog, args.offline), args.task)
            case "chat":
                return _chat(_agent(settings, catalog, args.offline))
            case "eval":
                return _eval(settings, catalog, _agent(settings, catalog, args.offline))
            case _:
                return _flywheel(settings, catalog, _agent(settings, catalog, args.offline))
    except (LLMError, SkillManifestError, EvalCaseError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agentlab", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    offline_help = "use the deterministic offline stand-in instead of Claude (no network)"

    ask = commands.add_parser("ask", help="run the agent on one task")
    ask.add_argument("task")
    ask.add_argument("--offline", action="store_true", help=offline_help)
    for name, help_text in [
        ("chat", "ask questions interactively (each is answered independently)"),
        ("eval", "run the eval suite"),
        ("flywheel", "run evals, record the iteration, and summarize failures"),
    ]:
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--offline", action="store_true", help=offline_help)
    commands.add_parser("skills", help="list discovered skills")
    return parser


def _agent(settings: Settings, catalog: SkillCatalog, offline: bool) -> Agent:
    llm: LLMClient
    if offline:
        llm = OfflineLLM()
    elif settings.anthropic_api_key is None:
        raise LLMError("ANTHROPIC_API_KEY is not set (or pass --offline)")
    else:
        llm = ClaudeClient(api_key=settings.anthropic_api_key.reveal(), model=settings.model)
    return Agent(llm, catalog)


def _skills(catalog: SkillCatalog) -> int:
    for manifest in catalog.manifests:
        print(f"{manifest.name}  [{', '.join(manifest.capabilities)}]")
        print(f"    {manifest.description}")
    return EXIT_OK


def _ask(agent: Agent, task: str) -> int:
    _print_run(agent.run(task))
    return EXIT_OK


def _chat(agent: Agent) -> int:
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
            _print_run(agent.run(task))
        except LLMError as exc:
            # Report and keep the session alive; one failed call shouldn't end the chat.
            print(f"error: {exc}", file=sys.stderr)


def _print_run(run: AgentRun) -> None:
    print(run.answer or "(no answer)")
    print(_format_trace(run))


def _eval(settings: Settings, catalog: SkillCatalog, agent: Agent) -> int:
    summary = run_suite(load_cases(settings.cases_dir), agent, catalog)
    print(format_summary(summary))
    return EXIT_OK if summary.all_passed else EXIT_EVAL_FAILED


def _flywheel(settings: Settings, catalog: SkillCatalog, agent: Agent) -> int:
    summary = run_suite(load_cases(settings.cases_dir), agent, catalog)
    iteration = record_iteration(summary, settings.runs_dir, datetime.now(UTC))
    print(iteration.report)
    print(f"Recorded in {iteration.run_dir}")
    return EXIT_OK


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
    return "\n".join(lines)
