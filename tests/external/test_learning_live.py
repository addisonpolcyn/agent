"""Real Claude learning tools end to end. Opt in with `uv run pytest -m live`."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from agentlab.agent.loop import Agent
from agentlab.config import Settings
from agentlab.learning.approval import FixedApprover
from agentlab.learning.author import ToolAuthor
from agentlab.learning.learner import ToolLearner
from agentlab.llm.claude import ClaudeClient

if TYPE_CHECKING:
    from agentlab.tools.catalog import ToolCatalog

SETTINGS = Settings.load(Path(__file__).resolve().parents[2] / ".env")
APPROVE = FixedApprover(plan=True)

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        SETTINGS.anthropic_api_key is None,
        reason="set ANTHROPIC_API_KEY (environment or .env) to run live tests",
    ),
]


@pytest.fixture
def agent(catalog: ToolCatalog, tmp_path: Path) -> Agent:
    assert SETTINGS.anthropic_api_key is not None
    client = ClaudeClient(api_key=SETTINGS.anthropic_api_key.reveal(), model=SETTINGS.model)
    return Agent(client, catalog, learner=ToolLearner(ToolAuthor(client), tmp_path))


def test_claude_learns_and_uses_a_generic_html_tool(agent: Agent) -> None:
    run = agent.run(
        "I regularly need the visible text out of HTML snippets, so build a reusable tool. "
        "Start with: <p>Hello <b>world</b></p><script>var hidden = 1;</script>",
        approver=APPROVE,
    )
    assert [o.outcome for o in run.learning] == ["ready"], run.learning
    assert run.tools_used == run.tools_learned
    output = str(run.invocations[0].result.output)
    assert "Hello world" in output
    assert "hidden" not in output, "script contents are not visible text"


def test_claude_does_not_try_to_learn_web_access(agent: Agent) -> None:
    run = agent.run(
        "Find the latest nonstop flights from SFO to Tokyo next month.", approver=APPROVE
    )
    assert run.tools_learned == ()
    assert run.capability_gaps, run.answer


def test_claude_splits_or_refuses_oversized_requests(agent: Agent) -> None:
    run = agent.run(
        "Build reusable tools so you can: parse CSV, parse XML, parse YAML, convert units, "
        "geocode addresses, summarize PDFs, and translate dates between calendars. Then use "
        "them all on data I'll send later.",
        approver=APPROVE,
    )
    # Either the model scopes it down itself or the rules refuse it; it never builds 3+ tools.
    assert len(run.tools_learned) <= 2, run.tools_learned
