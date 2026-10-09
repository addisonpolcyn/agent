"""Real Claude API. Opt in with `uv run pytest -m live`; the key comes from env or .env.

Skips when no key is configured. CI checks for the key separately, so a missing secret there
fails loudly instead of skipping.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from agentlab.agent.loop import Agent
from agentlab.config import Settings
from agentlab.llm.claude import ClaudeClient

if TYPE_CHECKING:
    from agentlab.tools.catalog import ToolCatalog

SETTINGS = Settings.load(Path(__file__).resolve().parents[2] / ".env")

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        SETTINGS.anthropic_api_key is None,
        reason="set ANTHROPIC_API_KEY (environment or .env) to run live tests",
    ),
]


@pytest.fixture
def agent(catalog: ToolCatalog) -> Agent:
    assert SETTINGS.anthropic_api_key is not None
    client = ClaudeClient(api_key=SETTINGS.anthropic_api_key.reveal(), model=SETTINGS.model)
    return Agent(client, catalog)


def test_claude_uses_calculator(agent: Agent) -> None:
    run = agent.run("What is 123 * 456? Use a tool.")
    assert "calculator" in run.tools_used
    assert run.answer is not None
    assert "56088" in run.answer.replace(",", "")


def test_claude_reports_missing_web_capability(agent: Agent) -> None:
    run = agent.run("Find the latest nonstop flights from SFO to Tokyo next month with prices.")
    assert run.capability_gaps, run.answer
