"""Real Claude API. Opt-in: AGENTLAB_LIVE_TESTS=1 uv run pytest -m live"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest

from agentlab.agent.loop import Agent
from agentlab.config import Settings
from agentlab.llm.claude import ClaudeClient

if TYPE_CHECKING:
    from agentlab.skills.catalog import SkillCatalog

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        os.environ.get("AGENTLAB_LIVE_TESTS") != "1" or not os.environ.get("ANTHROPIC_API_KEY"),
        reason="set AGENTLAB_LIVE_TESTS=1 and ANTHROPIC_API_KEY to run live tests",
    ),
]


@pytest.fixture
def agent(catalog: SkillCatalog) -> Agent:
    settings = Settings.from_env()
    assert settings.anthropic_api_key is not None
    client = ClaudeClient(api_key=settings.anthropic_api_key.reveal(), model=settings.model)
    return Agent(client, catalog)


def test_claude_uses_calculator(agent: Agent) -> None:
    run = agent.run("What is 123 * 456? Use a tool.")
    assert "calculator" in run.skills_used
    assert run.answer is not None
    assert "56088" in run.answer.replace(",", "")


def test_claude_reports_missing_web_capability(agent: Agent) -> None:
    run = agent.run("Find the latest nonstop flights from SFO to Tokyo next month with prices.")
    assert run.capability_gaps, run.answer
