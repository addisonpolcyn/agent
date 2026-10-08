"""The LLM boundary: the one interface the agent loop depends on."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Sequence

    from agentlab.models import LLMResponse, Message, ToolSpec


class LLMError(Exception):
    """A model call failed. The message is safe to show to users (never contains secrets)."""


class LLMClient(Protocol):
    def generate(
        self,
        *,
        system: str,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec],
    ) -> LLMResponse:
        """Return the model's next turn: final text, tool calls, or both."""
        ...
