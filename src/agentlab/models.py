"""Provider-neutral conversation types shared by the agent loop and every LLM client.

These are the only shapes that cross the LLM boundary. Vendor SDK types stay inside
their adapter (see ``agentlab.llm.claude``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

type JSONObject = dict[str, Any]


@dataclass(frozen=True)
class ToolSpec:
    """A capability offered to the model: what it is called, what it does, what it accepts."""

    name: str
    description: str
    input_schema: JSONObject


@dataclass(frozen=True)
class ToolCall:
    """The model's request to invoke a tool."""

    id: str
    name: str
    arguments: JSONObject


@dataclass(frozen=True)
class UserMessage:
    text: str


@dataclass(frozen=True)
class AssistantMessage:
    """A model turn.

    ``provider_state`` is opaque to everything except the client that produced it. Adapters
    use it to replay their native content verbatim (e.g. Claude's thinking blocks must be
    sent back unchanged on the next tool-use turn).
    """

    text: str | None
    tool_calls: tuple[ToolCall, ...] = ()
    provider_state: object = field(default=None, compare=False, repr=False)


@dataclass(frozen=True)
class ToolResultMessage:
    """The observation for one tool call, fed back to the model."""

    call_id: str
    content: JSONObject
    is_error: bool = False


type Message = UserMessage | AssistantMessage | ToolResultMessage


@dataclass(frozen=True)
class LLMResponse:
    """What a model returned for one ``generate`` call."""

    text: str | None
    tool_calls: tuple[ToolCall, ...] = ()
    provider_state: object = field(default=None, compare=False, repr=False)

    def as_message(self) -> AssistantMessage:
        return AssistantMessage(self.text, self.tool_calls, self.provider_state)
