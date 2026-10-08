"""Claude adapter. The only module that imports the Anthropic SDK.

Maps provider-neutral messages and tool specs to the Messages API and back. Assistant turns
carry their raw content blocks as opaque ``provider_state`` so thinking blocks are replayed
unchanged on the next tool-use turn, as the API requires.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import anthropic

from agentlab.llm.client import LLMError
from agentlab.models import AssistantMessage, LLMResponse, ToolCall, UserMessage

if TYPE_CHECKING:
    from collections.abc import Sequence

    from anthropic.types.beta import (
        BetaContentBlock,
        BetaContentBlockParam,
        BetaMessage,
        BetaMessageParam,
        BetaToolParam,
        BetaToolResultBlockParam,
    )

    from agentlab.models import Message, ToolSpec

type Effort = Literal["low", "medium", "high", "xhigh", "max"]

# Server-side refusal fallback: on a policy decline the API re-runs the request on a
# fallback model within the same call, so the agent loop never sees a dead turn.
FALLBACK_BETA = "server-side-fallback-2026-07-01"


@dataclass(frozen=True)
class _ClaudeTurn:
    content: list[BetaContentBlock]


class ClaudeClient:
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        max_tokens: int = 16000,
        effort: Effort = "medium",
    ) -> None:
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens
        self._effort: Effort = effort

    def generate(
        self, *, system: str, messages: Sequence[Message], tools: Sequence[ToolSpec]
    ) -> LLMResponse:
        try:
            response = self._client.beta.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                system=system,
                messages=to_anthropic_messages(messages),
                tools=[to_anthropic_tool(tool) for tool in tools],
                output_config={"effort": self._effort},
                betas=[FALLBACK_BETA],
                fallbacks="default",
            )
        except anthropic.AuthenticationError as exc:
            raise LLMError("Anthropic rejected the API key (check ANTHROPIC_API_KEY)") from exc
        except anthropic.RateLimitError as exc:
            raise LLMError("Anthropic rate limit exceeded; retry later") from exc
        except anthropic.APIStatusError as exc:
            raise LLMError(f"Anthropic API error {exc.status_code}: {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMError("could not reach the Anthropic API") from exc
        return from_anthropic_response(response)


def to_anthropic_tool(tool: ToolSpec) -> BetaToolParam:
    return {"name": tool.name, "description": tool.description, "input_schema": tool.input_schema}


def to_anthropic_messages(messages: Sequence[Message]) -> list[BetaMessageParam]:
    """Convert the conversation. Consecutive tool results share one user message."""
    converted: list[BetaMessageParam] = []
    results: list[BetaContentBlockParam] = []
    for message in messages:
        if isinstance(message, UserMessage | AssistantMessage) and results:
            converted.append({"role": "user", "content": results})
            results = []
        match message:
            case UserMessage(text=text):
                converted.append({"role": "user", "content": text})
            case AssistantMessage():
                converted.append({"role": "assistant", "content": _assistant_content(message)})
            case _:
                results.append(_tool_result(message.call_id, message.content, message.is_error))
    if results:
        converted.append({"role": "user", "content": results})
    return converted


def from_anthropic_response(response: BetaMessage) -> LLMResponse:
    if response.stop_reason == "refusal":
        raise LLMError("Claude declined this request")
    if response.stop_reason == "max_tokens":
        raise LLMError("Claude's response was cut off at max_tokens")
    texts = [block.text for block in response.content if block.type == "text"]
    calls = tuple(
        ToolCall(block.id, block.name, dict(block.input))
        for block in response.content
        if block.type == "tool_use"
    )
    return LLMResponse(
        text="\n".join(texts) if texts else None,
        tool_calls=calls,
        provider_state=_ClaudeTurn(list(response.content)),
    )


def _assistant_content(message: AssistantMessage) -> list[BetaContentBlockParam]:
    if isinstance(message.provider_state, _ClaudeTurn):
        return list(message.provider_state.content)
    content: list[BetaContentBlockParam] = []
    if message.text:
        content.append({"type": "text", "text": message.text})
    content.extend(
        {"type": "tool_use", "id": call.id, "name": call.name, "input": call.arguments}
        for call in message.tool_calls
    )
    return content


def _tool_result(call_id: str, content: object, is_error: bool) -> BetaToolResultBlockParam:
    return {
        "type": "tool_result",
        "tool_use_id": call_id,
        "content": json.dumps(content),
        "is_error": is_error,
    }
