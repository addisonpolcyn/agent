"""The agent loop: ask the model, run the skills it selects, feed back observations, repeat.

    task -> LLM -> (tool calls -> skill execution -> observations -> LLM)* -> answer

The loop knows nothing about individual skills. It offers the catalog as tools, plus one
built-in tool, ``request_capability``, which the model uses to say "none of these skills can
do what this task needs". That gap is recorded in the trace (so evals can check it) and is the
hook where future versions will search for and load a matching skill.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from agentlab.models import ToolResultMessage, ToolSpec, UserMessage

if TYPE_CHECKING:
    from agentlab.llm.client import LLMClient
    from agentlab.models import JSONObject, Message, ToolCall
    from agentlab.skills.catalog import SkillCatalog
    from agentlab.skills.models import SkillResult

SYSTEM_PROMPT = """\
You are a careful assistant that solves tasks using the tools available to you.

- If a tool fits the task, call it and base your answer on its result.
- If the task needs a capability none of your tools provide (for example current information \
from the web), call request_capability to name it, then tell the user what you cannot do.
- Never invent facts, prices, schedules, or links. Say what you don't know.
- If no tool is needed, answer directly."""

REQUEST_CAPABILITY = ToolSpec(
    name="request_capability",
    description=(
        "Report that the task needs a capability none of the available tools provide. Name the "
        "most general capability that is missing, not the specific task: use "
        "'current_information' for anything that needs live or recent data from the web "
        "(flights, prices, schedules, news, weather). Use a short snake_case name."
    ),
    input_schema={
        "type": "object",
        "required": ["capability", "reason"],
        "additionalProperties": False,
        "properties": {
            "capability": {"type": "string", "description": "snake_case capability name"},
            "reason": {"type": "string", "description": "why the task needs it"},
        },
    },
)

type StopReason = Literal["answered", "max_steps"]


@dataclass(frozen=True)
class ToolInvocation:
    name: str
    arguments: JSONObject
    result: SkillResult


@dataclass(frozen=True)
class CapabilityGap:
    capability: str
    reason: str


@dataclass(frozen=True)
class AgentRun:
    """The full trace of one task. Evals inspect this, not just the answer."""

    task: str
    answer: str | None
    stop_reason: StopReason
    steps: int
    invocations: tuple[ToolInvocation, ...]
    capability_gaps: tuple[CapabilityGap, ...]

    @property
    def skills_used(self) -> tuple[str, ...]:
        return tuple(invocation.name for invocation in self.invocations)


class Agent:
    def __init__(self, llm: LLMClient, catalog: SkillCatalog, *, max_steps: int = 5) -> None:
        if REQUEST_CAPABILITY.name in catalog:
            raise ValueError(f"skill name {REQUEST_CAPABILITY.name!r} is reserved")
        self._llm = llm
        self._catalog = catalog
        self._max_steps = max_steps

    def run(self, task: str) -> AgentRun:
        messages: list[Message] = [UserMessage(task)]
        tools = [*self._catalog.tool_specs(), REQUEST_CAPABILITY]
        invocations: list[ToolInvocation] = []
        gaps: list[CapabilityGap] = []

        def finish(answer: str | None, stop_reason: StopReason, steps: int) -> AgentRun:
            return AgentRun(task, answer, stop_reason, steps, tuple(invocations), tuple(gaps))

        for step in range(1, self._max_steps + 1):
            response = self._llm.generate(system=SYSTEM_PROMPT, messages=messages, tools=tools)
            messages.append(response.as_message())
            if not response.tool_calls:
                return finish(response.text, "answered", step)
            for call in response.tool_calls:
                if call.name == REQUEST_CAPABILITY.name:
                    gap = _capability_gap(call)
                    gaps.append(gap)
                    messages.append(_gap_observation(call, gap))
                else:
                    result = self._catalog.execute(call.name, call.arguments)
                    invocations.append(ToolInvocation(call.name, call.arguments, result))
                    messages.append(ToolResultMessage(call.id, result.as_content(), not result.ok))
        return finish(None, "max_steps", self._max_steps)


def _capability_gap(call: ToolCall) -> CapabilityGap:
    return CapabilityGap(
        capability=str(call.arguments.get("capability", "")),
        reason=str(call.arguments.get("reason", "")),
    )


def _gap_observation(call: ToolCall, gap: CapabilityGap) -> ToolResultMessage:
    return ToolResultMessage(
        call.id,
        {
            "available": False,
            "message": (
                f"No available skill provides '{gap.capability}'. Tell the user plainly what "
                "you cannot do and what they could do instead. Do not guess."
            ),
        },
    )
