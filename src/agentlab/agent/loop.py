"""The agent loop: ask the model, run the skills it selects, feed back observations, repeat.

    task -> LLM -> (tool calls -> skill execution -> observations -> LLM)* -> answer

The loop knows nothing about individual skills. It offers the catalog as tools, plus one
built-in tool, ``request_capability``, which the model uses to say "none of these skills can
do what this task needs". That gap is recorded in the trace, so evals can check it.

With a ``SkillLearner``, the loop also offers ``propose_skill_plan``: the model can propose
building generic new skills, and the learner validates, asks the user, builds and evaluates
them (see ``agentlab.learning``). Skills learned during a run become tools on the next step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from agentlab.learning.plan import PROPOSE_SKILL_PLAN
from agentlab.models import AssistantMessage, ToolResultMessage, ToolSpec, UserMessage

if TYPE_CHECKING:
    from collections.abc import Sequence

    from agentlab.learning.approval import Approver
    from agentlab.learning.learner import LearningOutcome, SkillLearner
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

LEARNING_PROMPT = """

You can also learn new skills, with the user's permission. Learning costs time, model calls \
and the user's attention, so use it only when a tool is clearly better than you:
- Do it yourself when the task is small, one-off, or a matter of judgment or language \
(summarizing, classifying, a short text, a few values).
- A tool is better when the result must be exact and the input is large or error-prone \
(long text, many records, big or nested JSON or HTML), when the same work will recur, or \
when the user asks for a reusable capability.
- First check whether your existing tools can do the job, alone or combined. Prefer them.
- If they can't, and a small generic program would do it (parsing, extracting, counting, \
converting, reading or writing local files, running local commands, fetching a known URL or \
API), call propose_skill_plan with a few small, generic steps. Reuse existing tools for steps \
they cover. Name new skills for the general operation (list_files, write_text_file, \
html_to_text), not this task.
- New skills run on the user's machine with full access: any library, files (set \
reads_files on steps that read them), the network and commands.
- If the user declines or learning fails, say so. Don't present a result worked out in your \
head as if a tool had produced it.
- Open-ended live research (finding flights, prices, news, weather) needs a web research \
capability you don't have yet: call request_capability for it rather than learning one.
- After propose_skill_plan, follow its 'next' instruction and tell the user what happened: \
the skill is ready and used, the user declined, it failed after several attempts, or the \
plan was refused (for example, too large)."""

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
    learning: tuple[LearningOutcome, ...] = ()
    # The conversation as it ended, to pass back as ``history`` for a follow-up task.
    messages: tuple[Message, ...] = ()

    @property
    def skills_used(self) -> tuple[str, ...]:
        return tuple(invocation.name for invocation in self.invocations)

    @property
    def skills_learned(self) -> tuple[str, ...]:
        return tuple(name for outcome in self.learning for name in outcome.learned)


class Agent:
    def __init__(
        self,
        llm: LLMClient,
        catalog: SkillCatalog,
        *,
        learner: SkillLearner | None = None,
        max_steps: int = 8,
    ) -> None:
        for reserved in (REQUEST_CAPABILITY.name, PROPOSE_SKILL_PLAN.name):
            if reserved in catalog:
                raise ValueError(f"skill name {reserved!r} is reserved")
        self._llm = llm
        self._catalog = catalog
        self._learner = learner
        self._max_steps = max_steps

    def run(
        self,
        task: str,
        *,
        approver: Approver | None = None,
        history: Sequence[Message] = (),
    ) -> AgentRun:
        """Solve ``task``, after the earlier turns in ``history`` (e.g. ``AgentRun.messages``).

        Without an ``approver``, every request to learn a skill is declined.
        """
        earlier = _without_provider_state(history)
        messages: list[Message] = [*earlier, UserMessage(task)]
        catalog: SkillCatalog = self._catalog
        builtins = [REQUEST_CAPABILITY]
        system = SYSTEM_PROMPT
        if self._learner is not None:
            catalog = catalog.with_skills(self._learner.load_learned())
            builtins.append(PROPOSE_SKILL_PLAN)
            system += LEARNING_PROMPT
        invocations: list[ToolInvocation] = []
        gaps: list[CapabilityGap] = []
        learning: list[LearningOutcome] = []

        def finish(answer: str | None, stop_reason: StopReason, steps: int) -> AgentRun:
            return AgentRun(
                task,
                answer,
                stop_reason,
                steps,
                tuple(invocations),
                tuple(gaps),
                tuple(learning),
                tuple(messages),
            )

        for step in range(1, self._max_steps + 1):
            tools = [*catalog.tool_specs(), *builtins]
            response = self._llm.generate(system=system, messages=messages, tools=tools)
            messages.append(response.as_message())
            if not response.tool_calls:
                return finish(response.text, "answered", step)
            for call in response.tool_calls:
                if call.name == REQUEST_CAPABILITY.name:
                    gap = _capability_gap(call)
                    gaps.append(gap)
                    messages.append(_gap_observation(call, gap))
                elif call.name == PROPOSE_SKILL_PLAN.name and self._learner is not None:
                    learned: tuple[LearningOutcome, SkillCatalog] = self._learner.learn(
                        call.arguments, catalog, approver
                    )
                    outcome, catalog = learned
                    learning.append(outcome)
                    if outcome.ok:
                        # The tool list just changed. Providers may bind earlier turns to the
                        # tools they were produced with (Claude's thinking blocks are), so
                        # editing them is not allowed. Keep every conversation append-only:
                        # continue in a fresh one, as if the run had started with the new skill.
                        messages = [*earlier, UserMessage(_after_learning(task, learning))]
                        break
                    messages.append(ToolResultMessage(call.id, outcome.observation(), True))
                else:
                    result = catalog.execute(call.name, call.arguments)
                    invocations.append(ToolInvocation(call.name, call.arguments, result))
                    messages.append(ToolResultMessage(call.id, result.as_content(), not result.ok))
        return finish(None, "max_steps", self._max_steps)


def _without_provider_state(history: Sequence[Message]) -> list[Message]:
    """Earlier turns as plain text and tool calls.

    Providers may bind replayed state to the request it came from. Claude's thinking blocks are
    bound to the tool list, which changes when a skill is learned. Earlier turns' answers and
    tool results carry the context; their private reasoning is left behind, never edited.
    """
    return [
        AssistantMessage(m.text, m.tool_calls) if isinstance(m, AssistantMessage) else m
        for m in history
    ]


def _after_learning(task: str, learning: list[LearningOutcome]) -> str:
    """The task again, plus what was just learned, so the answer can say so."""
    learned = ", ".join(name for outcome in learning for name in outcome.learned)
    return (
        f"{task}\n\n[agentlab] With the user's approval, you just learned and tested these "
        f"skills for this task: {learned}. Use them, and mention that they are newly learned."
    )


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
