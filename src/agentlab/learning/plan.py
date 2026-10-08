"""The model's plan for a missing capability, and the deterministic rules it must pass.

The model proposes; this module disposes. A plan is a few generic steps, each either reusing
an existing skill or describing a new one. Plans that are too large, that duplicate what the
catalog already provides, or that need the network or side effects never reach the user's
approval gate.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

from agentlab.learning.models import SkillSpec
from agentlab.models import ToolSpec

if TYPE_CHECKING:
    from agentlab.models import JSONObject
    from agentlab.skills.catalog import SkillCatalog

MAX_STEPS = 5
MAX_NEW_SKILLS = 2
MAX_NAME_LENGTH = 40
_SNAKE_CASE = re.compile(r"^[a-z][a-z0-9]*(_[a-z0-9]+)*$")

type PlanOutcome = Literal[
    "accepted", "malformed", "refused_too_large", "refused_reuse", "refused_not_learnable"
]

PROPOSE_SKILL_PLAN = ToolSpec(
    name="propose_skill_plan",
    description=(
        "Propose building new skills when none of the available tools can do what the task "
        "needs. Break the need into at most "
        f"{MAX_STEPS} small steps. Each step either reuses an existing tool (reuse = its name) or "
        f"describes a new GENERIC skill (new_skill), at most {MAX_NEW_SKILLS} new skills. Name "
        "new skills for the general operation (html_to_text, word_count), never for this task. "
        "New skills are sandboxed Python. They may READ local files (set reads_files; the user "
        "approves each folder when the skill first touches it) but can never use the network, "
        "write files or cause side effects: mark steps that need those and expect the plan to "
        "be refused. The user must approve the plan, and again the tested code, before use."
    ),
    input_schema={
        "type": "object",
        "required": ["goal", "steps"],
        "additionalProperties": False,
        "properties": {
            "goal": {"type": "string", "description": "what the skills will let you do"},
            "steps": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": [
                        "summary",
                        "capability",
                        "needs_network",
                        "has_side_effects",
                    ],
                    "additionalProperties": False,
                    "properties": {
                        "summary": {"type": "string"},
                        "capability": {
                            "type": "string",
                            "description": "general snake_case capability, e.g. html_parsing",
                        },
                        "reuse": {"type": "string", "description": "existing tool to use"},
                        "new_skill": {
                            "type": "object",
                            "required": ["name", "purpose", "inputs", "outputs"],
                            "additionalProperties": False,
                            "properties": {
                                "name": {"type": "string", "description": "snake_case"},
                                "purpose": {"type": "string"},
                                "inputs": {"type": "string"},
                                "outputs": {"type": "string"},
                            },
                        },
                        "reads_files": {
                            "type": "boolean",
                            "description": "the new skill reads local files or folders",
                        },
                        "needs_network": {"type": "boolean"},
                        "has_side_effects": {"type": "boolean"},
                    },
                },
            },
        },
    },
)


class PlanError(Exception):
    """The plan does not follow the tool's schema."""


@dataclass(frozen=True)
class PlanStep:
    summary: str
    capability: str
    reuse: str | None
    new_skill: SkillSpec | None
    needs_network: bool
    has_side_effects: bool


@dataclass(frozen=True)
class SkillPlan:
    goal: str
    steps: tuple[PlanStep, ...]

    @property
    def new_skills(self) -> tuple[SkillSpec, ...]:
        return tuple(step.new_skill for step in self.steps if step.new_skill is not None)

    def describe(self) -> str:
        """Human-readable plan for the approval gate."""
        lines = [f"Goal: {self.goal}"]
        for number, step in enumerate(self.steps, start=1):
            if step.new_skill is not None:
                how = f"build new skill '{step.new_skill.name}': {step.new_skill.purpose}"
                if step.new_skill.reads_files:
                    how += " (reads local files; you approve each folder)"
            else:
                how = f"reuse existing skill '{step.reuse}'"
            lines.append(f"  {number}. {step.summary} [{step.capability}] -> {how}")
        return "\n".join(lines)


@dataclass(frozen=True)
class PlanVerdict:
    outcome: PlanOutcome
    reason: str = ""


def parse_plan(arguments: JSONObject) -> SkillPlan:
    goal = arguments.get("goal")
    steps = arguments.get("steps")
    if not isinstance(goal, str) or not goal.strip():
        raise PlanError("'goal' must be a non-empty string")
    if not isinstance(steps, list) or not steps:
        raise PlanError("'steps' must be a non-empty list")
    return SkillPlan(goal, tuple(_parse_step(raw) for raw in cast("list[object]", steps)))


def validate_plan(plan: SkillPlan, catalog: SkillCatalog) -> PlanVerdict:
    """The first rule that objects decides; a plan no rule objects to is accepted."""
    for rule in (_size_rule, _learnable_rule, _reuse_rule, _naming_rule):
        verdict = rule(plan, catalog)
        if verdict is not None:
            return verdict
    return PlanVerdict("accepted")


def _size_rule(plan: SkillPlan, catalog: SkillCatalog) -> PlanVerdict | None:
    if len(plan.steps) > MAX_STEPS:
        return PlanVerdict(
            "refused_too_large",
            f"{len(plan.steps)} steps is more than the limit of {MAX_STEPS}. Split the task "
            "into smaller requests.",
        )
    if len(plan.new_skills) > MAX_NEW_SKILLS:
        return PlanVerdict(
            "refused_too_large",
            f"{len(plan.new_skills)} new skills is more than the limit of {MAX_NEW_SKILLS} per "
            "request. Learn the most general one first.",
        )
    return None


def _learnable_rule(plan: SkillPlan, catalog: SkillCatalog) -> PlanVerdict | None:
    # Reusing an existing (trusted) network skill is fine; only *learned* code is limited.
    unlearnable = [
        s.capability
        for s in plan.steps
        if s.new_skill is not None and (s.needs_network or s.has_side_effects)
    ]
    if not unlearnable:
        return None
    return PlanVerdict(
        "refused_not_learnable",
        f"learned skills cannot use the network or cause side effects ({', '.join(unlearnable)})",
    )


def _reuse_rule(plan: SkillPlan, catalog: SkillCatalog) -> PlanVerdict | None:
    provided = _capabilities(catalog)
    for step in plan.steps:
        if step.reuse is not None and step.reuse not in catalog:
            return PlanVerdict("malformed", f"step {step.summary!r} reuses unknown {step.reuse!r}")
        spec = step.new_skill
        if spec is not None and spec.name in catalog:
            return PlanVerdict("refused_reuse", f"a skill named {spec.name!r} exists; use it")
        if spec is not None and spec.capability in provided:
            return PlanVerdict(
                "refused_reuse",
                f"'{provided[spec.capability]}' already provides {spec.capability!r}; use it",
            )
    if not plan.new_skills:
        return PlanVerdict("refused_reuse", "every step reuses an existing skill; call them")
    return None


def _naming_rule(plan: SkillPlan, catalog: SkillCatalog) -> PlanVerdict | None:
    names = [spec.name for spec in plan.new_skills]
    for name in names:
        if not _SNAKE_CASE.match(name) or len(name) > MAX_NAME_LENGTH:
            return PlanVerdict("malformed", f"{name!r} is not a short snake_case name")
    if len(set(names)) != len(names):
        return PlanVerdict("malformed", "new skill names must be distinct")
    return None


def _capabilities(catalog: SkillCatalog) -> dict[str, str]:
    return {
        capability: manifest.name
        for manifest in catalog.manifests
        for capability in manifest.capabilities
    }


def _parse_step(raw: object) -> PlanStep:
    if not isinstance(raw, dict):
        raise PlanError("each step must be an object")
    step = cast("dict[str, Any]", raw)
    summary, capability = step.get("summary"), step.get("capability")
    if not isinstance(summary, str) or not isinstance(capability, str) or not capability:
        raise PlanError("each step needs a 'summary' and a 'capability'")
    reuse, new_skill = step.get("reuse"), step.get("new_skill")
    if (reuse is None) == (new_skill is None):
        raise PlanError(f"step '{summary}' needs exactly one of 'reuse' or 'new_skill'")
    if reuse is not None and not isinstance(reuse, str):
        raise PlanError(f"step '{summary}': 'reuse' must be a skill name")
    return PlanStep(
        summary=summary,
        capability=capability,
        reuse=reuse,
        new_skill=None
        if new_skill is None
        else _parse_spec(new_skill, capability, reads_files=step.get("reads_files") is True),
        needs_network=step.get("needs_network") is True,
        has_side_effects=step.get("has_side_effects") is True,
    )


def _parse_spec(raw: object, capability: str, *, reads_files: bool) -> SkillSpec:
    if not isinstance(raw, dict):
        raise PlanError("'new_skill' must be an object")
    spec = cast("dict[str, Any]", raw)
    fields = [spec.get(key) for key in ("name", "purpose", "inputs", "outputs")]
    if not all(isinstance(value, str) and value.strip() for value in fields):
        raise PlanError("'new_skill' needs non-empty name, purpose, inputs and outputs")
    name, purpose, inputs, outputs = cast("list[str]", fields)
    return SkillSpec(name, capability, purpose, inputs, outputs, reads_files)
