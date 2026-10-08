"""Skill metadata and results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from agentlab.models import JSONObject


class SkillError(Exception):
    """A skill rejected its input or could not produce a result. Reported back to the model."""


class SkillManifestError(Exception):
    """A skill manifest is malformed or points at an implementation that cannot be loaded."""


@dataclass(frozen=True)
class SkillManifest:
    """Everything the agent may know about a skill, read from its ``skill.toml``."""

    name: str
    description: str
    when_to_use: str
    limitations: str
    capabilities: tuple[str, ...]
    input_schema: JSONObject
    output_schema: JSONObject
    implementation: str
    path: Path
    # Learned skills only: may read user-approved local folders through the sandbox.
    reads_files: bool = False


@dataclass(frozen=True)
class SkillResult:
    """Outcome of one skill execution. Exactly one of ``output`` / ``error`` is set."""

    output: JSONObject | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    def as_content(self) -> JSONObject:
        if self.output is not None:
            return self.output
        return {"error": self.error}
