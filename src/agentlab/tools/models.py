"""Tool metadata and results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from agentlab.models import JSONObject


class ToolError(Exception):
    """A tool rejected its input or could not produce a result. Reported back to the model."""


class ToolManifestError(Exception):
    """A tool manifest is malformed or points at an implementation that cannot be loaded."""


@dataclass(frozen=True)
class ToolManifest:
    """Everything the agent may know about a tool, read from its ``tool.toml``."""

    name: str
    description: str
    when_to_use: str
    limitations: str
    capabilities: tuple[str, ...]
    input_schema: JSONObject
    output_schema: JSONObject
    implementation: str
    path: Path
    # Learned tools only: may read user-approved local folders through the sandbox.
    reads_files: bool = False


@dataclass(frozen=True)
class ToolResult:
    """Outcome of one tool execution. Exactly one of ``output`` / ``error`` is set."""

    output: JSONObject | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    def as_content(self) -> JSONObject:
        if self.output is not None:
            return self.output
        return {"error": self.error}
