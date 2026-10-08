"""Settings from the environment. Read once at the CLI edge and passed down explicitly."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

DEFAULT_MODEL = "claude-opus-5-5"


@dataclass(frozen=True)
class Secret:
    """A string that never appears in reprs, logs, or tracebacks."""

    _value: str = field(repr=False)

    def reveal(self) -> str:
        return self._value


@dataclass(frozen=True)
class Settings:
    model: str = DEFAULT_MODEL
    skills_dir: Path = Path("skills")
    cases_dir: Path = Path("evals/cases")
    runs_dir: Path = Path("runs")
    anthropic_api_key: Secret | None = None

    @classmethod
    def from_env(cls, env: Mapping[str, str] = os.environ) -> Settings:
        defaults = cls()
        key = env.get("ANTHROPIC_API_KEY", "").strip()
        return cls(
            model=env.get("AGENTLAB_MODEL") or defaults.model,
            skills_dir=Path(env.get("AGENTLAB_SKILLS_DIR") or defaults.skills_dir),
            cases_dir=Path(env.get("AGENTLAB_CASES_DIR") or defaults.cases_dir),
            runs_dir=Path(env.get("AGENTLAB_RUNS_DIR") or defaults.runs_dir),
            anthropic_api_key=Secret(key) if key else None,
        )
