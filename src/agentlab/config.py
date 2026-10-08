"""Settings from the environment (and an optional ``.env`` file).

Read once at the CLI edge and passed down explicitly. Real environment variables take
precedence over ``.env``, so CI secrets and one-off overrides always win.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

DEFAULT_MODEL = "claude-opus-5-5"
DEFAULT_ENV_FILE = Path(".env")


class ConfigError(Exception):
    """Configuration is malformed (e.g. an unparseable ``.env`` line)."""


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
    learned_dir: Path = Path(".agentlab/learned")
    anthropic_api_key: Secret | None = None

    @classmethod
    def load(cls, env_file: Path = DEFAULT_ENV_FILE) -> Settings:
        """Settings from ``env_file`` (if it exists) overlaid by the process environment."""
        return cls.from_env({**read_env_file(env_file), **os.environ})

    @classmethod
    def from_env(cls, env: Mapping[str, str] = os.environ) -> Settings:
        defaults = cls()
        key = env.get("ANTHROPIC_API_KEY", "").strip()
        return cls(
            model=env.get("AGENTLAB_MODEL") or defaults.model,
            skills_dir=Path(env.get("AGENTLAB_SKILLS_DIR") or defaults.skills_dir),
            cases_dir=Path(env.get("AGENTLAB_CASES_DIR") or defaults.cases_dir),
            runs_dir=Path(env.get("AGENTLAB_RUNS_DIR") or defaults.runs_dir),
            learned_dir=Path(env.get("AGENTLAB_LEARNED_DIR") or defaults.learned_dir),
            anthropic_api_key=Secret(key) if key else None,
        )


def read_env_file(path: Path) -> dict[str, str]:
    """Parse ``KEY=VALUE`` lines. Blank lines, ``#`` comments, ``export`` and quotes are allowed.

    A missing file is not an error: ``.env`` is optional.
    """
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, value = line.removeprefix("export ").partition("=")
        key = key.strip()
        if not sep or not key.isidentifier():
            raise ConfigError(f"{path}:{number}: expected KEY=VALUE")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        values[key] = value
    return values
