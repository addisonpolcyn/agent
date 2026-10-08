"""Skill discovery and execution.

Discovery reads ``<skills_dir>/*/skill.toml``. Each manifest names its implementation as
``module:function``; implementations must live inside the ``agentlab`` package, so nothing
from the skills directory itself is ever executed.
"""

from __future__ import annotations

import importlib
import tomllib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from agentlab.models import ToolSpec
from agentlab.skills.models import SkillError, SkillManifest, SkillManifestError, SkillResult

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from pathlib import Path

    from agentlab.models import JSONObject

type SkillFunction = Callable[[Mapping[str, Any]], JSONObject]

MANIFEST_FILENAME = "skill.toml"
TRUSTED_IMPLEMENTATION_PREFIX = "agentlab."
PROVIDES_PREFIX = "Provides: "


@dataclass(frozen=True)
class _LoadedSkill:
    manifest: SkillManifest
    run: SkillFunction


class SkillCatalog:
    """The skills available to an agent, keyed by name."""

    def __init__(self, skills: Mapping[str, _LoadedSkill]) -> None:
        self._skills = dict(skills)

    @property
    def manifests(self) -> tuple[SkillManifest, ...]:
        return tuple(skill.manifest for skill in self._skills.values())

    def __contains__(self, name: object) -> bool:
        return name in self._skills

    def tool_specs(self) -> list[ToolSpec]:
        return [_tool_spec(skill.manifest) for skill in self._skills.values()]

    def execute(self, name: str, arguments: Mapping[str, Any]) -> SkillResult:
        """Run a skill. Expected failures become an error result; bugs still raise."""
        skill = self._skills.get(name)
        if skill is None:
            return SkillResult(error=f"unknown skill: {name!r}")
        try:
            return SkillResult(output=skill.run(arguments))
        except SkillError as exc:
            return SkillResult(error=str(exc))


def discover_catalog(skills_dir: Path) -> SkillCatalog:
    if not skills_dir.is_dir():
        raise SkillManifestError(f"skills directory not found: {skills_dir}")
    skills: dict[str, _LoadedSkill] = {}
    for manifest_path in sorted(skills_dir.glob(f"*/{MANIFEST_FILENAME}")):
        manifest = load_manifest(manifest_path)
        if manifest.name in skills:
            raise SkillManifestError(f"duplicate skill name {manifest.name!r} in {manifest_path}")
        skills[manifest.name] = _LoadedSkill(manifest, _resolve(manifest))
    return SkillCatalog(skills)


def load_manifest(path: Path) -> SkillManifest:
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise SkillManifestError(f"{path}: invalid TOML: {exc}") from exc
    return SkillManifest(
        name=_str_field(data, "name", path),
        description=_str_field(data, "description", path),
        when_to_use=_str_field(data, "when_to_use", path),
        limitations=_str_field(data, "limitations", path),
        capabilities=_str_tuple_field(data, "capabilities", path),
        input_schema=_table_field(data, "input_schema", path),
        output_schema=_table_field(data, "output_schema", path),
        implementation=_str_field(data, "implementation", path),
        path=path,
    )


def _tool_spec(manifest: SkillManifest) -> ToolSpec:
    description = (
        f"{manifest.description.strip()}\n\n"
        f"When to use: {manifest.when_to_use.strip()}\n"
        f"Limitations: {manifest.limitations.strip()}\n"
        f"{PROVIDES_PREFIX}{', '.join(manifest.capabilities)}"
    )
    return ToolSpec(manifest.name, description, manifest.input_schema)


def _resolve(manifest: SkillManifest) -> SkillFunction:
    module_name, sep, attr = manifest.implementation.partition(":")
    if not sep or not module_name.startswith(TRUSTED_IMPLEMENTATION_PREFIX):
        raise SkillManifestError(
            f"{manifest.path}: implementation must be 'agentlab.<module>:<function>', "
            f"got {manifest.implementation!r}"
        )
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise SkillManifestError(f"{manifest.path}: cannot import {module_name!r}") from exc
    run = getattr(module, attr, None)
    if not callable(run):
        raise SkillManifestError(f"{manifest.path}: {manifest.implementation!r} is not callable")
    return cast("SkillFunction", run)


def _str_field(data: Mapping[str, Any], key: str, path: Path) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise SkillManifestError(f"{path}: '{key}' must be a non-empty string")
    return value


def _str_tuple_field(data: Mapping[str, Any], key: str, path: Path) -> tuple[str, ...]:
    value = data.get(key)
    if not isinstance(value, list) or not all(isinstance(v, str) for v in cast("list[Any]", value)):
        raise SkillManifestError(f"{path}: '{key}' must be a list of strings")
    return tuple(cast("list[str]", value))


def _table_field(data: Mapping[str, Any], key: str, path: Path) -> JSONObject:
    value = data.get(key)
    if not isinstance(value, dict):
        raise SkillManifestError(f"{path}: '[{key}]' table is required")
    return cast("JSONObject", value)
