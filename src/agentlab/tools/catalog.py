"""Tool discovery and execution.

Discovery reads ``<tools_dir>/*/tool.toml``. Each manifest names its implementation as
``module:function``; implementations must live inside the ``agentlab`` package, so nothing
from the tools directory itself is ever executed. Learned tools join through
``with_tools``; their functions run generated code only in the sandbox (``agentlab.learning``).
"""

from __future__ import annotations

import importlib
import tomllib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from agentlab.models import ToolSpec
from agentlab.tools.models import ToolError, ToolManifest, ToolManifestError, ToolResult

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping
    from pathlib import Path

    from agentlab.models import JSONObject

type ToolFunction = Callable[[Mapping[str, Any]], JSONObject]

MANIFEST_FILENAME = "tool.toml"
TRUSTED_IMPLEMENTATION_PREFIX = "agentlab."
PROVIDES_PREFIX = "Provides: "


@dataclass(frozen=True)
class _LoadedTool:
    manifest: ToolManifest
    run: ToolFunction


class ToolCatalog:
    """The tools available to an agent, keyed by name."""

    def __init__(self, tools: Mapping[str, _LoadedTool]) -> None:
        self._tools = dict(tools)

    @property
    def manifests(self) -> tuple[ToolManifest, ...]:
        return tuple(tool.manifest for tool in self._tools.values())

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def with_tools(self, tools: Iterable[tuple[ToolManifest, ToolFunction]]) -> ToolCatalog:
        """A new catalog with extra tools (e.g. learned ones). Names must stay unique."""
        merged = dict(self._tools)
        for manifest, run in tools:
            if manifest.name in merged:
                raise ToolManifestError(f"duplicate tool name {manifest.name!r} in {manifest.path}")
            merged[manifest.name] = _LoadedTool(manifest, run)
        return ToolCatalog(merged)

    def tool_specs(self) -> list[ToolSpec]:
        return [_tool_spec(tool.manifest) for tool in self._tools.values()]

    def execute(self, name: str, arguments: Mapping[str, Any]) -> ToolResult:
        """Run a tool. Expected failures become an error result; bugs still raise."""
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(error=f"unknown tool: {name!r}")
        try:
            return ToolResult(output=tool.run(arguments))
        except ToolError as exc:
            return ToolResult(error=str(exc))


def discover_catalog(tools_dir: Path) -> ToolCatalog:
    if not tools_dir.is_dir():
        raise ToolManifestError(f"tools directory not found: {tools_dir}")
    tools: dict[str, _LoadedTool] = {}
    for manifest_path in sorted(tools_dir.glob(f"*/{MANIFEST_FILENAME}")):
        manifest = load_manifest(manifest_path)
        if manifest.name in tools:
            raise ToolManifestError(f"duplicate tool name {manifest.name!r} in {manifest_path}")
        tools[manifest.name] = _LoadedTool(manifest, _resolve(manifest))
    return ToolCatalog(tools)


def load_manifest(path: Path) -> ToolManifest:
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ToolManifestError(f"{path}: invalid TOML: {exc}") from exc
    return manifest_from_data(data, path)


def manifest_from_data(data: Mapping[str, Any], path: Path) -> ToolManifest:
    """Validate manifest fields. Shared by ``tool.toml`` discovery and learned tools."""
    return ToolManifest(
        name=_str_field(data, "name", path),
        description=_str_field(data, "description", path),
        when_to_use=_str_field(data, "when_to_use", path),
        limitations=_str_field(data, "limitations", path),
        capabilities=_str_tuple_field(data, "capabilities", path),
        input_schema=_table_field(data, "input_schema", path),
        output_schema=_table_field(data, "output_schema", path),
        implementation=_str_field(data, "implementation", path),
        path=path,
        reads_files=data.get("reads_files") is True,
    )


def _tool_spec(manifest: ToolManifest) -> ToolSpec:
    description = (
        f"{manifest.description.strip()}\n\n"
        f"When to use: {manifest.when_to_use.strip()}\n"
        f"Limitations: {manifest.limitations.strip()}\n"
        f"{PROVIDES_PREFIX}{', '.join(manifest.capabilities)}"
    )
    return ToolSpec(manifest.name, description, manifest.input_schema)


def _resolve(manifest: ToolManifest) -> ToolFunction:
    module_name, sep, attr = manifest.implementation.partition(":")
    if not sep or not module_name.startswith(TRUSTED_IMPLEMENTATION_PREFIX):
        raise ToolManifestError(
            f"{manifest.path}: implementation must be 'agentlab.<module>:<function>', "
            f"got {manifest.implementation!r}"
        )
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ToolManifestError(f"{manifest.path}: cannot import {module_name!r}") from exc
    run = getattr(module, attr, None)
    if not callable(run):
        raise ToolManifestError(f"{manifest.path}: {manifest.implementation!r} is not callable")
    return cast("ToolFunction", run)


def _str_field(data: Mapping[str, Any], key: str, path: Path) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ToolManifestError(f"{path}: '{key}' must be a non-empty string")
    return value


def _str_tuple_field(data: Mapping[str, Any], key: str, path: Path) -> tuple[str, ...]:
    value = data.get(key)
    if not isinstance(value, list) or not all(isinstance(v, str) for v in cast("list[Any]", value)):
        raise ToolManifestError(f"{path}: '{key}' must be a list of strings")
    return tuple(cast("list[str]", value))


def _table_field(data: Mapping[str, Any], key: str, path: Path) -> JSONObject:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ToolManifestError(f"{path}: '[{key}]' table is required")
    return cast("JSONObject", value)
