from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from agentlab.skills.catalog import PROVIDES_PREFIX, SkillCatalog, discover_catalog
from agentlab.skills.models import SkillManifestError

if TYPE_CHECKING:
    from pathlib import Path

VALID_MANIFEST = """\
name = "{name}"
description = "Does a thing."
when_to_use = "When the thing is needed."
limitations = "Only the thing."
capabilities = ["thing"]
implementation = "{implementation}"

[input_schema]
type = "object"

[output_schema]
type = "object"
"""


def write_skill(
    root: Path,
    name: str = "calc",
    implementation: str = "agentlab.skills.builtin.calculator:run",
    text: str | None = None,
    dirname: str | None = None,
) -> None:
    skill_dir = root / (dirname or name)
    skill_dir.mkdir(parents=True)
    content = text or VALID_MANIFEST.format(name=name, implementation=implementation)
    (skill_dir / "skill.toml").write_text(content)


def test_discovers_repository_skills(catalog: SkillCatalog) -> None:
    assert [m.name for m in catalog.manifests] == ["calculator"]
    assert "calculator" in catalog


def test_tool_spec_exposes_manifest_metadata(catalog: SkillCatalog) -> None:
    (spec,) = catalog.tool_specs()
    assert spec.name == "calculator"
    assert "When to use:" in spec.description
    assert "Limitations:" in spec.description
    assert f"{PROVIDES_PREFIX}arithmetic" in spec.description
    assert spec.input_schema["required"] == ["expression"]


def test_execute_returns_structured_output(catalog: SkillCatalog) -> None:
    result = catalog.execute("calculator", {"expression": "120 * 3"})
    assert result.ok
    assert result.output == {"result": 360}


def test_execute_reports_skill_errors(catalog: SkillCatalog) -> None:
    result = catalog.execute("calculator", {"expression": "1 / 0"})
    assert not result.ok
    assert result.as_content() == {"error": "division by zero"}


def test_execute_reports_unknown_skill(catalog: SkillCatalog) -> None:
    result = catalog.execute("teleport", {})
    assert result.error == "unknown skill: 'teleport'"


def test_empty_directory_gives_empty_catalog(tmp_path: Path) -> None:
    assert discover_catalog(tmp_path).manifests == ()


def test_missing_directory_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(SkillManifestError, match="not found"):
        discover_catalog(tmp_path / "nope")


@pytest.mark.parametrize(
    "implementation",
    ["os:system", "builtins:eval", "agentlab.skills.builtin.calculator", "agentlab_evil.x:run"],
)
def test_rejects_untrusted_implementations(tmp_path: Path, implementation: str) -> None:
    write_skill(tmp_path, implementation=implementation)
    with pytest.raises(SkillManifestError, match="implementation must be"):
        discover_catalog(tmp_path)


@pytest.mark.parametrize(
    ("implementation", "message"),
    [
        ("agentlab.missing:run", "cannot import"),
        ("agentlab.skills.builtin.calculator:nope", "not callable"),
    ],
)
def test_rejects_unresolvable_implementations(
    tmp_path: Path, implementation: str, message: str
) -> None:
    write_skill(tmp_path, implementation=implementation)
    with pytest.raises(SkillManifestError, match=message):
        discover_catalog(tmp_path)


def test_rejects_duplicate_names(tmp_path: Path) -> None:
    write_skill(tmp_path, name="calc", dirname="one")
    write_skill(tmp_path, name="calc", dirname="two")
    with pytest.raises(SkillManifestError, match="duplicate skill name 'calc'"):
        discover_catalog(tmp_path)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("name = ", "invalid TOML"),
        ('name = "x"', "'description' must be a non-empty string"),
        (
            VALID_MANIFEST.format(name="x", implementation="a:b").replace(
                'capabilities = ["thing"]', 'capabilities = "thing"'
            ),
            "list of strings",
        ),
        (
            VALID_MANIFEST.format(name="x", implementation="a:b").split("[input_schema]")[0],
            r"'\[input_schema\]' table is required",
        ),
    ],
)
def test_rejects_malformed_manifests(tmp_path: Path, text: str, message: str) -> None:
    write_skill(tmp_path, text=text)
    with pytest.raises(SkillManifestError, match=message):
        discover_catalog(tmp_path)
