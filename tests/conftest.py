from __future__ import annotations

from pathlib import Path

import pytest

from agentlab.skills.catalog import SkillCatalog, discover_catalog

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture
def skills_dir() -> Path:
    return REPO_ROOT / "skills"


@pytest.fixture
def cases_dir() -> Path:
    return REPO_ROOT / "evals" / "cases"


@pytest.fixture
def catalog(skills_dir: Path) -> SkillCatalog:
    return discover_catalog(skills_dir)
