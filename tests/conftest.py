from __future__ import annotations

from pathlib import Path

import pytest

from agentlab.tools.catalog import ToolCatalog, discover_catalog

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture
def tools_dir() -> Path:
    return REPO_ROOT / "tools"


@pytest.fixture
def cases_dir() -> Path:
    return REPO_ROOT / "evals" / "cases"


@pytest.fixture
def catalog(tools_dir: Path) -> ToolCatalog:
    return discover_catalog(tools_dir)
