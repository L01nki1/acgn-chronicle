"""Checks for dependency requirements that support the project's storage contract."""

import tomllib
from pathlib import Path


def test_store_extra_declares_chroma_array_support_floor() -> None:
    manifest_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    project = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
    store_requirements = project["project"]["optional-dependencies"]["store"]

    assert "chromadb>=1.5.0,<2" in store_requirements, (
        "the store extra must require chromadb>=1.5.0,<2 for native array metadata"
    )
