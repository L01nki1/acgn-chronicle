"""`config.load_settings()` reads the project `.env` without overriding real environment vars.

Owner: D | Spec: issue #71, docs/RUNBOOK.md §3.6

`.env.example` tells every member to "copy to .env and fill in", so the file has to be read by
the code, not only by the shell. These tests pin the three properties that make that safe:
the file is found from any working directory, a real environment variable always wins, and a
missing file is not an error.

Every test points the loader at a temporary `.env` through an explicit ``start`` path, so a
developer's own `.env` at the repository root can never leak into a test result.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from acgn_rag import config


def _write_env(path: Path, **values: str) -> Path:
    path.write_text(
        "".join(f"{key}={value}\n" for key, value in values.items()),
        encoding="utf-8",
    )
    return path


def test_load_env_file_reads_the_given_directory(tmp_path: Path) -> None:
    _write_env(tmp_path / ".env", ACGN_EXTRACT_MODEL="from-dotenv")

    loaded = config.load_env_file(start=tmp_path)

    assert loaded == tmp_path / ".env"


def test_real_environment_variable_wins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_env(tmp_path / ".env", ACGN_EXTRACT_MODEL="from-dotenv")
    monkeypatch.setenv("ACGN_EXTRACT_MODEL", "from-shell")

    config.load_env_file(start=tmp_path)

    assert config.LLMConfig().extract_model == "from-shell"


def test_missing_env_file_is_not_an_error(tmp_path: Path) -> None:
    empty = tmp_path / "no-env-here"
    empty.mkdir()

    assert config.load_env_file(start=empty) is None


def test_env_file_is_found_from_another_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A script run from a subdirectory must see the same configuration."""
    (tmp_path / "scripts").mkdir()
    _write_env(tmp_path / ".env", ACGN_ANSWER_MODEL="root-model")
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    monkeypatch.delenv("ACGN_ANSWER_MODEL", raising=False)
    monkeypatch.chdir(tmp_path / "scripts")

    assert config.load_env_file() == tmp_path / ".env"
    assert config.LLMConfig().answer_model == "root-model"


def test_project_root_is_the_repository_checkout() -> None:
    assert (config.PROJECT_ROOT / "pyproject.toml").is_file()
    assert (config.PROJECT_ROOT / "src" / "acgn_rag" / "config.py").is_file()


def test_loading_twice_does_not_change_the_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_env(tmp_path / ".env", ACGN_EXTRACT_MODEL="first")
    monkeypatch.delenv("ACGN_EXTRACT_MODEL", raising=False)

    config.load_env_file(start=tmp_path)
    _write_env(tmp_path / ".env", ACGN_EXTRACT_MODEL="second")
    config.load_env_file(start=tmp_path)

    # dotenv does not override an already-set variable: the first load wins for this process.
    assert config.LLMConfig().extract_model == "first"


def test_load_settings_loads_the_project_env_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The wiring itself: `load_settings()` must call the loader, not just expose it."""
    _write_env(tmp_path / ".env", ACGN_ANSWER_MODEL="wired-model")
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    monkeypatch.delenv("ACGN_ANSWER_MODEL", raising=False)

    assert config.load_settings().llm.answer_model == "wired-model"
