"""Tests for the run_eval answerer switch and LLM provenance."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import run_eval  # noqa: E402

from acgn_rag.config import LLMConfig  # noqa: E402
from acgn_rag.core.eval.runner import NullAnswerer  # noqa: E402


def _config(**overrides: object) -> LLMConfig:
    values = {
        "base_url": "https://base.example/v1",
        "model": "base-model",
        "model_version": "v1",
        "answer_base_url": "https://answer.example/v1",
        "answer_api_key": "key",
        "answer_model": "answer-model",
        "temperature": 0.0,
    }
    values.update(overrides)
    return LLMConfig(**values)  # type: ignore[arg-type]


def test_null_answerer_is_the_default() -> None:
    assert isinstance(run_eval._build_answerer("null", _config()), NullAnswerer)


def test_live_answerer_requires_answer_line_key_and_model() -> None:
    with pytest.raises(ValueError, match="ACGN_ANSWER_API_KEY"):
        run_eval._build_answerer("live", _config(answer_api_key=""))
    with pytest.raises(ValueError, match="ACGN_ANSWER_MODEL"):
        run_eval._build_answerer("live", _config(answer_model=""))


def test_live_llm_metadata_uses_the_answer_line() -> None:
    metadata = run_eval._llm_metadata(_config(temperature=0.2), "live")

    assert metadata["provider"] == "https://answer.example/v1"
    assert metadata["model"] == "answer-model"
    assert metadata["version"] == "v1"
    assert metadata["temperature"] == 0.2


def test_null_llm_metadata_keeps_the_base_line() -> None:
    metadata = run_eval._llm_metadata(_config(), "null")

    assert metadata["provider"] == "https://base.example/v1"
    assert metadata["model"] == "base-model"


def test_main_live_without_key_fails_before_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    qa_path = tmp_path / "qa.jsonl"
    qa_path.write_text('{"qa_id": "A-001"}\n', encoding="utf-8")
    settings = SimpleNamespace(llm=_config(answer_api_key="", answer_model=""))
    monkeypatch.setattr(run_eval, "load_settings", lambda: settings)

    code = run_eval.main(
        [
            "--config",
            "A",
            "--snapshot",
            "20261105-d797b5f0ae8e",
            "--qa",
            str(qa_path),
            "--allow-invalid-qa",
            "--answerer",
            "live",
        ]
    )

    assert code == 2
