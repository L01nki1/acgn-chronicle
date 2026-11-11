"""Tests for the run_eval answerer switch and LLM provenance."""

from __future__ import annotations

import json
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
    assert metadata["seed"] is None


def test_null_llm_metadata_keeps_the_base_line() -> None:
    metadata = run_eval._llm_metadata(_config(), "null")

    assert metadata["provider"] == "https://base.example/v1"
    assert metadata["model"] == "base-model"
    assert metadata["seed"] is None


def _live_llm(**overrides: object) -> dict[str, object]:
    values = {
        "provider": "https://answer.example/v1",
        "model": "answer-model",
        "version": "v1",
        "temperature": 0.0,
        "seed": None,
    }
    values.update(overrides)
    return values


def _live_result(
    out_dir: Path, rows: list[dict[str, object]], llm: dict[str, object]
) -> SimpleNamespace:
    (out_dir / "answers.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    return SimpleNamespace(out_dir=out_dir, meta={"llm": llm})


def test_live_validation_accepts_an_answered_run(tmp_path: Path) -> None:
    result = _live_result(
        tmp_path,
        [{"qa_id": "A-001", "status": "answered", "answer": "ok"}],
        _live_llm(),
    )

    assert run_eval._validate_live_run(result) == []


def test_live_validation_rejects_unanswered_and_missing_provenance(tmp_path: Path) -> None:
    result = _live_result(
        tmp_path,
        [
            {"qa_id": "A-001", "status": "not_run", "answer": None},
            {"qa_id": "A-002", "status": "answered", "answer": "  "},
        ],
        _live_llm(provider="none", version="none"),
    )

    problems = run_eval._validate_live_run(result)

    assert any("2/2 item(s) not answered" in problem for problem in problems)
    assert any("meta.llm.provider" in problem for problem in problems)
    assert any("meta.llm.version" in problem for problem in problems)


def test_live_validation_rejects_an_empty_archive(tmp_path: Path) -> None:
    result = _live_result(tmp_path, [], _live_llm())

    assert "answers.jsonl is empty" in run_eval._validate_live_run(result)


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
