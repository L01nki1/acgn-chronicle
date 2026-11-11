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
        "answer_model_version": "v1",
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
        [
            {
                "qa_id": "A-001",
                "status": "answered",
                "answer": "ok",
                "provider_completion": True,
            }
        ],
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


def test_live_validation_rejects_an_all_refusal_archive(tmp_path: Path) -> None:
    result = _live_result(
        tmp_path,
        [
            {
                "qa_id": "A-001",
                "status": "answered",
                "answer": "I cannot answer that from the evidence.",
                "abstained": True,
                "provider_completion": False,
            }
        ],
        _live_llm(),
    )

    assert any(
        "no returned provider completion" in problem
        for problem in run_eval._validate_live_run(result)
    )


def test_live_validation_accepts_mixed_completion_and_refusal(tmp_path: Path) -> None:
    result = _live_result(
        tmp_path,
        [
            {
                "qa_id": "A-001",
                "status": "answered",
                "answer": "The answer.",
                "provider_completion": True,
            },
            {
                "qa_id": "A-002",
                "status": "answered",
                "answer": "I cannot answer that from the evidence.",
                "abstained": True,
                "provider_completion": False,
            },
        ],
        _live_llm(),
    )

    assert run_eval._validate_live_run(result) == []


def test_live_validation_rejects_missing_gpu_model_on_cuda(tmp_path: Path) -> None:
    result = _live_result(
        tmp_path,
        [
            {
                "qa_id": "A-001",
                "status": "answered",
                "answer": "ok",
                "provider_completion": True,
            }
        ],
        _live_llm(),
    )
    result.meta["build_identity"] = {"device": "cuda:1"}
    result.meta["determinism"] = {"gpu_model": None}

    assert any("gpu_model" in problem for problem in run_eval._validate_live_run(result))


def test_live_metadata_requires_answer_version_when_models_differ() -> None:
    config = _config(answer_model_version="", model="extract-model", answer_model="answer-model")

    problems = run_eval._live_metadata_problems(config, run_eval._llm_metadata(config, "live"))

    assert any("ACGN_ANSWER_MODEL_VERSION" in problem for problem in problems)


def test_live_metadata_prefers_the_answer_model_version() -> None:
    config = _config(answer_model_version="answer-v2", model_version="extract-v1")

    assert run_eval._llm_metadata(config, "live")["version"] == "answer-v2"


def test_live_validation_rejects_a_non_null_seed(tmp_path: Path) -> None:
    result = _live_result(
        tmp_path,
        [
            {
                "qa_id": "A-001",
                "status": "answered",
                "answer": "ok",
                "provider_completion": True,
            }
        ],
        _live_llm(seed=0),
    )

    assert any("seed must be null" in problem for problem in run_eval._validate_live_run(result))


def test_gpu_model_is_recorded_in_determinism() -> None:
    determinism = run_eval._configure_determinism()

    assert "gpu_model" in determinism
    assert determinism["gpu_model"] is None or isinstance(determinism["gpu_model"], str)


def _fake_torch(
    *, cuda_available: bool, name: str = "Tesla T4", names: dict[int, str] | None = None
) -> SimpleNamespace:
    return SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: cuda_available,
            get_device_name=lambda index: (names or {}).get(index, name),
        ),
        manual_seed=lambda seed: None,
        use_deterministic_algorithms=lambda *args, **kwargs: None,
    )


def test_gpu_model_records_the_cuda_device(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda_available=True))

    determinism = run_eval._configure_determinism(device="cuda")

    assert determinism["gpu_model"] == "Tesla T4"
    assert determinism["torch_available"] is True


def test_gpu_model_uses_the_encoder_device_index(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(
        sys.modules, "torch", _fake_torch(cuda_available=True, names={0: "GPU-0", 1: "GPU-1"})
    )

    determinism = run_eval._configure_determinism(device="cuda:1")

    assert determinism["gpu_model"] == "GPU-1"


def test_gpu_model_is_null_for_cpu_on_a_gpu_host(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda_available=True))

    determinism = run_eval._configure_determinism(device="cpu")

    assert determinism["gpu_model"] is None


def test_gpu_model_is_null_without_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda_available=False))

    determinism = run_eval._configure_determinism(device="cuda")

    assert determinism["gpu_model"] is None


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


def test_main_live_missing_answer_version_fails_before_answerer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    qa_path = tmp_path / "qa.jsonl"
    qa_path.write_text('{"qa_id": "A-001"}\n', encoding="utf-8")
    settings = SimpleNamespace(
        llm=_config(
            answer_model_version="",
            model="extract-model",
            answer_model="answer-model",
        )
    )
    monkeypatch.setattr(run_eval, "load_settings", lambda: settings)
    monkeypatch.setattr(
        run_eval,
        "_build_answerer",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("answerer built")),
    )

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
