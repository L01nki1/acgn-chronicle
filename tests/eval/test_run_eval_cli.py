"""CLI wiring tests for preflight-only and the abstention-calibration handoff.

These tests stub the settings/build/store/provider boundary; they do not touch the network or
write a real archive.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import run_eval  # noqa: E402

from acgn_rag.config import EmbedConfig, LLMConfig  # noqa: E402
from acgn_rag.core.eval import qa, runner  # noqa: E402
from acgn_rag.core.ground.generate import ANSWERER_VERSION  # noqa: E402

SNAPSHOT = "20261028abcdef"
CHUNKER = {
    "version": "composed-input-v2",
    "header_version": "h1",
    "config": "A",
    "max_tokens": 512,
    "overlap": 64,
    "max_seq_length": 512,
}
BUILD_IDENTITY = {
    "model": "fake",
    "backend": "torch",
    "device": "cuda",
    "precision": "fp32",
    "batch_size": 32,
    "identity_slug": "fake-torch-cuda-fp32-b32-12345678",
    "lib_torch": "2.15.0",
}


def _settings(root: Path, answer_threshold: float | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        paths=SimpleNamespace(index=root / "index", snapshots=root / "snap"),
        embed=EmbedConfig(),
        store=SimpleNamespace(over_fetch_factor=5),
        llm=LLMConfig(
            base_url="https://base.example/v1",
            model="base-model",
            model_version="v1",
            answer_base_url="https://answer.example/v1",
            answer_api_key="key",
            answer_model="answer-model",
            answer_model_version="v1",
            temperature=0.0,
            answer_min_retrieval_score=answer_threshold,
        ),
    )


def _stub_boundary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, threshold: float | None = None
):
    qa_path = ROOT / "eval" / "qa_set.jsonl"
    settings = _settings(tmp_path, threshold)
    monkeypatch.setattr(run_eval, "load_settings", lambda: settings)
    monkeypatch.setattr(
        run_eval, "resolve_window", lambda *args: ("2023-10-28", "2026-10-28", True)
    )
    monkeypatch.setattr(run_eval, "load_snapshot_member_urls", lambda path: frozenset())
    monkeypatch.setattr(
        run_eval,
        "load_build_records",
        lambda *args, **kwargs: (
            dict(BUILD_IDENTITY),
            "snap_20261028abc_A",
            {"docs": 36121, "chunks": 74245, "stored": 74245},
            dict(CHUNKER),
        ),
    )
    monkeypatch.setattr(run_eval, "verify_build_identity", lambda *args: [])
    monkeypatch.setattr(run_eval, "ChromaStore", lambda *args: SimpleNamespace())
    monkeypatch.setattr(
        run_eval, "bind_existing_collection", lambda *args, **kwargs: SimpleNamespace()
    )
    monkeypatch.setattr(run_eval, "build_evidence_resolver", lambda bound: lambda urls: {})
    monkeypatch.setattr(
        run_eval,
        "_configure_determinism",
        lambda **kwargs: {"gpu_model": "NVIDIA T4", "torch_seed": 0},
    )
    monkeypatch.setattr(run_eval, "_git_sha", lambda: "test-sha")
    monkeypatch.setattr(
        run_eval,
        "_build_answerer",
        lambda mode, config: SimpleNamespace(min_retrieval_score=config.answer_min_retrieval_score),
    )

    class _FakeEmbedder:
        def __init__(self, config: object) -> None:
            self.config = config

        def identity(self) -> object:
            return SimpleNamespace(to_metadata=lambda: dict(BUILD_IDENTITY))

    monkeypatch.setattr(run_eval, "Embedder", _FakeEmbedder)
    return run_eval, qa_path, settings


def test_preflight_only_writes_no_archive(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    run_eval, qa_path, _ = _stub_boundary(monkeypatch, tmp_path)
    called = False

    def _fail(*args: object, **kwargs: object) -> None:
        nonlocal called
        called = True
        raise AssertionError("run() must not be called for --preflight-only")

    monkeypatch.setattr(run_eval, "run", _fail)
    out = tmp_path / "runs"
    fingerprint = runner.qa_inputs_fingerprint(qa.load_jsonl(qa_path))

    code = run_eval.main(
        [
            "--config",
            "A",
            "--snapshot",
            SNAPSHOT,
            "--qa",
            str(qa_path),
            "--out",
            str(out),
            "--preflight-only",
            "--expect-docs",
            "36121",
            "--expect-chunks",
            "74245",
            "--expect-qa-fingerprint",
            fingerprint,
        ]
    )

    assert code == 0
    assert called is False
    assert not out.exists()


@pytest.mark.parametrize(
    ("flag", "value", "message"),
    [
        ("--expect-docs", "36000", "docs 36121 != expected 36000"),
        ("--expect-chunks", "74244", "chunks 74245 != expected 74244"),
        ("--expect-qa-fingerprint", "deadbeef", "qa fingerprint"),
    ],
)
def test_preflight_only_rejects_each_expectation_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    flag: str,
    value: str,
    message: str,
) -> None:
    run_eval, qa_path, _ = _stub_boundary(monkeypatch, tmp_path)
    called = False

    def _fail(*args: object, **kwargs: object) -> None:
        nonlocal called
        called = True
        raise AssertionError("run() must not be called for a failed preflight")

    monkeypatch.setattr(run_eval, "run", _fail)
    out = tmp_path / "runs"

    code = run_eval.main(
        [
            "--config",
            "A",
            "--snapshot",
            SNAPSHOT,
            "--qa",
            str(qa_path),
            "--out",
            str(out),
            "--preflight-only",
            flag,
            value,
        ]
    )

    assert code == 1
    assert message in capsys.readouterr().err
    assert called is False
    assert not out.exists()


def test_abstention_artifact_reaches_runspec(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    run_eval, qa_path, _ = _stub_boundary(monkeypatch, tmp_path, threshold=0.42)
    identity = runner.run_identity(
        snapshot_id=SNAPSHOT,
        config="A",
        chunker=CHUNKER,
        top_k=5,
        over_fetch=5,
        build_identity=BUILD_IDENTITY,
        query_transform="none",
        official_score_weight=1.0,
        answerer_version=ANSWERER_VERSION,
    )
    artifact = {
        "run_id": "20261115-cal-A-fake",
        "threshold": 0.42,
        "calibration": {"threshold": 0.42, "in_sample": True},
        "set": {"gate_set": True},
        "calibration_identity": identity,
    }
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    captured: dict[str, object] = {}

    def _capture(spec: runner.RunSpec, **kwargs: object) -> None:
        runner.validate_run_spec(spec)
        captured["spec"] = spec
        raise RuntimeError("stop after capture")

    monkeypatch.setattr(run_eval, "run", _capture)

    with pytest.raises(RuntimeError, match="stop after capture"):
        run_eval.main(
            [
                "--config",
                "A",
                "--snapshot",
                SNAPSHOT,
                "--qa",
                str(qa_path),
                "--answerer",
                "live",
                "--abstention-calibration",
                str(path),
            ]
        )

    spec = captured["spec"]
    assert isinstance(spec, runner.RunSpec)
    assert spec.abstention is not None
    assert spec.abstention["threshold"] == 0.42
    assert spec.abstention["source"] == str(path)
    assert spec.abstention["calibration_identity"] == identity
