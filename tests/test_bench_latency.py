import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "bench_latency", ROOT / "scripts" / "bench_latency.py"
)
assert SPEC is not None and SPEC.loader is not None
bench_latency = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench_latency)


def test_nearest_rank_percentile_on_thirty_samples() -> None:
    values = list(range(1, 31))
    assert bench_latency.percentile(values, 50) == 15
    assert bench_latency.percentile(values, 95) == 29
    assert bench_latency.percentile([7.0], 95) == 7.0


def test_widest_window_from_qa_set(tmp_path: Path) -> None:
    path = tmp_path / "qa.jsonl"
    items = [
        {"qa_id": "A-001", "window": ["2023-10-28", "2026-10-28"]},
        {"qa_id": "B-001", "window": ["2023-11-05", "2026-11-05"]},
    ]
    path.write_text("\n".join(json.dumps(item) for item in items) + "\n", encoding="utf-8")
    assert bench_latency.window_from_qa_set(path) == ("2023-10-28", "2026-11-05", 2)


def test_artifact_schema_is_fixed() -> None:
    artifact = bench_latency.build_artifact(
        gate_eligible=True,
        gate_ineligible_reasons=[],
        gate_shape={},
        environment={},
        code={},
        index={},
        latency={},
    )
    assert set(artifact) == set(bench_latency.ARTIFACT_SECTIONS)
    assert artifact["artifact_version"] == 1


def test_gate_eligibility_reasons() -> None:
    good = dict(
        chunk_config="A",
        qa_set=Path("eval/qa_set.jsonl"),
        dirty=False,
        sha="abc123",
        model="test CPU",
        sidecars={"build.json": {"ok": True}, "identity.json": {"ok": True}},
    )
    assert bench_latency.gate_ineligible_reasons(**good) == []
    assert (
        "Config A only"
        in bench_latency.gate_ineligible_reasons(**{**good, "chunk_config": "A-prime"})[0]
    )
    assert any(
        "QA per-item window" in reason
        for reason in bench_latency.gate_ineligible_reasons(**{**good, "qa_set": None})
    )
    assert "git worktree is dirty" in bench_latency.gate_ineligible_reasons(
        **{**good, "dirty": True}
    )
    assert "CPU model unavailable" in bench_latency.gate_ineligible_reasons(
        **{**good, "model": "unknown"}
    )
    assert any(
        "sidecars missing" in reason
        for reason in bench_latency.gate_ineligible_reasons(
            **{**good, "sidecars": {"build.json": None, "identity.json": None}}
        )
    )
