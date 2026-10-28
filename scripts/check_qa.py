"""CLI: check the QA set against the quotas in `docs/EVAL_PROTOCOL.md` §3.

Usage:
    python scripts/check_qa.py
    python scripts/check_qa.py --qa eval/qa_set.jsonl --calibration eval/abstain_calibration.jsonl

Exit codes: 0 = may be frozen, 1 = violations found, 2 = file not found.
Run this before declaring the set frozen; the protocol's numbers are only real if something enforces them.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from acgn_rag.core.eval.qa import load_jsonl, summary, validate


def _report(label: str, items: list[dict]) -> None:
    counts = summary(items)
    per_type = "  ".join(f"{k}={v}" for k, v in counts["by_type"].items())
    per_lang = "  ".join(f"{k}={v}" for k, v in counts["by_lang"].items())
    print(
        f"{label}: {counts['total']} items  ({per_type})  "
        f"lang ({per_lang})  "
        f"answerable={counts['answerable']}  unanswerable={counts['unanswerable']}  "
        f"overlap={counts['overlap']}  urls={counts['urls']}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Check the QA set against the evaluation protocol")
    parser.add_argument("--qa", default="eval/qa_set.jsonl", help="QA set (JSONL)")
    parser.add_argument(
        "--calibration",
        default="eval/abstain_calibration.jsonl",
        help="abstention calibration set (JSONL); skipped when absent",
    )
    args = parser.parse_args()

    qa_path = Path(args.qa)
    if not qa_path.exists():
        print(f"NOT FOUND: {qa_path} - nothing to check yet")
        return 2

    items = load_jsonl(qa_path)
    _report(qa_path.as_posix(), items)

    calibration = None
    cal_path = Path(args.calibration)
    if cal_path.exists():
        calibration = load_jsonl(cal_path)
        _report(cal_path.as_posix(), calibration)
    else:
        print(f"note: {cal_path} absent - disjointness from the calibration set is unchecked")

    problems = validate(items, calibration=calibration)
    if problems:
        print(f"\n{len(problems)} problem(s):")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    print("\nOK: the set satisfies the protocol and may be frozen")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
