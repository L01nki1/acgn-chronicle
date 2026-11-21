"""Measure answer density against the gold answer, for the answer-focus A/B.

The failure a labeller reported was not wrongness but dilution: the asked-for fact present, its
information density buried under surrounding event detail. Correctness alone cannot see that, so
this reports the three quantities that can:

  ratio      answer length / gold length, for items where the gold is short
  leads      does the answer open with the asked-for fact (gold's leading token)?
  contains   is the gold text still in the answer at all?

    .venv/Scripts/python.exe scripts/answer_density.py evalout/20261120-fc-on-r1 [...]
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

QA = Path("eval/qa_set_delivery.jsonl")


def load(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def resolve(qa: dict[str, dict], qa_id: str) -> dict | None:
    for candidate in (qa_id, f"delivery-{qa_id}", qa_id.removeprefix("delivery-")):
        if candidate in qa:
            return qa[candidate]
    return None


def leading_token(text: str) -> str:
    """The first meaningful token, used to ask 'does the answer lead with the fact?'."""
    cleaned = text.strip().strip('"“”')
    return cleaned.split(" ")[0].strip(" ,.;:()（）").lower() if cleaned else ""


def measure(run_dir: Path, qa: dict[str, dict]) -> dict | None:
    answers_path = run_dir / "answers.jsonl"
    if not answers_path.is_file():
        return None
    rows = load(answers_path)
    ratios: list[float] = []
    leads = 0
    contains = 0
    graded = 0
    for row in rows:
        if row.get("abstained"):
            continue
        item = resolve(qa, row["qa_id"])
        if item is None or not item.get("answerable"):
            continue
        gold = str(item.get("gold_answer") or "").strip()
        answer = str(row.get("answer") or "")
        if not gold or not answer:
            continue
        graded += 1
        ratios.append(len(answer) / max(len(gold), 1))
        if gold.lower() in answer.lower():
            contains += 1
        token = leading_token(gold)
        if token and answer.strip().lower().lstrip('"“”').startswith(token):
            leads += 1
    if not graded:
        return None
    return {
        "run": run_dir.name,
        "answered": len([r for r in rows if not r.get("abstained")]),
        "graded": graded,
        "ratio_mean": statistics.fmean(ratios),
        "ratio_median": statistics.median(ratios),
        "leads": leads,
        "contains_gold": contains,
    }


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    qa = {str(row["qa_id"]): row for row in load(QA)}
    print(
        f"{'run':<28}{'answered':>9}{'graded':>7}{'ratio(mean)':>12}{'median':>8}{'leads':>7}{'has gold':>10}"
    )
    print("-" * 82)
    for name in argv:
        result = measure(Path(name), qa)
        if result is None:
            print(f"{name:<28}  (no answers)")
            continue
        print(
            f"{result['run']:<28}{result['answered']:>9}{result['graded']:>7}"
            f"{result['ratio_mean']:>12.2f}{result['ratio_median']:>8.1f}"
            f"{result['leads']:>4}/{result['graded']:<2}{result['contains_gold']:>7}/{result['graded']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
