"""Summarise the answer-line arms of one run-id prefix into the four numbers.

    python experiments/summarize_arms.py 20261120-qd

Reports, per archive: answered / false-refused / unanswerable-correct / false-answers,
plus the abstain-reason split that tells `model_returned_no_claims` (the model wrote
nothing) apart from `all_claims_rejected` (the model wrote a claim and C4 dropped it).
The second distinction is the whole point of the quote-discipline experiment.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOTS = [Path("evalout"), Path("eval/runs")]
DELIVERY = Path("eval/qa_set_delivery.jsonl")


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def find_run(run_id: str) -> Path | None:
    for root in ROOTS:
        candidate = root / run_id
        if (candidate / "meta.json").is_file():
            return candidate
    return None


def abstain_reason(row: dict) -> str:
    code = (row.get("diagnostics") or {}).get("abstain_code")
    return str(code) if code else "<no code>"


def dropped_by_rule(row: dict) -> dict:
    return (row.get("diagnostics") or {}).get("dropped_by_rule") or {}


def summarize(run_id: str, qa: dict[str, dict]) -> dict | None:
    run = find_run(run_id)
    if run is None:
        print(f"  {run_id:<34} MISSING")
        return None
    meta = json.loads((run / "meta.json").read_text(encoding="utf-8"))
    answers = load_jsonl(run / "answers.jsonl")

    answered = 0
    false_refused = 0
    correct_refusals = 0
    false_answers = 0
    reasons: Counter[str] = Counter()
    answerable_refusal_reasons: Counter[str] = Counter()
    rules: Counter[str] = Counter()

    for row in answers:
        qid = row.get("qa_id") or row.get("id")
        item = qa.get(qid, {})
        answerable = bool(item.get("answerable", True))
        abstained = bool(row.get("abstained"))
        if not answerable:
            if abstained:
                correct_refusals += 1
            else:
                false_answers += 1
        else:
            if abstained:
                false_refused += 1
                answerable_refusal_reasons[abstain_reason(row)] += 1
                rules.update(dropped_by_rule(row))
            else:
                answered += 1
        if abstained:
            reasons[abstain_reason(row)] += 1

    n_ans = sum(
        1 for r in answers if qa.get(r.get("qa_id") or r.get("id"), {}).get("answerable", True)
    )
    n_unans = len(answers) - n_ans
    return {
        "run_id": run_id,
        "arm": meta.get("query_transform"),
        "bonus": meta.get("official_score_weight"),
        "n_answerable": n_ans,
        "n_unanswerable": n_unans,
        "answered": answered,
        "false_refused": false_refused,
        "false_refusal_rate": false_refused / n_ans if n_ans else float("nan"),
        "correct_refusals": correct_refusals,
        "false_answers": false_answers,
        "answerable_refusal_reasons": dict(answerable_refusal_reasons),
        "dropped_by_rule": dict(rules),
        "all_reasons": dict(reasons),
    }


def main(prefix: str) -> int:
    qa = {row["qa_id"]: row for row in load_jsonl(DELIVERY)}
    candidates: list[str] = []
    for root in ROOTS:
        if root.is_dir():
            for child in sorted(root.iterdir()):
                if (
                    child.is_dir()
                    and child.name.startswith(prefix)
                    and child.name not in candidates
                ):
                    candidates.append(child.name)

    if not candidates:
        print(f"no archives matching {prefix!r}")
        return 1

    rows = [r for r in (summarize(name, qa) for name in candidates) if r]
    if not rows:
        return 1

    header = f"{'run':<34}{'ans':>5}{'refus':>6}{'rate':>7}{'unans-ok':>9}{'false-ans':>10}"
    print()
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['run_id']:<34}{r['answered']:>5}{r['false_refused']:>6}"
            f"{r['false_refusal_rate']:>7.1%}"
            f"{r['correct_refusals']:>6}/{r['n_unanswerable']:<2}{r['false_answers']:>10}"
        )
        print(f"    answerable refusals by reason: {r['answerable_refusal_reasons']}")
        if r["dropped_by_rule"]:
            print(f"    dropped_by_rule:              {r['dropped_by_rule']}")

    # per-arm means, so three repeats are judged on the average rather than one draw
    for arm in ("off", "on"):
        group = [r for r in rows if f"-{arm}-" in r["run_id"]]
        if not group:
            continue
        mean_refused = sum(r["false_refused"] for r in group) / len(group)
        print(
            f"\narm {arm:<3} n={len(group)}  mean false-refused={mean_refused:.2f}"
            f" ({mean_refused / group[0]['n_answerable']:.1%})"
            f"  false-answers={sum(r['false_answers'] for r in group)}"
            f"  unanswerable-ok={sum(r['correct_refusals'] for r in group)}/{group[0]['n_unanswerable'] * len(group)}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "20261120-qd"))
