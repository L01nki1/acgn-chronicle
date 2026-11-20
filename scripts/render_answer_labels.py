"""Render a run archive's answered items as a self-contained human labelling page.

Owner: D (WP5) | Spec: docs/EVAL_PROTOCOL.md 5 (answer correctness is human judgement)

`EVAL_PROTOCOL.md` 5 says a human assigns `fully_correct` / `partially_correct` / `incorrect`
and that code never decides a label. This script produces the page that makes that possible
without a server: it inlines the run's items, the gold row, the generated answer and every
citation into one HTML file that works from `file://`.

The page is the only thing a labeller touches. It edits `labels.jsonl` **in place**, so a
finished sheet is exactly the artifact `scripts/score_labels.py` reads to fill in
`metrics.json` correctness.

    .venv/Scripts/python.exe scripts/render_answer_labels.py \
        --run evalout/20261120-qd-on-r2 \
        --out evalout/20261120-qd-on-r2/label_page.html

Only answered, answerable items are shown. An item that abstained has no answer to grade, and
an unanswerable item is graded by the abstention metric instead.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

TEMPLATE = Path(__file__).resolve().parent / "answer_label_template.html"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _existing_labels(path: Path) -> dict[str, dict[str, Any]]:
    """Labels already written for this run, so reopening the page resumes rather than resets."""
    return {str(row["qa_id"]): row for row in _read_jsonl(path) if row.get("qa_id")}


def _resolve(gold: dict[str, dict[str, Any]], qa_id: str) -> dict[str, Any] | None:
    """Find the QA row for an archived id.

    Archives written before the delivery set gained its `delivery-` namespace carry bare ids
    (`A-001`), and re-running every archive would be wasteful when the answer text is unchanged.
    The namespace is the only difference, so try both spellings rather than failing.
    """
    for candidate in (qa_id, f"delivery-{qa_id}", qa_id.removeprefix("delivery-")):
        if candidate in gold:
            return gold[candidate]
    return None


def build_items(run_dir: Path, qa_path: Path) -> list[dict[str, Any]]:
    qa = {str(row["qa_id"]): row for row in _read_jsonl(qa_path)}
    answers = _read_jsonl(run_dir / "answers.jsonl")
    existing = _existing_labels(run_dir / "labels.jsonl")

    items: list[dict[str, Any]] = []
    for answer in answers:
        qa_id = str(answer["qa_id"])
        gold = _resolve(qa, qa_id)
        if gold is None or not bool(gold.get("answerable", True)) or bool(answer.get("abstained")):
            continue
        canonical = str(gold["qa_id"])
        evidence = gold.get("gold_evidence") or {}
        prior = existing.get(canonical) or existing.get(qa_id) or {}
        items.append(
            {
                "qa_id": canonical,
                "archived_id": qa_id,
                "type": gold.get("type"),
                "lang": gold.get("lang"),
                "question": gold.get("question"),
                "gold_answer": gold.get("gold_answer"),
                "gold_spans": list(evidence.get("spans") or []),
                "gold_urls": list(evidence.get("urls") or []),
                "key_entities": list(gold.get("key_entities") or []),
                "answer": answer.get("answer") or "",
                "citations": [
                    {
                        "n": index + 1,
                        "publisher": citation.get("publisher"),
                        "title": citation.get("title"),
                        "url": citation.get("url"),
                        "source_type": citation.get("source_type"),
                        "published_at": citation.get("published_at"),
                        "in_gold": str(citation.get("url") or "")
                        in set(evidence.get("urls") or []),
                    }
                    for index, citation in enumerate(answer.get("citations") or [])
                ],
                "label": prior.get("label"),
                "note": prior.get("note") or "",
            }
        )
    return items


def render(items: Sequence[dict[str, Any]], template: str, *, run_id: str, source: str) -> str:
    payload = json.dumps(list(items), ensure_ascii=False, separators=(",", ":"))
    # A literal `</script>` inside a headline would close the embedding block early.
    payload = payload.replace("</", "<\\/")
    return (
        template.replace("/*__ITEMS__*/", payload)
        .replace("/*__RUN_ID__*/", json.dumps(run_id, ensure_ascii=False))
        .replace("/*__SOURCE__*/", json.dumps(source, ensure_ascii=False))
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render a human answer-labelling page")
    parser.add_argument("--run", required=True, help="run archive directory")
    parser.add_argument("--qa", default="eval/qa_set_delivery.jsonl")
    parser.add_argument("--out", default=None, help="defaults to <run>/label_page.html")
    parser.add_argument("--template", default=str(TEMPLATE))
    args = parser.parse_args(argv)

    run_dir = Path(args.run)
    if not (run_dir / "answers.jsonl").is_file():
        print(f"no answers.jsonl under {run_dir}", file=sys.stderr)
        return 2
    items = build_items(run_dir, Path(args.qa))
    if not items:
        print("no answered, answerable items to label", file=sys.stderr)
        return 2

    already = sum(1 for item in items if item.get("label"))
    page = render(
        items,
        Path(args.template).read_text(encoding="utf-8"),
        run_id=run_dir.name,
        source=str(Path(args.qa)),
    )
    out = Path(args.out) if args.out else run_dir / "label_page.html"
    out.write_text(page, encoding="utf-8", newline="\n")
    print(f"{out}: {len(items)} item(s), {already} already labelled")
    print(f"labels file the page overwrites: {run_dir / 'labels.jsonl'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
