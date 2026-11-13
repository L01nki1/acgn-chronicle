"""Check whether the headline-only QA representatives actually state their gold answer.

Owner: D (WP5 tooling) | Spec: docs/EVAL_PROTOCOL.md, issue #6 comment 6743557815, issue #44
       (derived from F's URL-membership precheck; the PR and the tool are D-authored)

F's URL-membership precheck found 13 answerable QA items whose only URL-matched representative
has `body == title` (a headline-only record, `body_level="headline"`). A headline is still text:
if it states the gold answer, the item is answerable from the index and the QA set needs no
change. This script answers that question mechanically and prints a table a human can check:

    python scripts/qa_headline_check.py --snapshot 20261105-d797b5f0ae8e

For every affected item it reports whether each `key_entities` value occurs, after NFKC + case
folding, in (a) the representative's title, (b) the representative's body, and (c) the title in
an equivalent form (`ANSWER_FORMS`). `gold_evidence.spans` are human locators, not the automatic
hit rule, so they are reported separately and never used as the verdict. The verdict is a
*reading aid*: a title that states the gold answer is evidence the item is answerable, not proof
that the question is unambiguous.
"""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from pathlib import Path
from typing import Any

from acgn_rag.core.ingest.snapshot import iter_snapshot_items

#: F's precheck (issue #6 comment 6743557815): QA id -> the representative doc_id whose body
#: equals its title. Kept here as data so the check is reproducible without re-deriving it.
#: This map is QA-set-bound: if an item is renumbered (e.g. A-006/B-005 -> D-911/D-912 under
#: option 2), update it here. `check()` rejects ids absent from the QA set instead of treating
#: them as empty rows.
HEADLINE_ONLY: dict[str, str] = {
    "A-001": "doc_8b75231893838a22db1d43744ec73d2b59ee7e7829d6f9e1ad60bdec62423360",
    "A-003": "doc_18f945682cc01f0ac1bb409961a4df75da4c4cb6569f7e8a126b5ac911e77f13",
    "A-006": "doc_5264985cdd77c98f50a581ac6bc44a4b60fbd2db7ef9d042115bb4480f227406",
    "A-007": "doc_e5f06d312334d9c0a4d9075cb9b62979bfa601ee890c8b71cad79f2079209420",
    "A-010": "doc_3b9cb7c315ce52a6fd50090916bec4d0c85c111e9f8290785838bb054916f0c6",
    "B-002": "doc_74a70264a2538504dbb2f243a797d73547e057fb1ad0a905333fec23ddcf3af7",
    "B-005": "doc_cf12523fcf5bd7cd84d5a434cb0048db0ecf1d17d9d9ade195a0f7d65d6908ee",
    "B-006": "doc_46cc35d3e1e95890ba80bc463ca1820dba602c69dbef198c1910e79165e25540",
    "B-007": "doc_d505366fd280a663d4963abde95f4c1bebc0c237afb3458503ac72ce851ba930",
    "B-008": "doc_5ec15a1c34aa77d0a542d48baa04aeaa5b6b7b0839266a968176103015a1d61f",
    # C-003 needs both representatives; the second one is the ONE PIECE store item.
    "C-003": "doc_3b9cb7c315ce52a6fd50090916bec4d0c85c111e9f8290785838bb054916f0c6",
}
SECOND_REPRESENTATIVE = {
    "C-003": "doc_5ec15a1c34aa77d0a542d48baa04aeaa5b6b7b0839266a968176103015a1d61f",
}

#: The *answer* stated in another form. "October 3 Debut" answers "October 2025" only when the
#: question asks for a premiere window; it does not answer "which platform", so B-005's
#: "Netflix" is deliberately absent here. Bare "October" is deliberately absent too: it would
#: accept a headline about a different year. Each entry is the reading a human would accept.
ANSWER_FORMS: dict[str, tuple[str, ...]] = {
    "October 2025": ("October Debut", "October 3 Debut"),
    "2025-11-14": ("2025年11月14日", "11月14日"),
    "70th": ("70周年", "70th"),
    "97th": ("第97回", "97th"),
    "第97届": ("第97回", "97th"),
    "August 2024": ("2024年8月", "8月"),
}


def _norm(text: str) -> str:
    return unicodedata.normalize("NFKC", text).lower()


def _contains(haystack: str, needle: str) -> bool:
    return bool(needle) and _norm(needle) in _norm(haystack)


def load_qa(path: Path) -> dict[str, dict[str, Any]]:
    # Iterate the file object rather than str.splitlines(): splitlines() also breaks on
    # U+2028 / U+2029 / U+0085, which are legal inside a JSON string in a JP/ZH QA row and
    # would split one record in two. Iteration splits on "\n" only.
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return {str(row["qa_id"]): row for row in rows}


def check(snapshot: str, data_root: Path, qa_path: Path) -> dict[str, Any]:
    qa = load_qa(qa_path)
    missing = sorted(set(HEADLINE_ONLY) - set(qa))
    if missing:
        raise ValueError(
            "HEADLINE_ONLY ids absent from the QA set: "
            + ", ".join(missing)
            + " — update the map when QA ids change (e.g. A-006/B-005 -> D-911/D-912 under "
            "option 2); a silent empty row would make the check look clean"
        )
    wanted = set(HEADLINE_ONLY.values()) | set(SECOND_REPRESENTATIVE.values())
    items = {
        item.doc_id: item
        for item in iter_snapshot_items(snapshot, data_root=data_root, verify=True)
        if item.doc_id in wanted
    }

    results: list[dict[str, Any]] = []
    for qa_id, doc_id in HEADLINE_ONLY.items():
        item = items.get(doc_id)
        if item is None:
            results.append({"qa_id": qa_id, "doc_id": doc_id, "status": "representative_not_found"})
            continue
        entities = [str(value) for value in qa.get(qa_id, {}).get("key_entities", [])]
        in_title = [value for value in entities if _contains(item.title, value)]
        in_body = [value for value in entities if _contains(item.body, value)]
        equivalent = [
            value
            for value in entities
            if value not in in_title
            and any(_contains(item.title, form) for form in ANSWER_FORMS.get(value, ()))
        ]
        spans = qa.get(qa_id, {}).get("gold_evidence", {}).get("spans", [])
        span_in_body = [span for span in spans if _contains(item.body, str(span))]
        second = items.get(SECOND_REPRESENTATIVE.get(qa_id, ""))
        results.append(
            {
                "qa_id": qa_id,
                "doc_id": doc_id,
                "source_key": item.source_key,
                "body_equals_title": item.body == item.title,
                "title_chars": len(item.title),
                "title": item.title,
                "key_entities": entities,
                "entities_in_title": in_title,
                "entities_in_title_equivalent_form": equivalent,
                "entities_in_body": in_body,
                "spans_in_body": span_in_body,
                "second_representative": (
                    None
                    if second is None
                    else {
                        "doc_id": second.doc_id,
                        "title": second.title,
                        "body_equals_title": second.body == second.title,
                        "entities_in_title": [
                            value
                            for value in entities
                            if _contains(second.title, value)
                            or any(
                                _contains(second.title, form)
                                for form in ANSWER_FORMS.get(value, ())
                            )
                        ],
                    }
                ),
                "answerable_from_title": bool(in_title or equivalent),
            }
        )

    answerable = sum(1 for row in results if row.get("answerable_from_title"))
    return {
        "snapshot_id": snapshot,
        "qa_path": str(qa_path).replace("\\", "/"),
        "checked": len(results),
        "answerable_from_title": answerable,
        "needs_adjudication": len(results) - answerable,
        "rows": results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check headline-only QA representatives")
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--qa", default="eval/qa_set.jsonl")
    parser.add_argument("--out", default=None, help="write the JSON report here")
    args = parser.parse_args(argv)

    report = check(args.snapshot, Path(args.data_root), Path(args.qa))
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
