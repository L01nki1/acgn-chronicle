"""Render the G1b dedup label sheet as a self-contained labelling page.

Owner: F (WP2) | Spec: docs/RUNBOOK.md §3.3, scripts/calibrate_dedup.py

`calibrate_dedup.py` writes `label_sheet.csv`; this script turns that CSV into one HTML file
with the rows embedded as JSON, so the page works from `file://` with no server and no network.
The page is a *labelling tool*, not a scoring tool: it exports the same CSV columns back, and
`calibrate_dedup.py --score` stays the only thing that computes precision.

    python scripts/render_label_sheet.py --csv eval/dedup_g1b/label_sheet.csv \
        --out eval/dedup_g1b/label_sheet.html

The output must stay out of git when the sheet contains full article text; the sheet only
carries titles and URLs, so it is safe to commit alongside the CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

TEMPLATE = Path(__file__).resolve().parent / "label_sheet_template.html"


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def render(rows: list[dict[str, str]], template: str, *, source: str) -> str:
    payload = json.dumps(rows, ensure_ascii=False, separators=(",", ":"))
    # `</script>` inside a title would end the embedding block early; escaping the slash keeps
    # the JSON valid and the page intact.
    payload = payload.replace("</", "<\\/")
    # The template writes the source inside a JS string literal (`"/*__SOURCE__*/"`), so the
    # replacement must stay a plain string, not a second quoted literal.
    return template.replace("/*__ROWS__*/", payload).replace("/*__SOURCE__*/", source)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render the dedup label sheet as HTML")
    parser.add_argument("--csv", default="eval/dedup_g1b/label_sheet.csv")
    parser.add_argument("--out", default="eval/dedup_g1b/label_sheet.html")
    parser.add_argument("--template", default=str(TEMPLATE))
    args = parser.parse_args(argv)

    csv_path = Path(args.csv)
    if not csv_path.is_file():
        print(f"NOT FOUND: {csv_path}", file=sys.stderr)
        return 2
    template_path = Path(args.template)
    if not template_path.is_file():
        print(f"NOT FOUND: {template_path}", file=sys.stderr)
        return 2

    rows = load_rows(csv_path)
    if not rows:
        print(f"{csv_path}: no rows", file=sys.stderr)
        return 2
    html = render(
        rows,
        template_path.read_text(encoding="utf-8"),
        source=str(csv_path).replace("\\", "/"),
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    labelled = sum(1 for row in rows if (row.get("decision") or "").strip())
    print(f"wrote {out} ({len(rows)} rows, {labelled} already labelled)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
