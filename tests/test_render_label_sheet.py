"""Tests for the HTML label-sheet renderer: rows embedded, placeholders gone, page self-contained."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import render_label_sheet  # noqa: E402

CSV_TEXT = (
    "threshold,pair_id,cosine,cluster_id,same_language,event_type_a,event_type_b,"
    "media_ids_a,media_ids_b,title_a,title_b,url_a,url_b,decision,note\n"
    '0.86,0.86-01,1.0000,cluster_abc,yes,,,,,"2026年 年賀状","2026年 年賀状",'
    "https://a.example/1,https://a.example/2,,\n"
    '0.86,0.86-02,0.5000,cluster_def,no,,,1,2,"A </script> title","B title",'
    "https://b.example/1,https://b.example/2,duplicate,checked\n"
)


def _csv(tmp_path: Path) -> Path:
    path = tmp_path / "label_sheet.csv"
    path.write_text(CSV_TEXT, encoding="utf-8")
    return path


def test_render_embeds_every_row_and_leaves_no_placeholder(tmp_path: Path) -> None:
    out = tmp_path / "label_sheet.html"
    assert render_label_sheet.main(["--csv", str(_csv(tmp_path)), "--out", str(out)]) == 0
    html = out.read_text(encoding="utf-8")

    assert "/*__ROWS__*/" not in html
    assert "/*__SOURCE__*/" not in html

    match = re.search(r'<script id="sheet-data" type="application/json">(.*?)</script>', html, re.S)
    assert match, "embedded JSON block missing"
    rows = json.loads(match.group(1).replace("<\\/", "</"))
    assert [row["pair_id"] for row in rows] == ["0.86-01", "0.86-02"]
    assert rows[1]["title_a"] == "A </script> title"
    assert rows[1]["decision"] == "duplicate"


def test_render_keeps_the_page_self_contained(tmp_path: Path) -> None:
    out = tmp_path / "label_sheet.html"
    render_label_sheet.main(["--csv", str(_csv(tmp_path)), "--out", str(out)])
    html = out.read_text(encoding="utf-8")

    # No network: no external stylesheet, script or font reference.
    assert "<link" not in html
    assert not re.search(r"<script[^>]+src=", html)
    assert "http://" not in html.replace("http://127.0.0.1", "")
    # The judgement guide and the export controls are part of the page.
    for marker in ("判断标准", "导出 CSV", "导入 CSV", "duplicate", "not_duplicate", "skip"):
        assert marker in html, marker


def test_render_rejects_a_missing_csv(tmp_path: Path, capsys) -> None:
    code = render_label_sheet.main(
        ["--csv", str(tmp_path / "nope.csv"), "--out", str(tmp_path / "x.html")]
    )
    assert code == 2
    assert "NOT FOUND" in capsys.readouterr().err


def test_render_rejects_an_empty_csv(tmp_path: Path, capsys) -> None:
    empty = tmp_path / "empty.csv"
    empty.write_text("threshold,pair_id,decision\n", encoding="utf-8")
    code = render_label_sheet.main(["--csv", str(empty), "--out", str(tmp_path / "x.html")])
    assert code == 2
    assert "no rows" in capsys.readouterr().err
