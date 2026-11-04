"""QA set rules (docs/EVAL_PROTOCOL.md §3): the quotas are only real if something enforces them."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import pytest

from acgn_rag.core.eval.qa import (
    W1_HOSTS,
    host_in_scope,
    host_of,
    load_jsonl,
    summary,
    validate,
    validate_hosts,
)


def _item(
    qa_id: str,
    qa_type: str,
    *,
    answerable: bool = True,
    overlap: bool = False,
    **overrides: Any,
) -> dict[str, Any]:
    item: dict[str, Any] = {
        "qa_id": qa_id,
        "type": qa_type,
        "lang": "en",
        "answerable": answerable,
        "question": f"question {qa_id}",
        "gold_answer": "2026-10-05",
        "gold_evidence": {
            "urls": ["https://www.toei-anim.co.jp/news/1"],
            "publisher": "Toei Animation",
            "spans": ["will begin airing on 2026-10-05"],
        },
        "key_entities": ["2026-10-05"],
        "window": ["2023-10-28", "2026-10-28"],
        "annotator": "D",
        "verified_by": "F",
        "overlap": overlap,
    }
    item.update(overrides)
    return item


def _valid_set() -> list[dict[str, Any]]:
    """11 A + 9 B + 7 C + 8 D = 35 items; A is 31% of the total; 5 unanswerable; 10 overlap."""
    items: list[dict[str, Any]] = []
    suffix = 1
    for qa_type, count in (("A", 11), ("B", 9), ("C", 7), ("D", 8)):
        for _ in range(count):
            id_prefix = qa_type
            items.append(
                _item(
                    f"{id_prefix}-{suffix:03d}",
                    qa_type,
                    answerable=not (qa_type == "D" and suffix > 30),
                )
            )
            suffix += 1
    for index, item in enumerate(items[:10]):
        items[index] = {**item, "overlap": True}
    for index in range(8):  # the language quota needs 8 zh items in a valid set
        items[index] = {**items[index], "lang": "zh"}
    return items


def _calibration(size: tuple[int, int] = (20, 10)) -> list[dict[str, Any]]:
    answerable, unanswerable = size
    items = [_item(f"CAL-{i:03d}", "A") for i in range(answerable)] + [
        _item(f"CAL-{i:03d}", "D", answerable=False) for i in range(unanswerable)
    ]
    for index, item in enumerate(items):
        item["question"] = f"calibration question {index}"
        item["qa_id"] = f"CAL-{index:03d}"
    return items


def test_a_valid_set_passes() -> None:
    items = _valid_set()
    assert validate(items, calibration=_calibration()) == []


def test_summary_counts_what_the_report_needs() -> None:
    counts = summary(_valid_set())
    assert counts["total"] == 35
    assert counts["by_type"] == {"A": 11, "B": 9, "C": 7, "D": 8}
    assert counts["by_lang"] == {"en": 27, "zh": 8}
    assert counts["unanswerable"] == 5
    assert counts["overlap"] == 10


def test_total_below_thirty_fails() -> None:
    problems = validate(_valid_set()[:20])
    assert any("total 20 < 30" in p for p in problems), problems


def test_type_a_above_one_third_fails() -> None:
    items = _valid_set()
    for index in range(4):  # 11 A -> 15 A of 35 = 43%
        items.append(_item(f"A-{900 + index:03d}", "A"))
    problems = validate(items)
    assert any("one third" in p for p in problems), problems


def test_unanswerable_minimum_is_enforced() -> None:
    items = [_item(i["qa_id"], i["type"], overlap=i["overlap"]) for i in _valid_set()]
    problems = validate(items)
    assert any("unanswerable" in p for p in problems), problems


def test_overlap_must_be_exactly_ten() -> None:
    items = _valid_set()
    items[0] = {**items[0], "overlap": False}
    problems = validate(items)
    assert any("need exactly 10" in p for p in problems), problems


def test_answerable_item_without_url_fails() -> None:
    items = _valid_set()
    items[0] = {**items[0], "gold_evidence": {"urls": [], "publisher": "x", "spans": []}}
    problems = validate(items)
    assert any("at least one gold_evidence url" in p for p in problems), problems


def test_url_must_be_https() -> None:
    items = _valid_set()
    items[0] = {**items[0], "gold_evidence": {"urls": ["http://example.com/a"]}}
    problems = validate(items)
    assert any("must be https" in p for p in problems), problems


def test_unanswerable_must_be_type_d() -> None:
    items = _valid_set()
    items[0] = {**items[0], "answerable": False}
    problems = validate(items)
    assert any("must be type D" in p for p in problems), problems


def test_missing_lang_fails() -> None:
    items = _valid_set()
    del items[0]["lang"]
    problems = validate(items)
    assert any("'lang'" in p for p in problems), problems


def test_lang_outside_en_zh_fails() -> None:
    items = _valid_set()
    items[0] = {**items[0], "lang": "jp"}
    problems = validate(items)
    assert any("lang must be one of" in p for p in problems), problems


def test_language_quota_is_enforced() -> None:
    # the same set with every zh mark dropped: the quota fails, not just the wording
    items = [_item(i["qa_id"], i["type"], overlap=i["overlap"]) for i in _valid_set()]
    problems = validate(items)
    assert any("lang='zh'" in p for p in problems), problems


def test_annotator_cannot_verify_themselves() -> None:
    items = _valid_set()
    items[0] = {**items[0], "verified_by": "D"}
    problems = validate(items)
    assert any("self-review" in p for p in problems), problems


def test_duplicate_ids_fail() -> None:
    items = _valid_set()
    items[1] = {**items[1], "qa_id": items[0]["qa_id"]}
    problems = validate(items)
    assert any("duplicate qa_id" in p for p in problems), problems


def test_calibration_must_be_disjoint_or_the_number_is_optimistic() -> None:
    items = _valid_set()
    calibration = _calibration()
    calibration[0] = {**calibration[0], "question": items[0]["question"]}
    problems = validate(items, calibration=calibration)
    assert any("shares 1 question" in p for p in problems), problems


def test_calibration_sizes_are_checked() -> None:
    problems = validate(_valid_set(), calibration=_calibration((20, 5)))
    assert any("need 10" in p for p in problems), problems


def test_load_jsonl_rejects_malformed_json(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps(_valid_set()[0]) + "\n{not json}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON"):
        load_jsonl(path)


def test_cli_reports_and_exits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    qa_path = tmp_path / "qa_set.jsonl"
    qa_path.write_text(
        "\n".join(json.dumps(item) for item in _valid_set()) + "\n", encoding="utf-8"
    )
    monkeypatch.setattr(
        sys, "argv", ["check_qa.py", "--qa", str(qa_path), "--calibration", str(tmp_path / "none")]
    )
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from check_qa import main

    assert main() == 0


def test_gold_url_outside_the_w1_sources_is_rejected() -> None:
    """A url on a host the corpus never crawls becomes gold_unavailable, not a 0 score."""
    items = _valid_set()
    items[0]["gold_evidence"] = {
        "urls": ["https://p-ch.jp/news/26745/"],
        "publisher": "Bushiroad",
        "spans": ["x"],
    }
    problems = validate_hosts(items)
    assert any("p-ch.jp" in problem for problem in problems), problems


def test_subdomain_of_a_w1_host_counts_as_in_scope() -> None:
    """The corpus takes press releases from corp.* while SOURCES.md §3 lists the www. endpoint."""
    assert host_in_scope("corp.toei-anim.co.jp")
    assert host_in_scope("graphql.anilist.co")
    assert host_in_scope("toei-anim.co.jp")
    assert not host_in_scope("nottoei-anim.co.jp")
    assert not host_in_scope("toei-anim.co.jp.evil.example")


def test_every_source_listed_in_sources_md_is_in_the_allowlist() -> None:
    """SOURCES.md §3 and W1_HOSTS must move together, or the checker silently lies.

    One direction only: the allowlist may carry hosts §3 mentions without a url (BONES and
    P.A.WORKS share a row with MAPPA), but a url §3 lists must never be missing from it.
    """
    doc = Path(__file__).resolve().parents[1] / "docs" / "SOURCES.md"
    section = doc.read_text(encoding="utf-8").split("## 3. W1 sources")[1].split("## 4.")[0]
    urls = re.findall(r"https?://[^\s`|)]+", section)
    assert urls, "SOURCES.md §3 parsed as empty - the section heading moved"
    missing = sorted({host_of(url) for url in urls if not host_in_scope(host_of(url))})
    assert not missing, f"SOURCES.md §3 lists hosts missing from W1_HOSTS: {missing}"
    assert "toei-anim.co.jp" in W1_HOSTS


def test_published_stamp_reads_metadata_then_time_tag() -> None:
    """The in-window check needs the page's own date; both markup shapes occur in the sources."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from check_qa import published_stamp

    assert (
        published_stamp(
            '<meta property="article:published_time" content="2026-10-29T13:30:00+09:00">'
        )
        == "2026-10-29T13:30:00+09:00"
    )
    assert published_stamp('<meta content="2026-10-28" name="datePublished">') == "2026-10-28"
    assert published_stamp('<time datetime="2026-10-28">28 Oct</time>') == "2026-10-28"
    assert published_stamp("<html><body>no date here</body></html>") is None


def test_published_stamp_reads_jsonld_before_the_first_time_tag() -> None:
    """PR TIMES pages state their date only in JSON-LD; ANN articles carry sidebar <time> decoys."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from check_qa import published_stamp

    jsonld = (
        '<html><head><script type="application/ld+json">'
        '{"@type":"NewsArticle","datePublished":"2026-01-22 17:00:01"}'
        "</script></head><body></body></html>"
    )
    assert published_stamp(jsonld) == "2026-01-22 17:00:01"

    graph = (
        '<script type="application/ld+json">'
        '{"@graph":[{"@type":"WebPage"},'
        '{"@type":"NewsArticle","datePublished":"2025-04-26 12:00:02"}]}'
        "</script>"
    )
    assert published_stamp(graph) == "2025-04-26 12:00:02"

    article_decoy = (
        '<html><body><nav><time datetime="2026-11-01T09:00:00">recent</time></nav>'
        '<article><time datetime="2025-03-21T11:30:00">posted</time></article></body></html>'
    )
    assert published_stamp(article_decoy) == "2025-03-21T11:30:00"


def test_published_stamp_takes_the_byline_after_the_headline() -> None:
    """ANN articles have no <article>/JSON-LD and sidebar <time> decoys before the headline."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from check_qa import published_stamp

    ann_shape = (
        '<html><head><meta property="og:type" content="article"></head><body>'
        '<div class="rail"><time datetime="2026-11-01T14:15:00+08:00">recent</time></div>'
        '<h1 id="page_header">A title</h1>'
        '<small>posted on <time datetime="2025-03-21T13:01:34+08:00">2025-03-21</time></small>'
        "<p>body</p></body></html>"
    )
    assert published_stamp(ann_shape) == "2025-03-21T13:01:34+08:00"
    # without any headline the legacy first-<time> fallback still applies
    assert published_stamp('<div><time datetime="2026-01-01">x</time></div>') == "2026-01-01"


def test_classify_stamp_offsets_decide_boundary_cases() -> None:
    """F's offline counterexamples: comparing stamp[:10] got both of these wrong."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from check_qa import classify_stamp

    window = ["2023-10-28", "2026-10-28"]
    # 2026-10-28T23:30-02:00 is 2026-10-29T01:30Z: the string compare passed it, the instant does not
    verdict, detail = classify_stamp("2026-10-28T23:30:00-02:00", window)
    assert verdict == "invalid" and "after window end" in detail
    # 2026-10-29T00:30+09:00 is 2026-10-28T15:30Z: the string compare failed it, the instant does not
    verdict, _ = classify_stamp("2026-10-29T00:30:00+09:00", window)
    assert verdict == "ok"
    # before window_start must fail too, not only after the end
    verdict, detail = classify_stamp("2010-01-01T00:00:00Z", window)
    assert verdict == "invalid" and "before window start" in detail


def test_classify_stamp_naive_stamps_pass_only_when_readings_agree() -> None:
    """A naive stamp near a boundary is ambiguous, not wrong: it must land in the manual bucket."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from check_qa import classify_stamp

    window = ["2023-10-28", "2026-10-28"]
    verdict, detail = classify_stamp("2026-10-23 15:00:02", window)
    assert verdict == "ok" and "both UTC and JST" in detail
    verdict, detail = classify_stamp("2026-10-29 13:39:00", window)
    assert verdict == "invalid" and "after window end" in detail
    verdict, _ = classify_stamp("2026-10-29 00:30:00", window)  # UTC says out, JST says in
    assert verdict == "manual"


def test_classify_stamp_date_only_and_garbage() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from check_qa import classify_stamp

    window = ["2023-10-28", "2026-10-28"]
    assert classify_stamp("2025-06-15", window)[0] == "ok"
    assert classify_stamp("2026-10-28", window)[0] == "manual"  # boundary day, hour unknown
    assert classify_stamp("2027-01-01", window)[0] == "invalid"
    assert classify_stamp(None, window)[0] == "manual"
    assert classify_stamp("not a date", window)[0] == "manual"


def _run_cli_with_fetch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stamp_html: str) -> int:
    """main() under --check-urls with the network stubbed: the stamp classification decides."""
    qa_path = tmp_path / "qa_set.jsonl"
    qa_path.write_text(
        "\n".join(json.dumps(item) for item in _valid_set()) + "\n", encoding="utf-8"
    )
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import check_qa

    monkeypatch.setattr(check_qa, "_fetch", lambda url, timeout: (200, stamp_html))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check_qa.py",
            "--qa",
            str(qa_path),
            "--calibration",
            str(tmp_path / "none"),
            "--check-urls",
        ],
    )
    return check_qa.main()


def test_check_urls_passes_when_stated_times_sit_inside_the_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    page = '<meta property="article:published_time" content="2026-06-01T00:00:00+09:00">'
    assert _run_cli_with_fetch(tmp_path, monkeypatch, page) == 0


def test_check_urls_fails_when_a_stated_time_sits_after_window_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    page = '<meta property="article:published_time" content="2026-10-29T13:30:00+09:00">'
    assert _run_cli_with_fetch(tmp_path, monkeypatch, page) == 1
    assert "after window end" in capsys.readouterr().out


def test_check_urls_lists_undated_pages_instead_of_failing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert _run_cli_with_fetch(tmp_path, monkeypatch, "<html><body>no date</body></html>") == 0
