"""CLI: check the QA set against the quotas in `docs/EVAL_PROTOCOL.md` §2-§4.

Usage:
    python scripts/check_qa.py
    python scripts/check_qa.py --qa eval/qa_set.jsonl --calibration eval/abstain_calibration.jsonl
    python scripts/check_qa.py --no-check-hosts          # quotas only, skip the source-scope check
    python scripts/check_qa.py --check-urls              # + network: every gold url answers 200 and
                                                         #   its stated publication time sits inside
                                                         #   the item's recorded window

Exit codes: 0 = pass (urls listed for manual verification may still be present), 1 = violations
found, 2 = file not found.

Three layers, cheapest first:

1. **Quotas and shape** (`qa.validate`) - offline, deterministic, always runs.
2. **Source scope** (`qa.validate_hosts`) - offline: every gold url must sit on a host the corpus
   actually crawls (`docs/SOURCES.md` §3). An out-of-scope url does not fail loudly later; it
   leaves both denominators as `gold_unavailable` (§4), so the loss is invisible in the number and
   visible only in the base it was computed on. This is the check that would have caught six such
   items before the freeze.
3. **Reachability and publication date** (`--check-urls`) - network, local only. A url that no
   longer answers 200, or one whose stated publication time sits outside the item's recorded
   window under every plausible reading, is invalid as evidence (exit 1). A page that states no
   time, an unparseable one, or one whose readings straddle a window boundary is listed for
   manual verification (exit stays 0): the list is the deliverable, not a pass. Run it before the
   freeze, not on every commit.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from acgn_rag.core.eval.qa import (
    W1_HOSTS,
    host_of,
    load_jsonl,
    summary,
    validate,
    validate_hosts,
)

#: A page states its publication time in one of these; the order below is the preference order.
_META_TAG = re.compile(r"<meta[^>]+>", re.IGNORECASE)
_META_KEYS = ("article:published_time", "datepublished", "pubdate", "publish-date")
_CONTENT = re.compile(r"""content\s*=\s*["']([^"']+)["']""", re.IGNORECASE)
_TIME_TAG = re.compile(r"""<time[^>]+datetime\s*=\s*["']([^"']+)["']""", re.IGNORECASE)
_ARTICLE_TAG = re.compile(r"<article\b.*?</article>", re.IGNORECASE | re.DOTALL)
_H1_TAG = re.compile(r"<h1\b", re.IGNORECASE)
_JSON_LD = re.compile(
    r"""<script[^>]+type\s*=\s*["']application/ld\+json["'][^>]*>(.*?)</script>""",
    re.IGNORECASE | re.DOTALL,
)
_USER_AGENT = "AcgnRagBot/0.1 (CS5481 course project; QA set evidence check)"

#: Naive stamps (no UTC offset on the page) are read under both conventions before a verdict:
#: several sources sit east of UTC and do not say so, and a naive stamp near a window boundary is
#: ambiguous rather than wrong. JST covers PR TIMES, whose help pages document Japan time.
_NAIVE_READINGS = (UTC, timezone(timedelta(hours=9)))


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


def _host_summary(items: list[dict]) -> str:
    """Distinct hosts with counts, so a scope decision is visible before it is enforced."""
    counts: dict[str, int] = {}
    for item in items:
        evidence = item.get("gold_evidence")
        urls = evidence.get("urls") if isinstance(evidence, dict) else None
        for url in urls if isinstance(urls, list) else []:
            host = host_of(url) if isinstance(url, str) else ""
            if host:
                counts[host] = counts.get(host, 0) + 1
    return "  ".join(f"{host}={n}" for host, n in sorted(counts.items()))


def _count_urls(*item_lists: list[dict] | None) -> int:
    """How many gold urls the url pass covered, so the OK line states its own base."""
    total = 0
    for item_list in item_lists:
        for item in item_list or []:
            evidence = item.get("gold_evidence")
            urls = evidence.get("urls") if isinstance(evidence, dict) else None
            if isinstance(urls, list):
                total += sum(isinstance(url, str) for url in urls)
    return total


def _jsonld_dates(html: str) -> list[str]:
    """`datePublished` values from the page's JSON-LD blocks, @graph lists included.

    PR TIMES pages carry no <meta> or <time> stamp - their only machine-readable publication time
    is the NewsArticle JSON-LD - so this pass sits between the meta pass and the <time> fallback.
    """
    dates: list[str] = []
    for block in _JSON_LD.findall(html):
        try:
            data = json.loads(block)
        except json.JSONDecodeError:
            continue
        stack = [data]
        while stack:
            node = stack.pop()
            if isinstance(node, dict):
                value = node.get("datePublished")
                if isinstance(value, str):
                    dates.append(value.strip())
                stack.extend(child for child in node.values() if isinstance(child, (dict, list)))
            elif isinstance(node, list):
                stack.extend(child for child in node if isinstance(child, (dict, list)))
    return dates


def published_stamp(html: str) -> str | None:
    """The page's own publication time as written, or None when it states none.

    Preference order: OpenGraph/article metadata, then JSON-LD `datePublished` (the only stamp
    PR TIMES pages carry), then a `<time datetime>` inside `<article>`, then the first `<time>`
    after the main `<h1>` - on news sites the first `<time>` in the document is a sidebar or
    render timestamp, and the article's byline sits below the headline (every ANN article in the
    set is that shape). Returned raw: the caller classifies it with `classify_stamp`, because the
    feeds mix offsets and a naive boundary stamp is ambiguous rather than wrong.
    """
    for tag in _META_TAG.findall(html):
        lowered = tag.lower()
        if any(key in lowered for key in _META_KEYS):
            content = _CONTENT.search(tag)
            if content:
                return content.group(1).strip()
    dates = _jsonld_dates(html)
    if dates:
        return dates[0]
    article = _ARTICLE_TAG.search(html)
    if article:
        match = _TIME_TAG.search(article.group(0))
        if match:
            return match.group(1).strip()
    heading = _H1_TAG.search(html)
    if heading:
        match = _TIME_TAG.search(html[heading.end() :])
        if match:
            return match.group(1).strip()
    match = _TIME_TAG.search(html)
    return match.group(1).strip() if match else None


def classify_stamp(stamp: str | None, window: list[str]) -> tuple[str, str]:
    """Classify one stated publication time against the item's recorded [start, end] window.

    Returns (verdict, detail), the verdict being:
    - "ok"      inside the window under every plausible reading of the stamp;
    - "manual"  undecidable from the stamp alone: absent, unparseable, a date-only stamp on a
                window boundary, or a naive timestamp whose UTC and JST readings straddle one;
    - "invalid" outside the window under every plausible reading - the page cannot be evidence
                for that snapshot.
    """
    if stamp is None or not stamp.strip():
        return "manual", "states no publication time - verify by hand"
    text = stamp.strip()
    try:
        window_start = datetime.fromisoformat(f"{window[0]}T00:00:00+00:00")
        # the end date is inclusive: the window runs until the end of that UTC day
        window_end = datetime.fromisoformat(f"{window[1]}T00:00:00+00:00") + timedelta(days=1)
    except (ValueError, TypeError, IndexError):
        return "manual", f"window {window} is not two ISO dates - verify by hand"

    if len(text) == 10 and text[4] == "-" and text[7] == "-":
        # a date-only stamp names no hour: a whole day strictly between the boundary days is
        # inside no matter which hour it was, a boundary day is not decidable
        if window[0] < text < window[1]:
            return "ok", f"date-only stamp {text} falls strictly inside the window"
        if text < window[0] or text > window[1]:
            return "invalid", (
                f"date-only stamp {text} is outside the window [{window[0]}, {window[1]}]"
            )
        return "manual", f"date-only stamp {text} sits on the window boundary - verify by hand"

    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return "manual", f"unparseable publication time {text!r} - verify by hand"
    if parsed.tzinfo is not None:
        if window_start <= parsed < window_end:
            return "ok", f"published {text}"
        if parsed >= window_end:
            return "invalid", (
                f"published {text}, after window end {window[1]} - it cannot be evidence for "
                "this snapshot"
            )
        return "invalid", (
            f"published {text}, before window start {window[0]} - it cannot be evidence for "
            "this snapshot"
        )
    instants = [parsed.replace(tzinfo=zone) for zone in _NAIVE_READINGS]
    if all(window_start <= instant < window_end for instant in instants):
        return "ok", f"naive stamp {text}: inside the window under both UTC and JST readings"
    if all(instant >= window_end for instant in instants):
        return "invalid", (
            f"published {text}, after window end {window[1]} under both UTC and JST readings - "
            "it cannot be evidence for this snapshot"
        )
    if all(instant < window_start for instant in instants):
        return "invalid", (
            f"published {text}, before window start {window[0]} under both UTC and JST readings - "
            "it cannot be evidence for this snapshot"
        )
    return "manual", (
        f"published {text}: UTC and JST readings straddle the window boundary - verify by hand"
    )


def _fetch(url: str, timeout: float) -> tuple[int, str]:
    """GET one url with the project UA; returns (status, body). Never raises for HTTP status."""
    import urllib.error
    import urllib.request

    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return int(response.status), response.read(2_000_000).decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return int(exc.code), ""
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return 0, f"{exc}"


def check_urls(items: list[dict], *, timeout: float = 15.0) -> tuple[list[str], list[str]]:
    """Network pass: every gold url answers 200 and its stated publication time sits inside the
    item's own recorded window (no more borrowing one item's window_end for a whole file).

    Returns (invalid, manual): `invalid` fails the run; `manual` is printed for a human - a page
    that states no time is not evidence yet, but it is not evidence against the item either.
    """
    invalid: list[str] = []
    manual: list[str] = []
    for item in items:
        where = str(item.get("qa_id") or "?")
        window = item.get("window")
        window = window if isinstance(window, list) and len(window) == 2 else ["", ""]
        evidence = item.get("gold_evidence")
        urls = evidence.get("urls") if isinstance(evidence, dict) else None
        for url in urls if isinstance(urls, list) else []:
            if not isinstance(url, str):
                continue
            status, body = _fetch(url, timeout)
            if status != 200:
                invalid.append(f"{where}: {url} answered {status or 'no response'}")
                continue
            verdict, detail = classify_stamp(published_stamp(body), window)
            line = f"{where}: {url} {detail}"
            if verdict == "invalid":
                invalid.append(line)
            elif verdict == "manual":
                manual.append(line)
    return invalid, manual


def main() -> int:
    parser = argparse.ArgumentParser(description="Check the QA set against the evaluation protocol")
    parser.add_argument("--qa", default="eval/qa_set.jsonl", help="QA set (JSONL)")
    parser.add_argument(
        "--calibration",
        default="eval/abstain_calibration.jsonl",
        help="abstention calibration set (JSONL); skipped when absent",
    )
    parser.add_argument(
        "--check-hosts",
        dest="check_hosts",
        action="store_true",
        default=True,
        help="require every gold url host to be a W1 source (default: on)",
    )
    parser.add_argument(
        "--no-check-hosts",
        dest="check_hosts",
        action="store_false",
        help="skip the source-scope check (quotas only)",
    )
    parser.add_argument(
        "--allow-host",
        action="append",
        default=[],
        metavar="DOMAIN",
        help="extra base domain to accept, for a source promoted into W1 by A (repeatable)",
    )
    parser.add_argument(
        "--check-urls",
        action="store_true",
        help="network: fetch every gold url, require 200 and a stated publication time inside "
        "the item's window",
    )
    parser.add_argument(
        "--timeout", type=float, default=15.0, help="per-request timeout for --check-urls"
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

    if args.check_hosts:
        allowed = (*W1_HOSTS, *args.allow_host)
        print(f"hosts: {_host_summary(items)}")
        problems.extend(validate_hosts(items, allowed=allowed, label=qa_path.name))
        if calibration is not None:
            problems.extend(validate_hosts(calibration, allowed=allowed, label=cal_path.name))

    manual: list[str] = []
    if args.check_urls:
        print("urls: fetching every gold url against its item's recorded window")
        invalid, manual = check_urls(items, timeout=args.timeout)
        if calibration is not None:
            cal_invalid, cal_manual = check_urls(calibration, timeout=args.timeout)
            invalid.extend(cal_invalid)
            manual.extend(cal_manual)
        for line in manual:
            print(f"  manual: {line}")
        problems.extend(invalid)

    if problems:
        print(f"\n{len(problems)} problem(s):")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    if args.check_urls:
        if manual:
            print(
                f"\nOK: quotas and source scope pass; {len(manual)} url(s) listed above could not "
                "be dated this pass and still need manual verification before the freeze."
            )
        else:
            print(
                f"\nOK: quotas, source scope and every stated publication time pass "
                f"({_count_urls(items, calibration)} urls). The frozen-set evidence pass "
                "(EVAL_PROTOCOL §1) still applies."
            )
    else:
        print(
            "\nOK: the set satisfies the quota and source-scope rules; publication dates are NOT "
            "checked - run --check-urls before the freeze."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
