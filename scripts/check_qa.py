"""CLI: check the QA set against the quotas in `docs/EVAL_PROTOCOL.md` §2-§4.

Usage:
    python scripts/check_qa.py
    python scripts/check_qa.py --qa eval/qa_set.jsonl --calibration eval/abstain_calibration.jsonl
    python scripts/check_qa.py --no-check-hosts          # quotas only, skip the source-scope check
    python scripts/check_qa.py --check-urls              # + network: every gold url is 200 and in-window

Exit codes: 0 = may be frozen, 1 = violations found, 2 = file not found.

Three layers, cheapest first:

1. **Quotas and shape** (`qa.validate`) - offline, deterministic, always runs.
2. **Source scope** (`qa.validate_hosts`) - offline: every gold url must sit on a host the corpus
   actually crawls (`docs/SOURCES.md` §3). An out-of-scope url does not fail loudly later; it
   leaves both denominators as `gold_unavailable` (§4), so the loss is invisible in the number and
   visible only in the base it was computed on. This is the check that would have caught six such
   items before the freeze.
3. **Reachability and publication date** (`--check-urls`) - network, local only. A page that no
   longer answers 200, or one whose own published time is after `window_end`, is not evidence for
   a frozen corpus. Run it before the freeze, not on every commit.
"""

from __future__ import annotations

import argparse
import re
import sys
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

#: A page states its publication time in one of these; news sites use the first far more often.
_META_TAG = re.compile(r"<meta[^>]+>", re.IGNORECASE)
_META_KEYS = ("article:published_time", "datepublished", "pubdate", "publish-date")
_CONTENT = re.compile(r"""content\s*=\s*["']([^"']+)["']""", re.IGNORECASE)
_TIME_TAG = re.compile(r"""<time[^>]+datetime\s*=\s*["']([^"']+)["']""", re.IGNORECASE)
_USER_AGENT = "AcgnRagBot/0.1 (CS5481 course project; QA set evidence check)"


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


def published_stamp(html: str) -> str | None:
    """The page's own publication time as written, or None when it states none.

    Prefers the OpenGraph/article metadata every news site emits, then `<time datetime=...>`.
    Returned raw: the caller compares only the leading `YYYY-MM-DD`, because the feeds mix
    offsets and a timestamp comparison across offsets is a bug waiting to happen.
    """
    for tag in _META_TAG.findall(html):
        lowered = tag.lower()
        if any(key in lowered for key in _META_KEYS):
            content = _CONTENT.search(tag)
            if content:
                return content.group(1).strip()
    match = _TIME_TAG.search(html)
    return match.group(1).strip() if match else None


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


def check_urls(items: list[dict], window_end: str, *, timeout: float = 15.0) -> list[str]:
    """Network pass: every gold url answers 200 and was published inside the frozen window.

    A url whose page states no date is reported separately from one that states a date after
    `window_end`: the first needs a human look, the second is simply invalid as evidence.
    """
    problems: list[str] = []
    for item in items:
        where = str(item.get("qa_id") or "?")
        evidence = item.get("gold_evidence")
        urls = evidence.get("urls") if isinstance(evidence, dict) else None
        for url in urls if isinstance(urls, list) else []:
            if not isinstance(url, str):
                continue
            status, body = _fetch(url, timeout)
            if status != 200:
                problems.append(f"{where}: {url} answered {status or 'no response'}")
                continue
            stamp = published_stamp(body)
            if stamp is None:
                problems.append(f"{where}: {url} states no publication time - verify by hand")
            elif stamp[:10] > window_end:
                problems.append(
                    f"{where}: {url} was published {stamp[:10]}, after window_end {window_end} - "
                    "it cannot be evidence for this snapshot"
                )
    return problems


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
        help="network: fetch every gold url, require 200 and a publication date inside the window",
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

    if args.check_urls:
        window_end = str(items[0].get("window", ["", ""])[1]) if items else ""
        print(f"urls: fetching every gold url (window_end {window_end})")
        problems.extend(check_urls(items, window_end, timeout=args.timeout))
        if calibration is not None:
            problems.extend(check_urls(calibration, window_end, timeout=args.timeout))

    if problems:
        print(f"\n{len(problems)} problem(s):")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    print("\nOK: the set satisfies the protocol and may be frozen")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
