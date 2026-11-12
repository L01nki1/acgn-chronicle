"""Structured metadata API adapter (`StructuredApiAdapter`) for AniList GraphQL,
Bangumi v0 API, Bilibili PGC timeline, and Syosetu `novelapi` (metadata only, C2).

Owner: A (WP1) | Spec: docs/SOURCES.md §3 & §7, docs/COMPLIANCE.md C1 & C2, Issue #9
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import Any, Literal

from acgn_rag.core.ingest.ann_archive import (
    BackfillNotSupportedError,
    EmptyBackfillWindowError,
    SelectorBrokenError,
    validate_backfill_window,
)
from acgn_rag.core.ingest.http import PoliteHttpClient
from acgn_rag.core.ingest.normalize import (
    RawIngestItem,
    clean_title,
    normalize_text,
    normalize_url,
    parse_utc_timestamp,
    strip_html_boilerplate,
    utc_iso_at_or_after,
)
from acgn_rag.core.ports import SourceType

ANILIST_GRAPHQL_URL = "https://graphql.anilist.co"
BANGUMI_SUBJECTS_URL = "https://api.bgm.tv/v0/subjects"
BILIBILI_TIMELINE_URL = "https://api.bilibili.com/pgc/web/timeline?types=1&before=7&after=7"
SYOSETU_NOVELAPI_URL = "https://api.syosetu.com/novelapi/api/?out=json&lim=500&order=weekly"

ANILIST_PAGE_QUERY = """
query ($page: Int, $perPage: Int, $startGreater: FuzzyDateInt, $startLesser: FuzzyDateInt) {
  Page(page: $page, perPage: $perPage) {
    pageInfo {
      hasNextPage
    }
    media(startDate_greater: $startGreater, startDate_lesser: $startLesser, sort: [POPULARITY_DESC, ID]) {
      id
      type
      format
      status
      episodes
      chapters
      volumes
      genres
      synonyms
      description(asHtml: false)
      title {
        romaji
        english
        native
      }
      startDate {
        year
        month
        day
      }
      studios(isMain: true) {
        nodes {
          name
        }
      }
    }
  }
}
""".strip()


def _fuzzy_date_int(d: date) -> int:
    return d.year * 10000 + d.month * 100 + d.day


@dataclass
class StructuredApiAdapter:
    """Two-mode structured metadata API adapter (`anilist`, `bangumi`, `bilibili`, `syosetu`)."""

    source_key: str
    provider: Literal["anilist", "bangumi", "bilibili", "syosetu"]
    endpoint_url: str
    publisher: str
    source_type: SourceType = "metadata"
    lang: str = "en"
    source_tz: str | None = None
    max_pages: int = 3
    per_page: int = 50
    http: PoliteHttpClient = field(default_factory=PoliteHttpClient)
    cache_responses: bool = True
    _cached_items: list[RawIngestItem] | None = field(init=False, default=None, repr=False)
    last_skipped_entries: int = field(init=False, default=0, repr=False)
    last_supplementary_errors: int = field(init=False, default=0, repr=False)
    last_not_modified: bool = field(init=False, default=False, repr=False)

    def _fetch_anilist(
        self,
        *,
        start_d: date | None,
        end_d: date | None,
        max_pages: int,
    ) -> list[RawIngestItem]:
        start_int = _fuzzy_date_int(start_d) - 1 if start_d else 20050100
        end_int = _fuzzy_date_int(end_d) + 1 if end_d else 20991232

        self.last_skipped_entries = 0
        items: list[RawIngestItem] = []
        seen_urls: set[str] = set()
        for page in range(1, max_pages + 1):
            variables = {
                "page": page,
                "perPage": self.per_page,
                "startGreater": start_int,
                "startLesser": end_int,
            }
            res = self.http.post(
                self.endpoint_url,
                json_body={"query": ANILIST_PAGE_QUERY, "variables": variables},
                headers={"Accept": "application/json"},
            )
            try:
                payload = json.loads(res.text)
            except json.JSONDecodeError as exc:
                raise SelectorBrokenError(
                    f"[{self.source_key}] Invalid JSON from AniList GraphQL: {exc}"
                ) from exc

            page_obj = (
                (payload.get("data") or {}).get("Page") if isinstance(payload, dict) else None
            )
            if not isinstance(page_obj, dict):
                raise SelectorBrokenError(
                    f"[{self.source_key}] Missing data.Page in AniList response from {self.endpoint_url!r}"
                )
            media_list = page_obj.get("media")
            if not isinstance(media_list, list):
                raise SelectorBrokenError(
                    f"[{self.source_key}] Missing media list in AniList response"
                )
            if page == 1 and not media_list:
                raise SelectorBrokenError(
                    f"[{self.source_key}] AniList returned 0 media items on page 1"
                )

            for media in media_list:
                if not isinstance(media, dict) or not media.get("id"):
                    self.last_skipped_entries += 1
                    continue
                media_id = str(media["id"])
                mtype = str(media.get("type") or "ANIME").upper()
                mformat = str(media.get("format") or mtype).upper()
                title_obj = media.get("title") or {}
                romaji = str(title_obj.get("romaji") or "").strip()
                english = str(title_obj.get("english") or "").strip()
                native = str(title_obj.get("native") or "").strip()
                primary_title = english or romaji or native
                if not primary_title:
                    self.last_skipped_entries += 1
                    continue

                sd = media.get("startDate") or {}
                year = sd.get("year")
                month = sd.get("month")
                day = sd.get("day")
                if (
                    not isinstance(year, int)
                    or not isinstance(month, int)
                    or not isinstance(day, int)
                    or year < 1970
                ):
                    self.last_skipped_entries += 1
                    continue
                try:
                    pub_date = date(year, month, day)
                except ValueError:
                    self.last_skipped_entries += 1
                    continue

                studios_nodes = ((media.get("studios") or {}).get("nodes")) or []
                studios = [
                    str(n.get("name"))
                    for n in studios_nodes
                    if isinstance(n, dict) and n.get("name")
                ]
                genres = [str(g) for g in (media.get("genres") or []) if g]
                aliases = [
                    a
                    for a in (romaji, english, native, *(media.get("synonyms") or []))
                    if a and isinstance(a, str)
                ]
                unique_aliases = list(dict.fromkeys(aliases))
                desc_raw = str(media.get("description") or "").strip()
                desc_clean = strip_html_boilerplate(desc_raw) if desc_raw else ""

                type_slug = "manga" if mtype == "MANGA" else "anime"
                canonical_url = normalize_url(f"https://anilist.co/{type_slug}/{media_id}")
                if canonical_url in seen_urls:
                    continue
                seen_urls.add(canonical_url)

                body_lines = [
                    f"Title: {primary_title}",
                    f"Aliases: {', '.join(unique_aliases[:6])}",
                    f"Format: {mformat} ({mtype})",
                    f"Status: {media.get('status') or 'UNKNOWN'}",
                    f"Release Date: {pub_date.isoformat()}",
                ]
                if studios:
                    body_lines.append(f"Studios: {', '.join(studios)}")
                if genres:
                    body_lines.append(f"Genres: {', '.join(genres)}")
                if desc_clean:
                    body_lines.append(f"Synopsis: {desc_clean}")

                items.append(
                    RawIngestItem(
                        url=canonical_url,
                        title=clean_title(primary_title),
                        body="\n".join(body_lines),
                        published_at=f"{pub_date.isoformat()}T00:00:00Z",
                        publisher=self.publisher,
                        source_type=self.source_type,
                        source_key=self.source_key,
                        lang=self.lang,
                        source_tz="UTC",
                        external_media_refs=(("anilist", media_id),),
                    )
                )

            has_next = bool((page_obj.get("pageInfo") or {}).get("hasNextPage"))
            if not has_next:
                break

        return items

    def _fetch_bangumi(self, *, max_pages: int) -> list[RawIngestItem]:
        if self.cache_responses and self._cached_items is not None:
            return list(self._cached_items)

        self.last_skipped_entries = 0
        self.last_supplementary_errors = 0
        items: list[RawIngestItem] = []
        seen_urls: set[str] = set()
        # Subject types: 2 = Anime, 4 = Game, 1 = Book (Manga / Light Novel)
        subject_types = (2, 4, 1)
        for stype in subject_types:
            for page in range(max_pages):
                offset = page * self.per_page
                sep = "&" if "?" in self.endpoint_url else "?"
                url = f"{self.endpoint_url}{sep}type={stype}&limit={self.per_page}&offset={offset}"
                res = self.http.get(
                    url, headers={"Accept": "application/json"}, use_conditional=False
                )
                try:
                    payload = json.loads(res.text)
                except json.JSONDecodeError as exc:
                    raise SelectorBrokenError(
                        f"[{self.source_key}] Invalid JSON from Bangumi API {url!r}: {exc}"
                    ) from exc

                rows: list[Any]
                if isinstance(payload, dict) and isinstance(payload.get("data"), list):
                    rows = payload["data"]
                elif isinstance(payload, list):
                    rows = payload
                else:
                    raise SelectorBrokenError(
                        f"[{self.source_key}] Unexpected Bangumi response shape at {url!r}"
                    )

                if not rows:
                    break

                type_label = {1: "manga/light_novel", 2: "anime", 4: "game"}.get(stype, "acgn")
                for subj in rows:
                    if not isinstance(subj, dict) or not subj.get("id"):
                        self.last_skipped_entries += 1
                        continue
                    sid = str(subj["id"])
                    name = str(subj.get("name") or "").strip()
                    name_cn = str(subj.get("name_cn") or "").strip()
                    raw_date = str(subj.get("date") or "").strip()
                    summary = str(subj.get("summary") or "").strip()
                    if not (name_cn or name) or not raw_date:
                        self.last_skipped_entries += 1
                        continue
                    try:
                        pub_utc = parse_utc_timestamp(raw_date, source_tz=self.source_tz)
                    except Exception:
                        self.last_skipped_entries += 1
                        continue

                    title = clean_title(name_cn or name)
                    platform = str(subj.get("platform") or type_label).strip()
                    canonical_url = normalize_url(f"https://bgm.tv/subject/{sid}")
                    if canonical_url in seen_urls:
                        continue
                    seen_urls.add(canonical_url)

                    body_parts = [
                        f"Title: {title}",
                        f"Original Name: {name}" if name and name != title else "",
                        f"Category: {type_label} ({platform})",
                        f"Release Date: {pub_utc[:10]}",
                        f"Summary: {normalize_text(summary)}" if summary else "",
                    ]
                    body = "\n".join(p for p in body_parts if p)
                    items.append(
                        RawIngestItem(
                            url=canonical_url,
                            title=title,
                            body=body,
                            published_at=pub_utc,
                            publisher=self.publisher,
                            source_type=self.source_type,
                            source_key=self.source_key,
                            lang=self.lang,
                            source_tz=self.source_tz,
                            external_media_refs=(("bangumi", sid),),
                        )
                    )

        if not items:
            raise SelectorBrokenError(
                f"[{self.source_key}] Bangumi API {self.endpoint_url!r} returned 0 valid items"
            )

        # Supplement with Bangumi's live airing calendar endpoint when hitting api.bgm.tv so
        # recent 2023-2026 seasonal subjects are covered alongside historical /v0/subjects IDs.
        if "api.bgm.tv" in self.endpoint_url:
            try:
                cal_res = self.http.get(
                    "https://api.bgm.tv/calendar",
                    headers={"Accept": "application/json"},
                    use_conditional=False,
                )
                cal_payload = json.loads(cal_res.text)
                if isinstance(cal_payload, list):
                    for day_block in cal_payload:
                        if not isinstance(day_block, dict):
                            continue
                        for subj in day_block.get("items") or []:
                            if not isinstance(subj, dict) or not subj.get("id"):
                                self.last_skipped_entries += 1
                                continue
                            sid = str(subj["id"])
                            name = str(subj.get("name") or "").strip()
                            name_cn = str(subj.get("name_cn") or "").strip()
                            raw_date = str(subj.get("air_date") or "").strip()
                            if not (name_cn or name) or not raw_date:
                                self.last_skipped_entries += 1
                                continue
                            try:
                                pub_utc = parse_utc_timestamp(raw_date, source_tz=self.source_tz)
                            except Exception:
                                self.last_skipped_entries += 1
                                continue
                            title = clean_title(name_cn or name)
                            canonical_url = normalize_url(f"https://bgm.tv/subject/{sid}")
                            if canonical_url in seen_urls:
                                continue
                            seen_urls.add(canonical_url)
                            body = (
                                f"Title: {title}\n"
                                f"Original Name: {name}\n"
                                f"Category: anime (TV/Web)\n"
                                f"Release Date: {pub_utc[:10]}"
                            )
                            items.append(
                                RawIngestItem(
                                    url=canonical_url,
                                    title=title,
                                    body=body,
                                    published_at=pub_utc,
                                    publisher=self.publisher,
                                    source_type=self.source_type,
                                    source_key=self.source_key,
                                    lang=self.lang,
                                    source_tz=self.source_tz,
                                    external_media_refs=(("bangumi", sid),),
                                )
                            )
            except Exception:
                self.last_supplementary_errors += 1

        if self.cache_responses:
            self._cached_items = list(items)
        return items

    def _fetch_bilibili(self) -> list[RawIngestItem]:
        if self.cache_responses and self._cached_items is not None:
            return list(self._cached_items)

        self.last_skipped_entries = 0
        res = self.http.get(
            self.endpoint_url,
            headers={"Accept": "application/json"},
            use_conditional=False,
        )
        try:
            payload = json.loads(res.text)
        except json.JSONDecodeError as exc:
            raise SelectorBrokenError(
                f"[{self.source_key}] Invalid JSON from Bilibili timeline: {exc}"
            ) from exc

        result_days = payload.get("result") if isinstance(payload, dict) else None
        if not isinstance(result_days, list) or not result_days:
            raise SelectorBrokenError(
                f"[{self.source_key}] Missing or empty `result` array in Bilibili timeline"
            )

        items: list[RawIngestItem] = []
        seen_urls: set[str] = set()
        for day_block in result_days:
            if not isinstance(day_block, dict):
                continue
            day_date = str(day_block.get("date") or "").strip()
            episodes = day_block.get("episodes") or []
            if not isinstance(episodes, list):
                continue
            for ep in episodes:
                if not isinstance(ep, dict):
                    self.last_skipped_entries += 1
                    continue
                season_id = ep.get("season_id")
                ep_id = ep.get("episode_id") or season_id
                raw_title = str(ep.get("title") or "").strip()
                pub_index = str(ep.get("pub_index") or "").strip()
                pub_time = str(ep.get("pub_time") or "").strip()
                ts_val = ep.get("pub_ts")
                if not raw_title or not ep_id:
                    self.last_skipped_entries += 1
                    continue

                if isinstance(ts_val, (int, float)) and ts_val > 0:
                    dt = datetime.fromtimestamp(int(ts_val), tz=UTC)
                    pub_utc = dt.strftime("%Y-%m-%dT%H:%M:%SZ")
                elif day_date:
                    try:
                        pub_utc = parse_utc_timestamp(day_date, source_tz=self.source_tz)
                    except Exception:
                        self.last_skipped_entries += 1
                        continue
                else:
                    self.last_skipped_entries += 1
                    continue

                url = normalize_url(f"https://www.bilibili.com/bangumi/play/ep{ep_id}")
                if url in seen_urls:
                    continue
                seen_urls.add(url)

                full_title = clean_title(f"{raw_title} {pub_index}".strip())
                body = f"Title: {raw_title}\nEpisode: {pub_index}\nBroadcast: {day_date} {pub_time}".strip()
                refs = (("bilibili_season", str(season_id)),) if season_id else ()
                items.append(
                    RawIngestItem(
                        url=url,
                        title=full_title,
                        body=body,
                        published_at=pub_utc,
                        publisher=self.publisher,
                        source_type=self.source_type,
                        source_key=self.source_key,
                        lang=self.lang,
                        source_tz=self.source_tz,
                        event_type="broadcast",
                        external_media_refs=refs,
                    )
                )

        if not items:
            raise SelectorBrokenError(
                f"[{self.source_key}] Bilibili timeline returned 0 parseable episode rows"
            )
        if self.cache_responses:
            self._cached_items = list(items)
        return items

    def _fetch_syosetu(self) -> list[RawIngestItem]:
        """Fetch light-novel metadata from Syosetu `novelapi` (strictly metadata-only per C2)."""
        if self.cache_responses and self._cached_items is not None:
            return list(self._cached_items)

        self.last_skipped_entries = 0
        res = self.http.get(self.endpoint_url, use_conditional=False)
        try:
            payload = json.loads(res.text)
        except json.JSONDecodeError as exc:
            raise SelectorBrokenError(
                f"[{self.source_key}] Invalid JSON from Syosetu novelapi: {exc}"
            ) from exc

        if not isinstance(payload, list) or len(payload) < 2:
            raise SelectorBrokenError(
                f"[{self.source_key}] Syosetu novelapi returned empty or malformed array"
            )

        items: list[RawIngestItem] = []
        seen_urls: set[str] = set()
        # Element 0 is `{"allcount": N}`; elements 1.. are novel metadata records
        for row in payload[1:]:
            if not isinstance(row, dict):
                self.last_skipped_entries += 1
                continue
            ncode = str(row.get("ncode") or "").strip().lower()
            raw_title = str(row.get("title") or "").strip()
            writer = str(row.get("writer") or "").strip()
            # Use general_firstup (immutable initial publication timestamp) before novelupdated_at
            # so published_at does not drift across chapter updates.
            raw_date = str(row.get("general_firstup") or row.get("novelupdated_at") or "").strip()
            keyword = str(row.get("keyword") or "").strip()
            story = str(row.get("story") or "").strip()
            episodes = row.get("general_all_no")
            if not ncode or not raw_title or not raw_date:
                self.last_skipped_entries += 1
                continue

            try:
                pub_utc = parse_utc_timestamp(raw_date, source_tz=self.source_tz)
            except Exception:
                self.last_skipped_entries += 1
                continue

            # Canonical reference URL; never fetches novel body pages (COMPLIANCE C2)
            canonical_url = normalize_url(f"https://ncode.syosetu.com/{ncode}/")
            if canonical_url in seen_urls:
                continue
            seen_urls.add(canonical_url)

            title = clean_title(raw_title)
            body_lines = [
                f"Light Novel Title: {title}",
                f"NCode: {ncode.upper()}",
                f"Author: {writer}" if writer else "",
                f"First Published: {pub_utc[:10]}",
                f"Chapters: {episodes}" if episodes is not None else "",
                f"Keywords: {normalize_text(keyword)}" if keyword else "",
                f"Synopsis: {normalize_text(story)}" if story else "",
            ]
            body = "\n".join(line for line in body_lines if line)
            items.append(
                RawIngestItem(
                    url=canonical_url,
                    title=title,
                    body=body,
                    published_at=pub_utc,
                    publisher=self.publisher,
                    source_type=self.source_type,
                    source_key=self.source_key,
                    lang=self.lang,
                    source_tz=self.source_tz,
                    external_media_refs=(("syosetu", ncode),),
                )
            )

        if not items:
            raise SelectorBrokenError(
                f"[{self.source_key}] Syosetu novelapi yielded 0 valid metadata records"
            )
        if self.cache_responses:
            self._cached_items = list(items)
        return items

    def _fetch_provider_items(
        self,
        *,
        start_d: date | None,
        end_d: date | None,
        max_pages: int,
    ) -> list[RawIngestItem]:
        if self.provider == "anilist":
            return self._fetch_anilist(start_d=start_d, end_d=end_d, max_pages=max_pages)
        if self.provider == "bangumi":
            return self._fetch_bangumi(max_pages=max_pages)
        if self.provider == "bilibili":
            return self._fetch_bilibili()
        if self.provider == "syosetu":
            return self._fetch_syosetu()
        raise ValueError(f"Unsupported StructuredApiAdapter provider: {self.provider!r}")

    def discover(
        self,
        since: date | datetime | str | None = None,
        *,
        use_conditional: bool = True,
    ) -> list[RawIngestItem]:
        """Incremental metadata discovery."""
        self.last_not_modified = False
        _ = use_conditional
        since_ts: str | None = None
        since_date: date | None = None
        if since is not None:
            since_ts = parse_utc_timestamp(since, source_tz=self.source_tz)
            since_date = date.fromisoformat(since_ts[:10])

        items = self._fetch_provider_items(start_d=since_date, end_d=None, max_pages=1)
        if since_ts is None:
            return items
        return [item for item in items if utc_iso_at_or_after(item.published_at, since_ts)]

    def backfill(
        self,
        date_range: tuple[date | datetime | str | None, date | datetime | str | None] | None = None,
    ) -> list[RawIngestItem]:
        """Historical metadata backfill within `[start_date, end_date]`."""
        start_d, end_d = validate_backfill_window(date_range)
        start_iso = start_d.isoformat()
        end_iso = end_d.isoformat()
        if self.provider == "bilibili":
            raise BackfillNotSupportedError(
                f"[{self.source_key}] Bilibili timeline is a rolling ±7d endpoint and cannot "
                f"backfill historical window {start_iso}..{end_iso}"
            )

        items = self._fetch_provider_items(
            start_d=start_d,
            end_d=end_d,
            max_pages=self.max_pages,
        )
        window_items = [item for item in items if start_iso <= item.published_at[:10] <= end_iso]
        if not window_items:
            raise EmptyBackfillWindowError(
                f"[{self.source_key}] 0 API metadata items fell inside window {start_iso}..{end_iso}"
            )
        return window_items


__all__ = [
    "ANILIST_GRAPHQL_URL",
    "BANGUMI_SUBJECTS_URL",
    "BILIBILI_TIMELINE_URL",
    "SYOSETU_NOVELAPI_URL",
    "StructuredApiAdapter",
]
