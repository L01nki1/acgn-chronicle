/**
 * Interface copy for the daily page, in English (default) and Chinese.
 *
 * Owner: C | UI chrome only. Corpus content (titles, quotes, publishers, dates, refusal reasons)
 * comes from the server and is shown in the language the corpus was written in.
 */

export const DEFAULT_LANG = "en";
export const LANGS = Object.freeze(["en", "zh"]);
const STORAGE_KEY = "acgn:lang";

const EN = {
  "site.title": "ACGN Chronicle · Archive Search",
  "site.description": "ACGN Chronicle's daily archive search, served from fixed mock data.",
  "a11y.skip": "Skip to main content",
  "brand.aria": "ACGN Chronicle",
  "brand.name": "ACGN Chronicle",
  "brand.sub": "Daily Archive",
  "nav.aria": "Page navigation",
  "nav.ask": "Ask",
  "nav.results": "Results",
  "nav.contract": "Contract",
  "chip.demo": "DEMO · mock data",
  "lang.aria": "Language",

  "hero.eyebrow": "ACGN Daily · Daily archive",
  "hero.title1": "Ask the daily,",
  "hero.title2": "leave a clue.",
  "hero.intro": "Ask one question and the archive retrieves the evidence for you.",
  "hero.scroll": "Scroll to the archive",

  "form.question": "Question or search terms",
  "form.submit.retrieve": "Search excerpts",
  "form.submit.ask": "Answer with citations",
  "form.mode": "Output",
  "form.mode.retrieve": "Excerpts",
  "form.mode.ask": "Cited answer",
  "form.filters": "Advanced filters",
  "form.sources": "Source type",
  "form.since": "Start date",
  "form.until": "End date",
  "form.media": "Media ID (optional)",
  "form.topk": "Results per query",
  "form.note.retrieve":
    "Results are bound to the snapshot below; an empty result and a refusal are two different states.",
  "form.note.ask":
    "A refusal shows its reason with zero citations; an empty result is never called a refusal.",
  "form.count": "{used} / {limit}",
  "examples.title": "Example questions",

  "source.official": "Official",
  "source.media": "Media",
  "source.metadata": "Metadata",
  "source.unknown": "Unknown source",

  "banner.tag": "Note",
  "banner.text":
    "This page runs on fixed mock data; citation links point to the example domain example.com.",

  "results.eyebrow": "Archive Results",
  "results.title.retrieve": "Search results",
  "results.title.ask": "Cited answer",
  "meta.waiting": "Waiting for a query",
  "meta.formChanged": "form changed; these results belong to the previous request",
  "meta.mode.retrieve": "Excerpt retrieval",
  "meta.mode.ask": "Cited answer",
  "meta.cached": "cache hit",
  "meta.thisResponse": "this response",
  "meta.failure": "Request failed · HTTP {status}",

  "overview.title": "Issue overview",
  "overview.nothing": "Nothing to preview in this response.",
  "overview.excerpts": "Excerpts",
  "overview.citations": "Citations",
  "overview.passed": "passed the citation check",
  "overview.sourceTypes": "Source types",
  "overview.inResponse": "within this response",
  "overview.avg": "Average score",
  "overview.span": "Time span",
  "overview.snapshot": "Snapshot",
  "overview.hit": "Hit",
  "overview.cache": "Cache",
  "overview.sameKey": "same cache key",
  "overview.initial": "Run a query and the issue overview appears here.",
  "overview.refusal": "The system refused; no citations to preview.",
  "overview.empty": "An empty result is a valid answer, not a refusal.",
  "overview.hidden": "{n} results lacked source information and were not rendered.",
  "overview.error": "The request failed; the overview is unavailable.",

  "snapshot.label": "Snapshot",
  "snapshot.reading": "Reading…",
  "snapshot.config": "Config A · mock",
  "snapshot.current": "Current",
  "snapshot.retained": "Retained",
  "snapshot.bound": "Bound {id}",
  "snapshot.none": "No snapshot selected",

  "views.aria": "Result view",
  "view.cards": "Cards",
  "view.timeline": "Timeline",
  "view.magazine": "Magazine",
  "view.board": "Board",
  "views.hint": "One result · four readings",
  "results.placeholder": "Reading the archive…",

  "rail.aria": "Quantitative profile for this query",
  "rail.title": "This profile",
  "rail.empty": "Run a query to see the source mix, scores and time spread.",

  "contract.eyebrow": "Contract",
  "contract.title": "What this page guarantees",
  "contract.1": "A citation without a URL is never rendered, greyed out or tucked into a tooltip.",
  "contract.2":
    "Bodies are clipped once on the server and again before client rendering, at 200 characters.",
  "contract.3":
    "Every result shows its snapshot; the cache key covers snapshot, question, filters, top_k and config.",
  "contract.4": "410 means the snapshot is retired; the page never switches to the current one.",
  "contract.5": "No excerpts is not a refusal; a refusal always states its reason.",

  "footer.brand": "ACGN Chronicle",
  "footer.sub": "ACGN Daily · Daily search",
  "footer.rules": "Updated daily · Traceable sources · Replayable snapshots",
  "detail.kicker": "Archive Detail",
  "detail.close": "Close details",
  "noscript": "This page needs JavaScript to load the archive.",

  "note.queued": "The previous request is still running; this one is queued and starts automatically.",
  "note.requesting": "Requesting the mock snapshot…",
  "note.cache": "Cache hit; the cache key covers the snapshot and the filters.",
  "note.done": "Request complete.",
  "note.failed": "Request failed; the reason is in the results panel.",
  "note.refreshed": "Snapshot list refreshed; confirm the selection and rerun.",
  "note.switched": "Snapshot switched; rerun to get results for the new snapshot.",
  "note.inventoryFailed": "Could not read the snapshot list; check that the local host is running.",

  "filters.none": "No filters",
  "filters.sources": "Sources {list}",
  "filters.dates": "Dates {since} → {until}",
  "filters.media": "Media ID {ids}",
  "filters.any": "Any",
  "list.sep": ", ",

  "err.form.empty": "Enter a question or search terms.",
  "err.form.noSnapshot": "No snapshot is selected yet; refresh the snapshot list first.",
  "err.form.mediaId": "The media ID must be a positive integer.",
  "err.form.sources":
    "Select at least one source type; selecting none is not the same as not filtering.",
  "err.form.dateOrder": "The start date cannot be later than the end date.",

  "retrieve.compliance":
    "{n} excerpts passed the citation check; ranked only within this response; each quote is at most {limit} characters.",
  "retrieve.empty": "No excerpts matched the current filters.",
  "retrieve.chars": "Chars {n} / {limit}",
  "retrieve.clipped": "Clipped client-side",
  "citation.open": "Open original",
  "citation.openTitle": "Open the original in a new window: {url}",
  "citation.source": "Source",

  "answer.title": "Cited answer",
  "answer.snapshot": "Snapshot {id}",
  "answer.cited": "cited_chunk_ids {n}",
  "answer.displayOnly": "The answer is shown as returned; the client never completes citations.",
  "answer.caveat":
    "This answer comes from local mock data; the server keeps cited_chunk_ids ⊆ retrieved and the client only checks whether a citation can be rendered.",
  "answer.compliance":
    "{n} citations carry a URL and source fields and are shown per contract; “Open original” opens the source in a new window.",
  "answer.noCitations":
    "This answer carries no citations; with an empty citation list the page renders no source cards.",
  "answer.citationMeta": "Citation drawn from this answer's retrieved evidence",

  "empty.title": "No excerpts found",
  "empty.explainer":
    "An empty result means there is no evidence under the current filters; it is not a refusal. The page keeps the two states distinct.",
  "empty.filters": "Filters: {summary}",
  "empty.snapshot": "Snapshot: {id}",
  "empty.clear": "Search without filters",

  "refusal.title": "The system refused to answer",
  "refusal.missingReason": "The server returned no refusal reason.",
  "refusal.explainer":
    "A refusal is not an empty result: the knowledge base lacks enough evidence, so no citation can be shown.",
  "refusal.reason": "Reason: {reason}",
  "refusal.counts": "Citations: 0; cited_chunk_ids: 0",

  "compliance.title": "Results were not rendered without source information",
  "compliance.detail":
    "{n} results lack a URL or a required source field; each is hidden entirely under “no URL, no render”.",
  "compliance.explainer":
    "The page never greys out a card, never fills missing fields with placeholders, and never shows untraceable content just to fill space.",
  "compliance.hidden":
    "{n} results lack a URL or a required source field and were hidden as a whole under “no URL, no render”.",

  "stats.tab.source": "Sources",
  "stats.tab.metrics": "Metrics",
  "stats.tab.distribution": "Distribution",
  "stats.tab.time": "Time",
  "stats.panelAria": "{tab} {index} / {total}",
  "stats.tabFallback": "Profile {n}",
  "stats.prev": "Previous profile",
  "stats.next": "Next profile",
  "stats.group": "Profile groups",
  "stats.loading": "Computing this response's profile…",
  "stats.noCitations": "No citations to summarise in this response.",
  "stats.noExcerpts": "No excerpts to summarise under these filters.",
  "stats.noAnswerSources": "This answer has no citations, so sources cannot be summarised.",
  "stats.paused": "Nothing renderable to summarise; stats are paused.",
  "stats.sourcesTitle": "Source mix",
  "stats.countLabel": "{n} items",
  "stats.sourceShare": "{label} {n} items, {share}%",
  "stats.idleSource": "Cited source",
  "stats.avgScore": "avg {score}",
  "stats.metricsTitle.retrieve": "Quantitative slice",
  "stats.metricsTitle.ask": "Answer slice",
  "stats.metricsCount.retrieve": "5 metrics",
  "stats.metricsCount.ask": "4 metrics",
  "stats.avg": "Average score",
  "stats.max": "Top score",
  "stats.avgHint": "within this response",
  "stats.maxHint": "for ordering only",
  "stats.span": "Time span",
  "stats.days": "{n} days",
  "stats.singleDay": "Single day",
  "stats.dateCount": "{n} dates",
  "stats.avgChars": "Avg. length",
  "stats.chars": "{n} chars",
  "stats.limit": "limit {n}",
  "stats.citations": "Citations",
  "stats.publishers": "Publishers",
  "stats.publisherCount": "{n} unique",
  "stats.dedup": "distinct publishers",
  "stats.answerChars": "Answer length",
  "stats.displayOnly": "shown, never completed",
  "stats.meterTitle.retrieve": "Score distribution",
  "stats.meterTitle.ask": "Evidence strength",
  "stats.meterCount.retrieve": "by score",
  "stats.meterCount.ask": "by source",
  "stats.published": "Published timeline",
  "stats.months": "{n} months",
  "stats.noDates": "No dates",
  "stats.noDatesNote": "No cited source has a parseable publication date.",

  "detail.cta.cards": "Open full record →",
  "detail.cta.timeline": "Open record →",
  "detail.cta.magazine": "Read in full →",
  "detail.aria": "View the full record for “{title}”",
  "detail.clipped":
    "This quote was clipped at the {limit}-character ceiling; the client will not expand or complete it.",
  "detail.citationOnly":
    "Citations in an answer carry source metadata only, not the body excerpt; body excerpts are returned by excerpt search only.",
  "detail.fields": "Record fields",
  "detail.sourceType": "Source type",
  "detail.publisher": "Publisher",
  "detail.publishedAt": "Published at",
  "detail.score": "Rank score",
  "detail.quoteChars": "Quote length",
  "detail.snapshot": "Snapshot",
  "detail.context": "Request context",
  "detail.question": "Question",
  "detail.mode": "Mode",
  "detail.filters": "Filters",
  "detail.config": "Config",
  "detail.cacheKey": "Cache key",
  "detail.links": "Source link",
  "detail.linksNote":
    "Only fields returned by the server are shown; missing fields are never filled in on the client.",
  "detail.peers": "Same batch",

  "board.rank.retrieve": "Score ranking",
  "board.rank.ask": "Evidence strength",
  "board.sub.retrieve": "Ranked only within this response",
  "board.sub.ask": "Official 3 / Media 2 / Metadata 1",
  "board.items": "{n} items",
  "board.unit": "items",
  "board.sourceMix": "Source mix",
  "board.total": "Items",
  "board.thisResponse": "this response",
  "board.publishers": "Publishers",
  "board.dedup": "distinct publishers",
  "board.span": "Time span",
  "board.snapshot": "Snapshot",
  "board.config": "Config",
  "board.mockCorpus": "mock corpus",

  "error.network": "network",
  "error.connect": "Cannot reach the local service: {message}",
  "error.badRequest.title": "The request was malformed",
  "error.badRequest.text": "The server rejected the request body; fix the input and submit again.",
  "error.notFound.title": "Snapshot not found",
  "error.notFound.text":
    "The local host has never seen this id; the page will not guess another snapshot.",
  "error.refresh": "Refresh snapshot list",
  "error.retired.title": "Snapshot retired",
  "error.retired.text":
    "This snapshot is beyond the retention window. The page will not switch to the current one; rerunning with it is an explicit choice.",
  "error.rerun": "Rerun with the current snapshot",
  "error.invalid.title": "Invalid filters",
  "error.invalid.text":
    "The server returned 422; check the date range or the filter combination and retry.",
  "error.clearDates": "Clear date filters",
  "error.unavailable.title": "Service unavailable",
  "error.unavailable.text":
    "The local host has no complete snapshot to serve yet, or is switching safely.",
  "error.retryAfter": "Retry-After: {seconds}s",
  "error.retry": "Retry",
  "error.generic.title": "Request failed",
  "error.generic.text": "The request could not complete; nothing untraceable was shown.",
  "error.unknown": "Unknown error",
  "error.requestLine": "Request snapshot: {snapshot} · filters: {filters}",
  "error.summary": "Request failed: HTTP {status}.",

  "refresh.done.title": "Snapshot list refreshed",
  "refresh.done.text":
    "Choose a snapshot and rerun explicitly; the page does not rewrite the original request.",
  "refresh.done.current": "Current snapshot: {id}",
  "refresh.unknown": "unknown",
};

const ZH = {
  "site.title": "ACGN 日报 · 档案检索",
  "site.description": "ACGN 日报的档案检索页面，使用固定 mock 数据。",
  "a11y.skip": "跳到主要内容",
  "brand.aria": "ACGN 日报",
  "brand.name": "ACGN 日报",
  "brand.sub": "ACGN Daily",
  "nav.aria": "页面导航",
  "nav.ask": "问档案",
  "nav.results": "检索结果",
  "nav.contract": "页面契约",
  "chip.demo": "DEMO · mock 数据",
  "lang.aria": "语言",

  "hero.eyebrow": "ACGN Daily · 每日档案",
  "hero.title1": "问日报，",
  "hero.title2": "留一条线索。",
  "hero.intro": "输入一个问题，日报会帮你检索答案",
  "hero.scroll": "向下，查看档案",

  "form.question": "问题或检索词",
  "form.submit.retrieve": "检索片段",
  "form.submit.ask": "带引用问答",
  "form.mode": "输出",
  "form.mode.retrieve": "片段检索",
  "form.mode.ask": "带引用问答",
  "form.filters": "高级筛选",
  "form.sources": "来源类型",
  "form.since": "开始日期",
  "form.until": "结束日期",
  "form.media": "作品 ID（可选）",
  "form.topk": "返回条数",
  "form.note.retrieve": "结果绑定到下方快照；空结果与拒答是两种状态。",
  "form.note.ask": "拒答会显示原因与空引用；空结果不会被说成拒答。",
  "form.count": "{used} / {limit}",
  "examples.title": "示例问题",

  "source.official": "官方",
  "source.media": "媒体",
  "source.metadata": "元数据",
  "source.unknown": "未知来源",

  "banner.tag": "提示",
  "banner.text": "当前页使用固定 mock 数据，引用链接指向示例地址 example.com。",

  "results.eyebrow": "Archive Results · 档案结果",
  "results.title.retrieve": "检索结果",
  "results.title.ask": "带引用回答",
  "meta.waiting": "等待请求",
  "meta.formChanged": "表单已修改，结果对应上一次请求",
  "meta.mode.retrieve": "片段检索",
  "meta.mode.ask": "带引用问答",
  "meta.cached": "缓存命中",
  "meta.thisResponse": "本次响应",
  "meta.failure": "请求失败 · HTTP {status}",

  "overview.title": "本期速览",
  "overview.nothing": "本次响应没有可速览的条目。",
  "overview.excerpts": "片段",
  "overview.citations": "引用",
  "overview.passed": "通过引用检查",
  "overview.sourceTypes": "来源类型",
  "overview.inResponse": "本次响应内",
  "overview.avg": "平均分",
  "overview.span": "时间跨度",
  "overview.snapshot": "快照",
  "overview.hit": "命中",
  "overview.cache": "缓存",
  "overview.sameKey": "同一缓存键",
  "overview.initial": "提交检索后，这里给出本期速览。",
  "overview.refusal": "系统拒答；没有引用可速览。",
  "overview.empty": "空结果是有效回答，不是拒答。",
  "overview.hidden": "{n} 条结果因缺少来源信息未渲染。",
  "overview.error": "请求失败；速览不可用。",

  "snapshot.label": "快照",
  "snapshot.reading": "正在读取…",
  "snapshot.config": "配置 A · mock",
  "snapshot.current": "当前",
  "snapshot.retained": "保留",
  "snapshot.bound": "绑定 {id}",
  "snapshot.none": "未选择快照",

  "views.aria": "结果视图切换",
  "view.cards": "卡片流",
  "view.timeline": "时间轴",
  "view.magazine": "杂志",
  "view.board": "数据看板",
  "views.hint": "同一份结果 · 四种读法",
  "results.placeholder": "正在读取日报数据…",

  "rail.aria": "本次检索的量化剖面",
  "rail.title": "本次剖面",
  "rail.empty": "提交检索后显示来源构成、分数与时间分布。",

  "contract.eyebrow": "Contract · 页面契约",
  "contract.title": "页面遵守的契约",
  "contract.1": "引用对象没有 URL 就不渲染，不灰显、不藏在提示里。",
  "contract.2": "正文在服务端裁剪一次，客户端渲染前再裁剪一次，上限 200 个字符。",
  "contract.3": "每次结果都显示对应快照；缓存键包含快照、问题、筛选、top_k 与配置。",
  "contract.4": "410 表示快照已退役，页面不会自动切到当前快照。",
  "contract.5": "没有检索到片段不是拒答；拒答一定会给出原因。",

  "footer.brand": "ACGN 日报",
  "footer.sub": "ACGN Daily · 每日检索",
  "footer.rules": "每日更新 · 来源可追溯 · 快照可回看",
  "detail.kicker": "Archive Detail · 档案详情",
  "detail.close": "关闭详情",
  "noscript": "这个页面需要 JavaScript 才能加载日报数据。",

  "note.queued": "上一个请求尚未完成；这次操作已排队，完成后自动运行。",
  "note.requesting": "正在请求 mock 快照…",
  "note.cache": "缓存命中；缓存键包含快照与筛选条件。",
  "note.done": "请求完成。",
  "note.failed": "请求失败：原因见结果区。",
  "note.refreshed": "快照列表已刷新；请确认选择后重跑。",
  "note.switched": "已切换快照；重新提交后才会得到该快照的结果。",
  "note.inventoryFailed": "无法读取快照列表；请确认本地服务已启动。",

  "filters.none": "无筛选",
  "filters.sources": "来源 {list}",
  "filters.dates": "日期 {since} → {until}",
  "filters.media": "作品 ID {ids}",
  "filters.any": "不限",
  "list.sep": "；",

  "err.form.empty": "请输入问题或检索词。",
  "err.form.noSnapshot": "还没有选择快照；请先刷新快照列表。",
  "err.form.mediaId": "作品 ID 必须是正整数。",
  "err.form.sources": "至少选择一个来源类型；全部不选不等于不筛选。",
  "err.form.dateOrder": "开始日期不能晚于结束日期。",

  "retrieve.compliance":
    "{n} 条片段通过引用检查；按 score 排序，仅在本次响应内有效；每条引文最多 {limit} 个字符。",
  "retrieve.empty": "当前筛选条件下没有检索到片段。",
  "retrieve.chars": "字符 {n} / {limit}",
  "retrieve.clipped": "客户端二次裁剪",
  "citation.open": "打开原文",
  "citation.openTitle": "在新窗口打开原文：{url}",
  "citation.source": "来源",

  "answer.title": "带引用回答",
  "answer.snapshot": "快照 {id}",
  "answer.cited": "cited_chunk_ids {n} 条",
  "answer.displayOnly": "回答只展示，不在客户端补全引文",
  "answer.caveat":
    "当前回答来自本地 mock 数据；服务端负责 cited_chunk_ids ⊆ retrieved，客户端只检查引用能否展示。",
  "answer.compliance":
    "{n} 条引用带 URL 与来源字段，按契约展示；“打开原文”在新窗口打开来源链接。",
  "answer.noCitations": "本次回答没有附带引用；引用为空时页面不渲染任何来源卡片。",
  "answer.citationMeta": "引用来自本次回答的检索证据",

  "empty.title": "没有检索到片段",
  "empty.explainer":
    "空结果表示当前筛选下没有证据，不是系统拒答。页面会保持这两种状态的区别。",
  "empty.filters": "筛选：{summary}",
  "empty.snapshot": "快照：{id}",
  "empty.clear": "改用无筛选检索",

  "refusal.title": "系统拒绝回答",
  "refusal.missingReason": "服务端没有提供拒答原因。",
  "refusal.explainer": "拒答不是空结果：它表示知识库没有足够依据，因此没有引用可展示。",
  "refusal.reason": "原因：{reason}",
  "refusal.counts": "引用数量：0；cited_chunk_ids：0",

  "compliance.title": "结果因缺少来源信息而未渲染",
  "compliance.detail": "{n} 条结果缺少 URL 或必需来源字段；按“无 URL，不渲染”整条隐藏。",
  "compliance.explainer":
    "页面不会灰显、不会把缺失字段补成占位符，也不会为了好看而展示不可追溯的内容。",
  "compliance.hidden":
    "{n} 条结果缺少 URL 或必需来源字段，已按“无 URL，不渲染”整体隐藏。",

  "stats.tab.source": "来源",
  "stats.tab.metrics": "量化",
  "stats.tab.distribution": "分布",
  "stats.tab.time": "时间",
  "stats.panelAria": "{tab} {index} / {total}",
  "stats.tabFallback": "第 {n} 页",
  "stats.prev": "上一张剖面",
  "stats.next": "下一张剖面",
  "stats.group": "剖面分组",
  "stats.loading": "正在计算本次响应剖面…",
  "stats.noCitations": "本次没有可统计的引用。",
  "stats.noExcerpts": "本次筛选没有可统计的片段。",
  "stats.noAnswerSources": "本次回答没有引用，无法统计来源。",
  "stats.paused": "没有可展示的引用，统计暂停。",
  "stats.sourcesTitle": "来源构成",
  "stats.countLabel": "{n} 条",
  "stats.sourceShare": "{label} {n} 条，占 {share}%",
  "stats.idleSource": "引用来源",
  "stats.avgScore": "均分 {score}",
  "stats.metricsTitle.retrieve": "量化切片",
  "stats.metricsTitle.ask": "回答切片",
  "stats.metricsCount.retrieve": "5 项",
  "stats.metricsCount.ask": "4 项",
  "stats.avg": "平均分",
  "stats.max": "最高分",
  "stats.avgHint": "本次响应内",
  "stats.maxHint": "仅用于排序",
  "stats.span": "时间跨度",
  "stats.days": "{n} 天",
  "stats.singleDay": "单日",
  "stats.dateCount": "{n} 个日期",
  "stats.avgChars": "平均字符",
  "stats.chars": "{n} 字",
  "stats.limit": "上限 {n}",
  "stats.citations": "引用",
  "stats.publishers": "来源方",
  "stats.publisherCount": "{n} 个",
  "stats.dedup": "发布方去重",
  "stats.answerChars": "回答",
  "stats.displayOnly": "只展示，不补全",
  "stats.meterTitle.retrieve": "分数分布",
  "stats.meterTitle.ask": "证据等级",
  "stats.meterCount.retrieve": "按 score",
  "stats.meterCount.ask": "按来源",
  "stats.published": "发布时间线",
  "stats.months": "{n} 个月",
  "stats.noDates": "无日期",
  "stats.noDatesNote": "本次引用没有可解析的发布日期。",

  "detail.cta.cards": "展开完整档案 →",
  "detail.cta.timeline": "展开档案 →",
  "detail.cta.magazine": "展开全文 →",
  "detail.aria": "查看「{title}」的完整档案",
  "detail.clipped": "该引文已按 {limit} 字符上限裁剪；客户端不会展开或补全。",
  "detail.citationOnly":
    "本次回答的引用只携带来源元数据，不包含正文摘录；正文摘录只在片段检索中返回。",
  "detail.fields": "档案字段",
  "detail.sourceType": "来源类型",
  "detail.publisher": "发布方",
  "detail.publishedAt": "发布日期",
  "detail.score": "排序分数",
  "detail.quoteChars": "引文字符",
  "detail.snapshot": "快照",
  "detail.context": "请求上下文",
  "detail.question": "问题",
  "detail.mode": "模式",
  "detail.filters": "筛选",
  "detail.config": "配置",
  "detail.cacheKey": "缓存键",
  "detail.links": "来源链接",
  "detail.linksNote": "仅展示服务端返回的字段；缺失字段不会在客户端补全。",
  "detail.peers": "同批结果",

  "board.rank.retrieve": "分数排行",
  "board.rank.ask": "证据强度",
  "board.sub.retrieve": "仅本次响应内排序",
  "board.sub.ask": "官方 3 / 媒体 2 / 元数据 1",
  "board.items": "{n} 条",
  "board.unit": "条",
  "board.sourceMix": "来源构成",
  "board.total": "条数",
  "board.thisResponse": "本次响应",
  "board.publishers": "来源方",
  "board.dedup": "发布方去重",
  "board.span": "时间跨度",
  "board.snapshot": "快照",
  "board.config": "配置",
  "board.mockCorpus": "mock 语料",

  "error.network": "网络错误",
  "error.connect": "无法连接本地服务：{message}",
  "error.badRequest.title": "请求格式有误",
  "error.badRequest.text": "服务端拒绝了请求体；请修正输入后重新提交。",
  "error.notFound.title": "快照不存在",
  "error.notFound.text": "这个 id 从未被本机服务知道；页面不会自动猜测另一个快照。",
  "error.refresh": "刷新快照列表",
  "error.retired.title": "快照已退役",
  "error.retired.text":
    "该快照超出保留范围。页面不会自动切到当前快照；要用新快照，必须由你显式重跑。",
  "error.rerun": "改用当前快照显式重跑",
  "error.invalid.title": "筛选条件无效",
  "error.invalid.text": "服务端返回 422；请检查日期范围或筛选组合后重试。",
  "error.clearDates": "清空日期筛选",
  "error.unavailable.title": "服务暂不可用",
  "error.unavailable.text": "本机还没有可服务的完整快照，或正在安全切换。",
  "error.retryAfter": "Retry-After：{seconds} 秒",
  "error.retry": "重试",
  "error.generic.title": "请求失败",
  "error.generic.text": "无法完成这次请求；结果区没有展示任何不可追溯的内容。",
  "error.unknown": "未知错误",
  "error.requestLine": "请求快照：{snapshot} · 筛选：{filters}",
  "error.summary": "请求失败：HTTP {status}。",

  "refresh.done.title": "快照列表已刷新",
  "refresh.done.text": "选择快照后，再显式重跑；页面不会自动改写原请求。",
  "refresh.done.current": "当前快照：{id}",
  "refresh.unknown": "未知",
};

const DICTIONARIES = { en: EN, zh: ZH };

function readStoredLang() {
  try {
    return window.localStorage.getItem(STORAGE_KEY);
  } catch {
    return null;
  }
}

function initialLang() {
  const fromQuery = new URLSearchParams(window.location.search).get("lang");
  if (LANGS.includes(fromQuery)) {
    return fromQuery;
  }
  const stored = readStoredLang();
  return LANGS.includes(stored) ? stored : DEFAULT_LANG;
}

let lang = initialLang();

/** `{name}` placeholders keep the copy translatable without string concatenation at call sites. */
export function t(key, vars = {}) {
  const template = DICTIONARIES[lang][key] ?? DICTIONARIES[DEFAULT_LANG][key] ?? key;
  return template.replace(/\{(\w+)\}/g, (_, name) =>
    vars[name] === undefined || vars[name] === null ? "" : String(vars[name]),
  );
}

export function getLang() {
  return lang;
}

export function setLang(next) {
  if (!LANGS.includes(next) || next === lang) {
    return lang;
  }
  lang = next;
  try {
    window.localStorage.setItem(STORAGE_KEY, lang);
  } catch {
    // Private mode: the choice lasts for this page view only.
  }
  document.documentElement.lang = lang === "zh" ? "zh-CN" : "en";
  return lang;
}

export function sourceLabel(sourceType) {
  return t(`source.${sourceType}`);
}
