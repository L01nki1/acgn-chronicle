# COMPLIANCE —— C1–C6 的实现规格

> 归属：合规的**工程落地**。红线声明在 06 §3，本文件只写「代码里怎么落地、怎么断言」。
> 凡本文件与 06 冲突，以 06 为准并回来改本文件。

## C1 采集只走官方公开渠道

| 规格 | 落地位置 |
|---|---|
| 默认自描述 UA：`AcgnRagBot/0.1 (CS5481 course project; +repo url)` | `config.HttpConfig.user_agent` |
| per-host token bucket，默认 1 req/2.5s + 随机抖动 | `core/ingest/http.py` |
| 抓取前读 `robots.txt` 并缓存 24h；`Disallow` 路径不进 | `core/ingest/http.py` |
| 回填必须带时间窗参数，无窗口请求直接抛 `ValueError` | 各 `backfill(date_range)` |
| 禁止：绕登录、IP 池、伪造搜索引擎 UA、无头浏览器过盾、注册小号 | 代码里不提供这些开关 |

**单测**：`tests/ingest/test_c1_window.py` —— 无窗口请求必须抛错；`Disallow` 路径必须被拒。

## C2 两区存储

| 区 | 内容 | 对外 |
|---|---|---|
| 事实字段区 `data/facts/` | 作品名 / 发售日 / 平台 / 事件类型（带 `chunkId` + `url`） | ✅ 可对外 |
| 原文区 `data/raw/` | 采集到的完整正文 | ❌ 仅内部检索，永不序列化 |

**断言**：任何 `to_dict()` / `model_dump()` 出站路径上都不能出现 `data/raw/` 的正文；由
`core/ground/serialize.py` 统一裁剪，其他模块不得自行拼响应。

## C3 对外输出四件套（≤200 字符）

```
摘要 + 原文 URL + ≤200 字符片段 + 强制出处
```

- 常量：`MAX_QUOTE_CHARS = 200`（`core/ground/serialize.py`，唯一来源）
- **计量单位：Unicode 字符数**（`len(str)`）。理由：日中英混排下按字节计会让日文只有 60 余字，
  三语展示长度差异过大；按字符计最接近"引用一段话"的直觉。
- 截断方式：`text[:199] + "…"`，保证总长 ≤200（省略号计入额度）。
- 空白归一允许（压连续空白），**改词、换语序、删句不允许**——那是改写。
- 双防线：后端 `serialize.py` + 前端渲染前再裁一次（C 负责，`web/static/`）。

**单测**：`tests/compliance/test_serialize_limits.py`（已就位，11 条）。

## C4 `media` 不得转述评论观点

判定口径（写作与 prompt 都用这一条）：

| 允许 | 禁止 |
|---|---|
| 从 `media` 抽取可验证事实（"报道称 X 于 3 月 14 日发售"） | 复述其观点、评价、推测（"该作被评为本季最佳"） |
| 原样引用 ≤200 字符片段 + 出处 | 改写、概括其评论倾向 |

**prompt 条款**（B 负责，写进 `core/ground/` 的 system prompt 并固化为常量）：
> 若来源 `source_type` 为 `media`，只能提取其中可核验的事实与数字，不得转述其评论、评价或推测；
> 若来源为 `official`，可复述事实与官方表述。

**单测**：`tests/compliance/test_media_no_paraphrase.py` —— 对固定 prompt 做快照比对，防止条款被悄悄删掉。

## C5 三类来源必须标注

`SourceType = Literal["official", "media", "metadata"]`，**落库即标注**，不允许事后猜。
前端展示强度按类型区分（official 最强、metadata 最弱）。`DailyDigest.source_coverage` 顺带复用这个字段做采集监控。

## C6 不做训练 / 微调

仓库不得出现训练脚本、`train*.py`、LoRA / fine-tune 配置。CI 不做检查（无法穷举），靠评审把关。

## 底线速记

🚫 展示原文全文　🚫 转述媒体评论　🚫 训练模型　🚫 商业化/广告/付费

## 单测清单（`pytest tests/compliance -q` 必须全绿）

| 用例 | 断言 |
|---|---|
| `test_serialize_limits.py` | ≤200 字符（日中英三语）、省略号计入、空白归一、载荷带 url |
| `test_media_no_paraphrase.py` | prompt 内禁止转述条款存在 |
| `test_no_raw_export.py` | 任一 API 响应序列化结果不含原文全文（待 WP4 落地） |
