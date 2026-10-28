# RUNBOOK —— 命令、故障、SOP

> 归属：运维手册。个人机器上的临时技巧不要写进来，只写「换一台机器照样能用」的东西。

## 1. 命令表

| 目的 | 命令 | 状态 |
|---|---|---|
| 装依赖（不含重包） | `pip install -e ".[dev]"` | ✅ |
| 装编码与向量库 | `pip install -e ".[dev,embed,store]"` | ✅ |
| 契约一致性 | `python scripts/export_schema.py --check` | ✅ |
| 全量单测（零网络） | `pytest -q` | ✅ |
| 需要真网的用例 | `pytest -m network -q` | ✅（仅本地） |
| 历史回填 | `python scripts/backfill.py --since … --until …` | 待 A |
| 每日增量 | `python scripts/discover_daily.py` | 待 A |
| 建索引 | `python scripts/build_index.py --config A` | 待 F |
| 事实抽取 | `python scripts/extract_facts.py --snapshot <id>` | 待 F |
| 评测 | `python scripts/run_eval.py --config A --snapshot <id>` | 待 E |
| 起服务 | `uvicorn acgn_rag.web.api:app --reload` | 待 C |

## 2. 快照重建 SOP

1. 确认 `data/` 已备份（在跑回填之前）。
2. `python scripts/backfill.py --since <start> --until <end>`，看每源 `items > 0`。
3. `python scripts/build_index.py --config A`（三配置逐个来，别并发，CPU 抢满反而更慢）。
4. 校验幂等：重建一次，对比 `snapshot_id` 是否一致。
5. 记录：把 `snapshot_id` + `window` 贴到群里，所有人后续 run 都用这个 id。

> 幂等失败的第一嫌疑永远是**内容清单排序 / 时区 / 时间格式**，不是哈希算法。

## 3. 常见故障

| 症状 | 首查 | 处置 |
|---|---|---|
| 采集条数变 0 | HTML 选择器是否随站点改版失效 | 更新选择器 + 换 fixture；**不要**加 `try: return []` |
| 采集条数骤降 | 该源是否 403/429/分页失效 | 走 01 §5.4 处置阶梯，禁止换 IP |
| 日文正文乱码 | 响应头 charset 与真实编码不一致 | 用 `charset-normalizer` 探测，断言 UTF-8 解码错误率 |
| `top_k=5` 只回 2 条 | Chroma where-filter 裁掉了候选 | 提高 over-fetch（`ACGN_OVERFETCH`），并断言返回数 |
| 同语料两个 `snapshot_id` | CRLF / 排序 / 时区 | 见 §2 注 |
| 索引重建后 chunk 数变了 | 切块参数被改 / tokenizer 不同源 | 检查 `chunk/variants.py` 参数与 tokenizer |
| 模型编造 `chunk_id` | prompt 给了 id 而不是序号 | 改序号映射，断言 `cited ⊆ retrieved` |
| CI 报 `NetworkBlocked` | 测试里写了真实 HTTP | 用 `respx` 拦截；确需真网标 `@pytest.mark.network` |
| macOS 本地过、CI 红 | 文件名大小写 | 模块名全小写 |
| Windows 本地过、CI 红 | CRLF 或路径拼接 | `.gitattributes` 已强制 LF；路径用 `pathlib` |

## 4. 紧急降级

- **语料不足 8,000 篇**：W3 前决策，A/B 对比降级为「定性 + 小样本定量」，报告如实说明（04 R6）。
- **LLM API 不可用**：`/ask` 走 stub 回答（带 `abstained=true` + 说明），保证网站与引用跳转可演示。
- **演示现场断网**：用录屏兜底；本地索引与前端可离线跑。
