# EVAL_PROTOCOL —— 评测执行标准

> 归属：评估的**执行细则**。设计理由与配置矩阵在 04 §4；本文件只写「怎么标、怎么判、怎么记」。
> 目标：同一份 QA 集 + 同一份语料快照，任何人重跑都得到同一组数字。

## 1. QA 集格式（`eval/qa_set.jsonl`，一行一题）

```json
{
  "qa_id": "A-007",
  "type": "A",
  "question": "《XXX》第二季的放送开始日是？",
  "gold_answer": "2026-10-05",
  "gold_evidence": {
    "chunk_ids": ["c_8f21"],
    "urls": ["https://www.toei-anim.co.jp/news/..."],
    "publisher": "東映アニメーション"
  },
  "key_entities": ["2026-10-05"],
  "source_type": "official",
  "window": ["2026-07-30", "2026-10-28"],
  "annotator": "D",
  "verified_by": "F"
}
```

- `key_entities`：规则匹配命中率的判定依据（实体/数值），必须能在 `gold_answer` 里找到。
- `window`：该题的有效时间窗，超出窗的语料不参与判定。
- 四类配额：A 10–12 / B 8–10 / C 6–8 / D ≥8（含拒答 ≥5），总量 ≥30，**A 类 ≤ 总量 1/3**。

## 2. 三档判定细则（`answer_correctness`）

| 档位 | 判定 | 例子 |
|---|---|---|
| `fully_correct` | 关键实体/数值全对，且无与 `gold_evidence` 矛盾的陈述 | 日期、平台、话数均对 |
| `partially_correct` | 核心正确但缺 1 项，或正确但夹带 1 处未被支持的多余断言 | 日期对、平台漏 |
| `incorrect` | 关键实体错、答非所问、或引用与答案不一致 | 说 10-05 实际 10-12 |

判定纪律：
1. **先看引用，再看答案**。引用不到原文的答案，即便文字对也降一档（可验证性是本题核心）。
2. 两人独立标 10 条对表，一致率 <85% 就回来改细则，不要把分歧留到最后。
3. 争议题由 D 裁决并记进 `eval/runs/<run_id>/disputes.md`。

## 3. 拒答（`abstention_accuracy`）

- 拒答成立的条件：知识库确实无答案，且回答里**明确说明"知识库中没有"**而非猜测。
- 校准集：20 条有答案 + 10 条无答案，扫拒答阈值，画出「误拒 / 漏拒」曲线后取工作点。
- 报告里必须同时给两个数：无答案集拒答率、有答案集误拒率。只报一个数会被质疑。

## 4. run 记录（`eval/runs/<run_id>/meta.json`）

```json
{
  "run_id": "20261115-A-20261028abcdef",
  "snapshot_id": "20261028abcdef",
  "window_start": "2026-07-30",
  "window_end": "2026-10-28",
  "chunk_config": "A",
  "retrieval": {"mode": "vector", "top_k": 5, "over_fetch": 5},
  "embed": {"model": "intfloat/multilingual-e5-base", "device": "cpu"},
  "llm": {"provider": "…", "model": "…", "version": "…", "temperature": 0.0},
  "git_sha": "…",
  "metrics": {"retrieval_hit_rate": 0.0, "answer_correctness": {}, "abstention_accuracy": {}}
}
```

**缺 `snapshot_id` 或 `window` 的 run 记录无效**（02 §3 不变量 2）。

## 5. 复现纪律

- `temperature=0` **不等于确定性**：MoE 路由与批处理仍会带来抖动。报告里必须写明
  「同版本同参数重跑存在极小非确定性」，并给出重跑两次的差异范围。
- 换 `device`（cpu ↔ mps）、换模型版本、换 chunk 配置，都必须产生新 run_id。
- 指标分母一律 `deduped_count`（02 §3 不变量 3）。

## 6. 报告口径（写进 §6 的三条）

1. 总体数字之外，**按 `sourceType` 分组**报告：官方短公告上 A/B 几乎无差异，长文档上才有差异。
2. 结论必须条件化：「细节题 A'/A 优；跨源综合题 B 优」，不要写"A 更好"。
3. 30 条样本的置信区间很宽（±10 个百分点量级），必须附区间或明确说明样本量限制。
