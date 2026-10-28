# acgn-rag

CS5481 Data Engineering · Topic 5 —— **带可验证引用的 ACGN 跨时间档案问答 RAG 系统**。

一句话边界：核心产物是 `src/acgn_rag/core/`；网站、日报、Isla 都只是 KnowledgePort 的消费者。
模型训练 / 微调**永不做**（06 C6）。

- 提交：2026-11-24 23:59（Canvas），6 页正文 + 独立签名页 + 附 slides + 15 分钟演示
- 规划与协作的唯一真相在资料库 `GroupProjctMartial`（00–07 八份），本仓库只放**工程制品**
- 工作分解与验收闸门：[`PLAN.md`](PLAN.md)（组内文档，默认不进提交）

## 目录

```
src/acgn_rag/
├── core/          ★ 唯一核心产物
│   ├── ports.py   # 契约唯一真相（Pydantic）→ contracts/*.json 由脚本导出
│   ├── ingest/    # WP1 · A：四类 adapter / 归一化 / 去重聚类 / 快照
│   ├── facts/     # WP2 · F：L0 事实字段抽取
│   ├── chunk/     # WP2 · F：A(512/64) / A'(256/0) / B(事件聚合)
│   ├── embed/     # WP2 · F：multilingual-e5-base + 缓存
│   ├── store/     # WP2 · F：事实字段区 / 原文区（两区）
│   ├── retrieve/  # WP3 · E：纯向量 + filters；stretch BM25+RRF
│   ├── ground/    # WP4 · B：grounded 生成 / 拒答 / 合规裁剪
│   └── eval/      # WP5 · D+E：QA 集 / 三指标 / runner / 报告表
├── daily/         # WP6 · C：日报 consumer（后置）
└── web/           # WP6 · C：FastAPI + web/static
scripts/           # 可重复执行的入口（backfill / build_index / run_eval / export_schema …）
contracts/         # 跨语言契约：knowledge_port.schema.json + KnowledgePort.md
eval/              # 进 git：qa_set.jsonl + runs/（小文件、可复现的证据）
data/              # 永不进 git：raw / facts / index / snapshots
tests/             # 含 tests/compliance/ 合规单测（原文不出库、≤200 字符、media 不转述）
docs/              # 工程文档：RUNBOOK / DECISIONS / EVAL_PROTOCOL / COMPLIANCE / ONBOARDING
```

## 快速开始

需要 Python 3.11+（CI 固定 3.11；3.13 未验证 torch 轮子，不建议本地用）。

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"            # 只装开发依赖：不下载模型、不装 torch
cp .env.example .env               # 需要跑 LLM 时再填 key
```

> `[embed]`（sentence-transformers + torch）与 `[store]`（chromadb）是重依赖，按需单独装：
> `pip install -e ".[dev,embed,store]"`。CI 不装它们。

## 三条命令（骨架已就绪，实现随各 WP 落地）

```bash
python scripts/export_schema.py --check     # 契约一致性（CI 必过）
pytest -q                                   # 全部单测，默认零网络
ruff check . && ruff format --check .       # 风格
```

管线落地后追加：

```bash
python scripts/backfill.py --since 2026-07-30 --until 2026-10-28   # WP1 回填
python scripts/build_index.py --config A                           # WP2 建索引
python scripts/run_eval.py --config A --snapshot <snapshot_id>     # WP5 评测
```

## 两条硬规则

1. **测试不联网。** `tests/conftest.py` 会拦截 socket；需要真实网络的用例标 `@pytest.mark.network`，默认不跑。
   CI 只允许装依赖时联网，测试期间零外部请求（不打真站、不下载模型、不调 LLM API）。
2. **契约改动必须重导出。** 改 `ports.py` 而不跑 `python scripts/export_schema.py`，CI 直接红。

详见 [`docs/ONBOARDING.md`](docs/ONBOARDING.md)（macOS / Windows 差异与提交规范）与 [`docs/COMPLIANCE.md`](docs/COMPLIANCE.md)（C1–C6 实现规格）。
