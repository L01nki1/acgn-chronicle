# ONBOARDING —— 新人第一天看这一份

> 归属：工程协作细则（06 §1 是规则声明，本文件是可执行的清单）。
> 规则变了先改这里，再在 PR 里 @ 全员。

## 1. 环境（组内 5 人 macOS，1 人 Windows）

| 事项 | macOS（多数） | Windows（1 人） |
|---|---|---|
| Python | 3.11（`brew install python@3.11` 或 `uv python install 3.11`） | 3.11（官方安装包，勾选 Add to PATH） |
| Shell | zsh（系统 bash 是 3.2，别写 bash 4+ 语法） | Git Bash（不是 cmd / PowerShell） |
| 虚拟环境 | `source .venv/bin/activate` | `.venv\Scripts\activate` |
| torch 轮子 | arm64 原生，安装正常 | 正常 |

**不要做的事**：不要提交 `.sh` / `.ps1` / `.bat` 脚本当入口。所有可重复动作走
`python scripts/xxx.py` 或 `python -m acgn_rag...`，两端命令完全一致。

## 2. 分支与提交（06 §1）

- `main` 只进 PR，不直接推。
- 每个 WP 一条长期分支：`wp1-ingest` / `wp2-kb` / `wp3-retrieve` / `wp4-ground` / `wp5-eval` / `wp6-web`。
  个人在其上开短命分支：`wp1-ingest/a-toei-adapter`。
- 提交信息：`[WP1] add Toei html list adapter`；涉及契约变更加前缀 `[contract]`。
- 契约变更（`ports.py`）必须在 PR 描述里说明理由，并重新导出 schema（见 §4）。

## 3. PR 检查清单（评审人逐条打勾）

1. CI 绿（lint / format / 契约校验 / 测试）
2. 没有把 `data/`、`.env`、模型文件带进提交（`git status` 自查）
3. 新增目录符合 02 §2 的归属，跨 WP 改动在描述里 @ 了对应负责人
4. 触及不变量（02 §3）的改动，有对应单测
5. 评审人是该 WP 的交叉评审人，而不是自己

## 4. CI 不联网 —— 精确含义与写法

**含义**：CI 只在「装依赖」时联网；跑测试时**不允许任何外部请求**。

| 允许 | 禁止 |
|---|---|
| `pip install -e ".[dev]"` 从 PyPI 装包 | 测试里 `httpx.get("https://...")` 打真站 |
| 通过 fixture 读取仓库内样本页 | 测试里下载 e5 模型 / 调 LLM API |
| — | 依赖网络才能通过的"集成测试" |

**为什么**：外部站点的可用性、限速、日本站点的 403，都会让 CI 假红；而假红会让人开始无视 CI。

**怎么写**：
- HTTP 一律用 `respx` 拦截 `httpx`，返回 `tests/ingest/fixtures/` 里的样本。
- Embedding 一律用 `fake_embedder` fixture（`tests/conftest.py`），返回确定性假向量。
- 确实需要真网的验证，标 `@pytest.mark.network`，**只在本地手动跑**：
  `pytest -m network -q`。CI 的 `addopts` 里已经有 `-m 'not network'`。
- `tests/conftest.py` 会拦截 socket，违反规则会直接报 `NetworkBlocked` —— 这是故意设计的。

**采集脚本例外**：`scripts/backfill.py` / `discover_daily.py` 天生要联网，但它们不进 CI，
也不写单测（选择器逻辑用 fixture 离线测，见 01 §5.3）。

## 5. 跨平台差异（macOS ↔ Windows，会影响结果的）

| 差异 | 后果 | 处理 |
|---|---|---|
| 文件系统大小写：mac 默认不敏感，Linux/CI 敏感 | `RawItem.py` 与 `rawitem.py` 在 mac 上不报错，CI 红 | 模块名全小写；case bug 交给 ubuntu job 抓 |
| 换行：Windows CRLF | **污染快照内容哈希**，同语料算出不同 `snapshotId` | `.gitattributes` 强制 LF；读文件统一 `encoding="utf-8"` + `newline=""` |
| 路径分隔符 | 拼接错 | 一律 `pathlib.Path`，禁止字符串拼 `"/"` 或 `"\\"` |
| 本地编码：mac 默认 UTF-8，Windows 默认 cp936 | 读样本页乱码 | 显式 `encoding=`；日文页用 `charset-normalizer` 探测 |
| Unicode 归一化：mac 文件名 NFD | 日文作品名做文件名会分裂 | **文件名一律用 media_id / 哈希，不用作品名** |
| 时区 | 快照窗口错位 | 内部一律 UTC 存储，展示层再转 JST/CST |
| Embedding device：mac 有 MPS | 不同 device 的浮点结果不完全一致 | 评测与建索引固定 `ACGN_EMBED_DEVICE=cpu`；换 device 必须记进 run 记录 |

## 6. 分工（2026-10-28 修订，替代 03 §2 中的对应两行）

| 成员 | 主责 | 兼责 | 评审 |
|---|---|---|---|
| A | WP1 采集与快照 | WP2 实体抽取协作 | WP2 |
| B | WP4 生成/引用/**契约作者** | WP1 采集器轮子；契约变更仲裁 | WP6 |
| C | WP6 网站与演示 | WP4 合规裁剪（前端最后防线）+ WP5 可视化 | WP4 |
| D | WP5 评估 + PM | WP6 演示脚本 / QA 组织 / 贡献记录 | 与 F 双向 |
| E | WP3 检索 | WP5 runner 工程化；**指标定义与评分规则**（原 F）；WP1 快照评审 | WP1 |
| F | WP2 知识库与事实抽取 | **Isla 对接**（原 B）；事实抽取 prompt | WP3 |

**这次改动只动两行**，理由与边界：
1. **Isla 对接从 B 移到 F**：F 是知识库与 `DailyDigest` 的产出方，最清楚检索结果长什么样；
   B 仍是 `ports.py` 唯一作者，形成「F 对外、B 对内」的双人结构，满足 bus factor ≥2。
2. **指标定义从 F 移到 E**：E 本来就是 runner 的实现者，指标规则与 runner 挨着，避免两人对齐成本；
   F 腾出的带宽正好覆盖 Isla 对接。

**F 的阶段一边界（重要）**：04 §1.1 定的是「本阶段不做 Isla 集成，只冻结契约」。
所以 F 在 11/24 之前**只做三件事**，不碰 Isla 仓库代码（02 首段：不碰 Isla 仓库）：

- 读 `contracts/KnowledgePort.md` + schema，从消费者角度提问题（W1 末，依赖 B 的 G1a 产出）
- 写一份对接说明 + 一个 curl 示例（放 `contracts/`，进 git）
- 12 月之后的合并动作（`context.ts` 第三个桶、`provenance.ts` 第四种来源）才动 TS 代码

## 7. 节奏

- 每日：谁改了契约 / 谁动了不变量相关代码，在群里一句话同步。
- 每周一次：D 记贡献表；E 跑一次 `run_eval.py` 留档。
- 遇阻 30 分钟没进展就发群里，不要一个人啃——尤其采集侧的选择器与限速。
