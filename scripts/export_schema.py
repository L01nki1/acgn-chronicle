"""命令行包装：导出 / 校验 KnowledgePort JSON Schema。

用法：
    python scripts/export_schema.py            # 写入 contracts/knowledge_port.schema.json
    python scripts/export_schema.py --check    # 只校验，不一致退出码 1（CI 用这个）

真正的逻辑在 acgn_rag/contracts_export.py，本文件只负责退出码。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from acgn_rag.contracts_export import check, write


def main() -> int:
    parser = argparse.ArgumentParser(description="导出 / 校验 KnowledgePort JSON Schema")
    parser.add_argument("--check", action="store_true", help="仅校验，不写文件")
    args = parser.parse_args()

    if args.check:
        ok, message = check()
        print(message)
        return 0 if ok else 1

    path = write()
    print(f"WROTE: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
