"""统一命令行入口（pyproject 里注册为 `acgn`）。

负责人：D　|　依据：02 §1 / docs/RUNBOOK.md

原则：所有可重复动作走 `python -m` 或本入口，不写 shell 脚本 ——
组内 macOS（zsh + 系统 bash 3.2）与 Windows（Git Bash）共用同一套命令。
"""

from __future__ import annotations

import typer

from acgn_rag import __version__
from acgn_rag.contracts_export import check, write

app = typer.Typer(add_completion=False, help="acgn-rag 命令行入口")


@app.command()
def version() -> None:
    """打印当前包版本。"""
    typer.echo(f"acgn-rag {__version__}")


@app.command()
def schema(check_only: bool = typer.Option(False, "--check", help="只校验不写文件")) -> None:
    """导出 / 校验 KnowledgePort JSON Schema（CI 必过项）。"""
    if check_only:
        ok, message = check()
        typer.echo(message)
        raise typer.Exit(code=0 if ok else 1)
    typer.echo(f"WROTE: {write().name}")


if __name__ == "__main__":  # pragma: no cover
    app()
