"""Single command-line entry point (registered as `acgn` in pyproject).

Owner: D | Spec: 02 §1 / docs/RUNBOOK.md

Rule: repeatable actions go through `python -m` or this entry point, never a shell script -
the team runs macOS (zsh, system bash 3.2) and Windows (Git Bash), and both must share one command set.
"""

from __future__ import annotations

import typer

from acgn_rag import __version__
from acgn_rag.contracts_export import check, write

app = typer.Typer(add_completion=False, help="acgn-rag command line interface")


@app.command()
def version() -> None:
    """Print the current package version."""
    typer.echo(f"acgn-rag {__version__}")


@app.command()
def schema(
    check_only: bool = typer.Option(False, "--check", help="verify only, do not write"),
) -> None:
    """Export or verify the KnowledgePort JSON Schema (a CI gate)."""
    if check_only:
        ok, message = check()
        typer.echo(message)
        raise typer.Exit(code=0 if ok else 1)
    typer.echo(f"WROTE: {write().name}")


if __name__ == "__main__":  # pragma: no cover
    app()
