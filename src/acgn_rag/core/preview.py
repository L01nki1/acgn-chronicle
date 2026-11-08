"""Fresh, isolated output directories for deliberately limited runs."""

from __future__ import annotations

import ntpath
import re
from pathlib import Path

_PREVIEW_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")


def validate_preview_options(*, preview: bool, preview_id: str | None, limited: bool) -> None:
    """Require a fresh preview ID for runs that would write a limited subset."""
    if preview != (preview_id is not None):
        raise ValueError("--preview and --preview-id must be supplied together")
    if limited and not preview:
        raise ValueError("limited output requires --preview and --preview-id")
    if preview_id is not None and (
        not _PREVIEW_ID.fullmatch(preview_id) or ntpath.isreserved(preview_id)
    ):
        raise ValueError("invalid preview ID; use a safe 1-64 character name")


def validate_snapshot_component(snapshot_id: str) -> None:
    """Keep a snapshot name within one path component on Linux and Windows."""
    if (
        not snapshot_id
        or snapshot_id in {".", ".."}
        or any(char in snapshot_id for char in ("/", "\\", ":", "\0"))
        or ntpath.isreserved(snapshot_id)
    ):
        raise ValueError("invalid snapshot name: expected a single safe path component")


def create_preview_root(data_root: Path, preview_id: str) -> Path:
    """Create a new run directory; never reuse a preview or follow a redirected parent."""
    validate_preview_options(preview=True, preview_id=preview_id, limited=False)
    root = Path(data_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    parent = root / "previews"
    parent.mkdir(exist_ok=True)
    if parent.is_symlink() or parent.is_junction() or not parent.is_dir():
        raise ValueError("preview parent must be a real directory")
    preview_root = parent / preview_id
    try:
        preview_root.mkdir()
    except FileExistsError as exc:
        raise ValueError(f"preview ID already exists: {preview_id!r}") from exc
    return preview_root
