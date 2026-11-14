"""Fail-closed marker for an in-place index build that has not completed."""

from __future__ import annotations

from pathlib import Path
from stat import S_ISREG

_INCOMPLETE = "build.incomplete"


def incomplete_marker(index_dir: Path, snapshot_id: str, label: str) -> Path:
    return Path(index_dir) / snapshot_id / label / _INCOMPLETE


def assert_no_incomplete_build(index_dir: Path, snapshot_id: str, label: str) -> None:
    """Reject a pending build; absence of the marker does not validate its sidecars."""
    marker = incomplete_marker(index_dir, snapshot_id, label)
    try:
        marker.lstat()
    except FileNotFoundError:
        return
    except OSError as exc:
        raise ValueError(f"cannot inspect incomplete build marker {marker}: {exc}") from exc
    raise ValueError(f"incomplete build for {snapshot_id!r}/{label!r}: {marker}")


def begin_build(index_dir: Path, snapshot_id: str, label: str, *, recover: bool) -> Path:
    """Mark a build before mutation; recovery assumes the earlier writer has stopped."""
    marker = incomplete_marker(index_dir, snapshot_id, label)
    marker.parent.mkdir(parents=True, exist_ok=True)
    if recover:
        try:
            mode = marker.lstat().st_mode
        except FileNotFoundError:
            pass
        else:
            if not S_ISREG(mode):
                raise ValueError(f"invalid incomplete build marker {marker}")
            return marker
    try:
        with marker.open("x", encoding="utf-8") as handle:
            handle.write("index build in progress or interrupted\n")
    except FileExistsError as exc:
        raise ValueError(
            f"incomplete build for {snapshot_id!r}/{label!r}; "
            "after confirming no writer is active, use --fresh --recover-incomplete"
        ) from exc
    return marker
