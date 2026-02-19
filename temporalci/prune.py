"""Run directory pruning for TemporalCI artifact management.

Public API
----------
prune_model_runs(model_root, keep_last, dry_run=False)  ->  dict
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any


def _read_jsonl(runs_jsonl: Path) -> list[dict[str, Any]]:
    if not runs_jsonl.exists():
        return []
    entries: list[dict[str, Any]] = []
    try:
        for line in runs_jsonl.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    except OSError:
        pass
    return entries


def _dir_size(path: Path) -> int:
    """Return total byte size of all files under *path*."""
    total = 0
    try:
        for f in path.rglob("*"):
            if f.is_file():
                try:
                    total += f.stat().st_size
                except OSError:
                    pass
    except OSError:
        pass
    return total


def prune_model_runs(
    model_root: Path,
    *,
    keep_last: int,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Delete old run directories in *model_root*, keeping the *keep_last* most recent.

    Rewrites ``runs.jsonl`` to remove entries for deleted runs (unless ``dry_run``).

    Returns a summary dict::

        {
            "kept":        int,   # runs kept
            "deleted":     int,   # run dirs deleted
            "skipped":     int,   # entries with no corresponding directory
            "bytes_freed": int,   # bytes reclaimed (0 when dry_run)
        }
    """
    if keep_last < 1:
        raise ValueError("keep_last must be >= 1")

    entries = _read_jsonl(model_root / "runs.jsonl")
    if not entries:
        return {"kept": 0, "deleted": 0, "skipped": 0, "bytes_freed": 0}

    # Entries are chronological (oldest first); keep the last N.
    to_keep = entries[-keep_last:]
    to_delete = entries[: len(entries) - keep_last]

    deleted = 0
    skipped = 0
    bytes_freed = 0

    for entry in to_delete:
        run_id = str(entry.get("run_id", "")).strip()
        if not run_id:
            skipped += 1
            continue
        run_dir = model_root / run_id
        if not run_dir.exists() or not run_dir.is_dir():
            skipped += 1
            continue
        size = _dir_size(run_dir)
        if not dry_run:
            shutil.rmtree(run_dir, ignore_errors=True)
        bytes_freed += size
        deleted += 1

    # Rewrite runs.jsonl with only the retained entries.
    if not dry_run and to_delete:
        jsonl_path = model_root / "runs.jsonl"
        if jsonl_path.exists():
            lines = [json.dumps(e) for e in to_keep]
            jsonl_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    return {
        "kept": len(to_keep),
        "deleted": deleted,
        "skipped": skipped,
        "bytes_freed": bytes_freed,
    }
