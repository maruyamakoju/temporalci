from __future__ import annotations

import json
from pathlib import Path

import pytest

from temporalci.prune import prune_model_runs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_model_root(tmp_path: Path, n_runs: int = 5) -> Path:
    """Create a minimal model_root with n_runs run directories."""
    import datetime

    model_root = tmp_path / "model"
    model_root.mkdir()
    for i in range(n_runs):
        ts = datetime.datetime(2026, 2, i + 1, 0, 0, 0)
        run_id = ts.strftime("%Y%m%dT%H%M%SZ")
        run_dir = model_root / run_id
        run_dir.mkdir()
        # Write a small file to give the dir a non-zero size
        (run_dir / "run.json").write_text(json.dumps({"run_id": run_id}), encoding="utf-8")
        with (model_root / "runs.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps({"run_id": run_id, "status": "PASS"}) + "\n")
    return model_root


# ---------------------------------------------------------------------------
# prune_model_runs
# ---------------------------------------------------------------------------


def test_prune_deletes_oldest_runs(tmp_path: Path) -> None:
    model_root = _make_model_root(tmp_path, n_runs=5)
    result = prune_model_runs(model_root, keep_last=3)
    assert result["kept"] == 3
    assert result["deleted"] == 2
    assert result["skipped"] == 0
    # Verify old dirs gone
    entries = [
        line.strip()
        for line in (model_root / "runs.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(entries) == 3


def test_prune_rewrites_jsonl(tmp_path: Path) -> None:
    model_root = _make_model_root(tmp_path, n_runs=4)
    all_ids_before = [
        json.loads(l)["run_id"]
        for l in (model_root / "runs.jsonl").read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    prune_model_runs(model_root, keep_last=2)
    remaining = [
        json.loads(l)["run_id"]
        for l in (model_root / "runs.jsonl").read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    # Most recent 2 are kept
    assert remaining == all_ids_before[-2:]


def test_prune_keeps_all_when_keep_last_exceeds_total(tmp_path: Path) -> None:
    model_root = _make_model_root(tmp_path, n_runs=3)
    result = prune_model_runs(model_root, keep_last=10)
    assert result["deleted"] == 0
    assert result["kept"] == 3


def test_prune_dry_run_does_not_delete(tmp_path: Path) -> None:
    model_root = _make_model_root(tmp_path, n_runs=4)
    dirs_before = sorted(d.name for d in model_root.iterdir() if d.is_dir())
    result = prune_model_runs(model_root, keep_last=2, dry_run=True)
    dirs_after = sorted(d.name for d in model_root.iterdir() if d.is_dir())
    # Nothing deleted
    assert dirs_before == dirs_after
    assert result["deleted"] == 2  # would-be deleted count still reported
    # jsonl not rewritten
    entries = [
        l for l in (model_root / "runs.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()
    ]
    assert len(entries) == 4


def test_prune_returns_bytes_freed(tmp_path: Path) -> None:
    model_root = _make_model_root(tmp_path, n_runs=3)
    result = prune_model_runs(model_root, keep_last=1)
    assert result["bytes_freed"] > 0


def test_prune_empty_jsonl_returns_zero_counts(tmp_path: Path) -> None:
    model_root = tmp_path / "model"
    model_root.mkdir()
    (model_root / "runs.jsonl").write_text("", encoding="utf-8")
    result = prune_model_runs(model_root, keep_last=5)
    assert result == {"kept": 0, "deleted": 0, "skipped": 0, "bytes_freed": 0}


def test_prune_missing_jsonl_returns_zero_counts(tmp_path: Path) -> None:
    model_root = tmp_path / "model"
    model_root.mkdir()
    result = prune_model_runs(model_root, keep_last=5)
    assert result == {"kept": 0, "deleted": 0, "skipped": 0, "bytes_freed": 0}


def test_prune_skips_entry_with_missing_run_dir(tmp_path: Path) -> None:
    model_root = _make_model_root(tmp_path, n_runs=3)
    # Prepend a jsonl entry (oldest position) pointing to a non-existent run dir
    existing = (model_root / "runs.jsonl").read_text(encoding="utf-8")
    phantom_line = json.dumps({"run_id": "phantom_run"}) + "\n"
    (model_root / "runs.jsonl").write_text(phantom_line + existing, encoding="utf-8")
    # keep_last=3 keeps the 3 real runs; phantom_run is the oldest → goes to to_delete
    result = prune_model_runs(model_root, keep_last=3)
    # phantom_run has no dir → skipped (not counted as deleted)
    assert result["skipped"] >= 1


def test_prune_keep_last_zero_raises(tmp_path: Path) -> None:
    model_root = _make_model_root(tmp_path, n_runs=2)
    with pytest.raises(ValueError, match="keep_last"):
        prune_model_runs(model_root, keep_last=0)


def test_prune_actual_dirs_deleted(tmp_path: Path) -> None:
    model_root = _make_model_root(tmp_path, n_runs=5)
    all_run_ids = [
        json.loads(l)["run_id"]
        for l in (model_root / "runs.jsonl").read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    prune_model_runs(model_root, keep_last=2)
    deleted_ids = all_run_ids[:-2]
    kept_ids = all_run_ids[-2:]
    for rid in deleted_ids:
        assert not (model_root / rid).exists()
    for rid in kept_ids:
        assert (model_root / rid).exists()
