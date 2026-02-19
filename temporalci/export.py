"""Run history export for TemporalCI.

Exports metric history from ``runs.jsonl`` + ``run.json`` files to CSV or JSONL
for downstream analysis (pandas, Excel, BI tools, etc.).

Public API
----------
export_runs(model_root, output, *, last_n, fmt)  ->  int   (rows written)
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    entries: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
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


def _load_run_json(model_root: Path, run_id: str) -> dict[str, Any] | None:
    run_json = model_root / run_id / "run.json"
    if not run_json.exists():
        return None
    try:
        data = json.loads(run_json.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (json.JSONDecodeError, OSError):
        return None


def _scalar_metric_paths(metrics: dict[str, Any]) -> list[str]:
    """Return all dotted paths to finite scalar values in *metrics*."""
    paths: list[str] = []

    def _walk(obj: Any, prefix: str) -> None:
        if not isinstance(obj, dict):
            return
        for k, v in obj.items():
            path = f"{prefix}.{k}" if prefix else k
            if (
                isinstance(v, (int, float))
                and not isinstance(v, bool)
                and math.isfinite(float(v))
            ):
                paths.append(path)
            elif isinstance(v, dict):
                _walk(v, path)

    _walk(metrics, "")
    return paths


def _resolve_dotted(obj: Any, path: str) -> Any:
    current = obj
    for part in path.split("."):
        if not isinstance(current, dict):
            raise KeyError(path)
        current = current[part]
    return current


# ---------------------------------------------------------------------------
# Format writers
# ---------------------------------------------------------------------------


def _export_csv(runs: list[dict[str, Any]], output: Path) -> int:
    # Discover all metric paths across runs (ordered, deduped, stable)
    seen: set[str] = set()
    all_metric_paths: list[str] = []
    for run in runs:
        for path in _scalar_metric_paths(run.get("metrics") or {}):
            if path not in seen:
                seen.add(path)
                all_metric_paths.append(path)

    # Include model_name if any run carries it (e.g. suite-level export)
    has_model_name = any("model_name" in r for r in runs)
    base_fields = (
        ["model_name"] if has_model_name else []
    ) + ["run_id", "timestamp_utc", "status", "sample_count",
         "gate_failed", "regression_failed", "baseline_run_id"]
    fieldnames = base_fields + all_metric_paths

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for run in runs:
            row: dict[str, Any] = {
                "run_id": run.get("run_id", ""),
                "timestamp_utc": str(run.get("timestamp_utc", ""))[:19],
                "status": run.get("status", ""),
                "sample_count": run.get("sample_count", ""),
                "gate_failed": run.get("gate_failed", ""),
                "regression_failed": run.get("regression_failed", ""),
                "baseline_run_id": run.get("baseline_run_id", ""),
            }
            if has_model_name:
                row["model_name"] = run.get("model_name", "")
            metrics = run.get("metrics") or {}
            for path in all_metric_paths:
                try:
                    val = _resolve_dotted(metrics, path)
                    row[path] = (
                        val
                        if isinstance(val, (int, float)) and not isinstance(val, bool)
                        else ""
                    )
                except (KeyError, TypeError):
                    row[path] = ""
            writer.writerow(row)
    return len(runs)


# Keys excluded from JSONL export to keep file lean
_JSONL_SKIP_KEYS = {"samples", "artifacts_policy"}
# Also strip nested llr_history from gate entries (can be very large)


def _strip_gate(gate: dict[str, Any]) -> dict[str, Any]:
    sprt = gate.get("sprt")
    if not isinstance(sprt, dict):
        return gate
    lean_sprt = {k: v for k, v in sprt.items() if k != "llr_history"}
    return {**gate, "sprt": lean_sprt}


def _export_jsonl(runs: list[dict[str, Any]], output: Path) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fh:
        for run in runs:
            lean: dict[str, Any] = {}
            for k, v in run.items():
                if k in _JSONL_SKIP_KEYS:
                    continue
                if k == "gates" and isinstance(v, list):
                    lean[k] = [_strip_gate(g) if isinstance(g, dict) else g for g in v]
                else:
                    lean[k] = v
            fh.write(json.dumps(lean, default=str) + "\n")
    return len(runs)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def export_runs(
    model_root: Path,
    output: Path,
    *,
    last_n: int = 0,
    fmt: str = "csv",
) -> int:
    """Export run history from *model_root* to *output*.

    Parameters
    ----------
    model_root:
        Model artifact directory containing ``runs.jsonl``.
    output:
        Destination file path (created / overwritten).
    last_n:
        Number of most recent runs to export.  ``0`` means all runs.
    fmt:
        ``"csv"`` (default) or ``"jsonl"``.

    Returns
    -------
    int
        Number of rows written.
    """
    if fmt not in {"csv", "jsonl"}:
        raise ValueError(f"unsupported export format: {fmt!r}. choose 'csv' or 'jsonl'")

    entries = _read_jsonl(model_root / "runs.jsonl")
    if last_n > 0:
        entries = entries[-last_n:]

    runs: list[dict[str, Any]] = []
    for entry in entries:
        run_id = str(entry.get("run_id", "")).strip()
        if not run_id:
            continue
        full = _load_run_json(model_root, run_id)
        # Fall back to index-entry data when run.json is missing (e.g. after prune)
        runs.append(full if full is not None else dict(entry))

    if not runs:
        return 0

    if fmt == "jsonl":
        return _export_jsonl(runs, output)
    return _export_csv(runs, output)


def export_suite_runs(
    suite_root: Path,
    output: Path,
    *,
    last_n: int = 0,
    fmt: str = "csv",
) -> int:
    """Export run history for **all models** under *suite_root* into one file.

    A ``model_name`` column (CSV) or key (JSONL) is prepended to each row.
    Models with no run history are silently skipped.

    Returns the total number of rows written across all models.
    """
    from temporalci.index import discover_models

    if fmt not in {"csv", "jsonl"}:
        raise ValueError(f"unsupported export format: {fmt!r}. choose 'csv' or 'jsonl'")

    models = discover_models(suite_root)
    if not models:
        return 0

    merged: list[dict[str, Any]] = []
    for model_name, model_root in models:
        entries = _read_jsonl(model_root / "runs.jsonl")
        if last_n > 0:
            entries = entries[-last_n:]
        for entry in entries:
            run_id = str(entry.get("run_id", "")).strip()
            if not run_id:
                continue
            full = _load_run_json(model_root, run_id)
            run = full if full is not None else dict(entry)
            # Ensure model_name is set (index entries may omit it)
            run.setdefault("model_name", model_name)
            merged.append(run)

    if not merged:
        return 0

    if fmt == "jsonl":
        return _export_jsonl(merged, output)
    return _export_csv(merged, output)
