"""Baseline-run management extracted from ``engine.py``.

Handles tag-based, latest, latest_pass, and rolling-window baseline
selection strategies.  Pure I/O + logic with no dependency on the
adapter layer or gate-evaluation code.

Public names (used by ``engine.py``):
    _read_tags, _write_tag, _load_previous_run,
    _average_metrics_dicts, _validate_baseline_mode
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from temporalci.constants import BASELINE_MODES
from temporalci.utils import atomic_write_json, read_json_dict


# ---------------------------------------------------------------------------
# Tag persistence
# ---------------------------------------------------------------------------


def _read_tags(model_root: Path) -> dict[str, str]:
    """Load tag → run_id mapping from model_root/tags.json."""
    data = read_json_dict(model_root / "tags.json")
    return data if data is not None else {}


def _write_tag(model_root: Path, tag: str, run_id: str) -> None:
    """Persist a tag → run_id mapping into model_root/tags.json."""
    tags = _read_tags(model_root)
    tags[str(tag)] = run_id
    try:
        atomic_write_json(model_root / "tags.json", tags)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Metric averaging (used by rolling baseline)
# ---------------------------------------------------------------------------


def _average_metrics_dicts(dicts: list[dict[str, Any]]) -> dict[str, Any]:
    """Recursively average scalar values across a list of metric dicts."""
    if not dicts:
        return {}
    result: dict[str, Any] = {}
    all_keys: set[str] = {k for d in dicts for k in d}
    for k in all_keys:
        values = [d[k] for d in dicts if k in d]
        if all(isinstance(v, dict) for v in values):
            result[k] = _average_metrics_dicts(values)
        elif all(
            isinstance(v, (int, float)) and not isinstance(v, bool) for v in values
        ):
            result[k] = sum(float(v) for v in values) / len(values)
        else:
            result[k] = values[-1]
    return result


# ---------------------------------------------------------------------------
# Baseline-mode validation
# ---------------------------------------------------------------------------


def _validate_baseline_mode(baseline_mode: str) -> None:
    if baseline_mode in BASELINE_MODES:
        return
    if baseline_mode.startswith("tag:") and len(baseline_mode) > 4:
        return
    if baseline_mode.startswith("rolling:"):
        n_str = baseline_mode[8:].strip()
        try:
            if int(n_str) > 0:
                return
        except ValueError:
            pass
        raise ValueError(
            f"rolling baseline requires a positive integer: 'rolling:N', got '{baseline_mode}'"
        )
    available = ", ".join(sorted(BASELINE_MODES))
    raise ValueError(
        f"invalid baseline_mode '{baseline_mode}'. "
        f"choose: {available}, tag:<name>, or rolling:<N>"
    )


# ---------------------------------------------------------------------------
# Baseline run loading
# ---------------------------------------------------------------------------


def _load_previous_run(
    model_root: Path,
    current_run_id: str,
    *,
    baseline_mode: str,
) -> dict[str, Any] | None:
    if baseline_mode == "none":
        return None
    if not model_root.exists():
        return None

    # Tag-based baseline: "tag:<name>"
    if baseline_mode.startswith("tag:"):
        tag_name = baseline_mode[4:].strip()
        if not tag_name:
            raise ValueError("tag name must not be empty in baseline_mode 'tag:<name>'")
        tags = _read_tags(model_root)
        tagged_run_id = tags.get(tag_name)
        if not tagged_run_id:
            return None  # tag not found yet — treat as no baseline
        run_json = model_root / str(tagged_run_id) / "run.json"
        if not run_json.exists():
            return None
        try:
            payload = json.loads(run_json.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else None
        except (json.JSONDecodeError, OSError):
            return None

    candidates: list[tuple[str, dict[str, Any]]] = []
    for child in model_root.iterdir():
        if not child.is_dir() or child.name == current_run_id:
            continue
        run_json = child / "run.json"
        if run_json.exists():
            payload = json.loads(run_json.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                candidates.append((child.name, payload))
    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0], reverse=True)
    if baseline_mode == "latest":
        return candidates[0][1]
    if baseline_mode == "latest_pass":
        for _, payload in candidates:
            if payload.get("status") == "PASS":
                return payload
        return None

    # Rolling-window baseline: average last N passing runs
    if baseline_mode.startswith("rolling:"):
        n = int(baseline_mode[8:].strip())
        pass_runs = [p for _, p in candidates if p.get("status") == "PASS"][:n]
        if not pass_runs:
            return None
        avg_metrics = _average_metrics_dicts([r.get("metrics") or {} for r in pass_runs])
        return {
            "run_id": f"rolling:{n}(n={len(pass_runs)})",
            "status": "PASS",
            "metrics": avg_metrics,
        }

    raise ValueError(
        f"unsupported baseline_mode '{baseline_mode}'. "
        f"supported: {sorted(BASELINE_MODES)}, tag:<name>, or rolling:<N>"
    )
