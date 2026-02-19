"""Shared utility functions used across TemporalCI modules.

This module consolidates small helpers that were previously duplicated in
``config``, ``prompt_sources``, ``engine``, ``autopilot_utils``, and the
metric backends.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------

def utc_now() -> datetime:
    """Return the current UTC time as an aware :class:`datetime`."""
    return datetime.now(timezone.utc)


def utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return utc_now().isoformat()


# ---------------------------------------------------------------------------
# Numeric / type coercion helpers
# ---------------------------------------------------------------------------

_TRUE_VALUES = {"1", "true", "yes", "y", "on"}
_FALSE_VALUES = {"0", "false", "no", "n", "off"}


def is_number(value: Any) -> bool:
    """Return ``True`` if *value* is an ``int`` or ``float`` (excluding ``bool``)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp *value* into ``[lo, hi]``."""
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


def as_bool(value: Any, *, default: bool) -> bool:
    """Coerce *value* to ``bool`` with common string truthy/falsy parsing."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _TRUE_VALUES:
            return True
        if normalized in _FALSE_VALUES:
            return False
        return default
    if is_number(value):
        return bool(value)
    return default


def as_int(value: Any, *, default: int, minimum: int) -> int:
    """Coerce *value* to ``int`` with a *minimum* floor."""
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    if parsed < minimum:
        return minimum
    return parsed


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def resolve_path(raw_path: str, *, suite_dir: Path) -> Path:
    """Resolve *raw_path* relative to *suite_dir*, falling back to cwd.

    If the path is absolute it is returned as-is.  Otherwise the function
    checks ``suite_dir / path`` first, then ``cwd / path``, and defaults to
    the suite-relative candidate when neither exists on disk.
    """
    path = Path(raw_path.strip())
    if path.is_absolute():
        return path

    suite_candidate = (suite_dir / path).resolve()
    cwd_candidate = (Path.cwd() / path).resolve()
    if suite_candidate.exists():
        return suite_candidate
    if cwd_candidate.exists():
        return cwd_candidate
    return suite_candidate


# ---------------------------------------------------------------------------
# Prompt normalization / deduplication
# ---------------------------------------------------------------------------

def normalize_prompt(text: str) -> str:
    """Collapse whitespace and lowercase *text* for comparison."""
    return " ".join(text.strip().lower().split())


def dedupe_prompts(prompts: list[str]) -> list[str]:
    """Return *prompts* with duplicates removed (preserves first occurrence).

    Comparison is whitespace-normalized via :func:`normalize_prompt`.
    """
    unique: list[str] = []
    seen: set[str] = set()
    for prompt in prompts:
        key = normalize_prompt(prompt)
        if key in seen:
            continue
        seen.add(key)
        unique.append(prompt)
    return unique


# ---------------------------------------------------------------------------
# JSON I/O
# ---------------------------------------------------------------------------

def read_json_dict(path: Path) -> dict[str, Any] | None:
    """Read *path* as JSON and return the top-level dict, or ``None`` on error."""
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    return payload if isinstance(payload, dict) else None


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write *payload* as pretty-printed JSON atomically via a temp file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temp.replace(path)


def safe_write_json(path: Path, payload: dict[str, Any]) -> bool:
    """Like :func:`atomic_write_json` but returns ``False`` on error."""
    try:
        atomic_write_json(path, payload)
        return True
    except Exception:  # noqa: BLE001
        return False


# ---------------------------------------------------------------------------
# Dotted-path traversal
# ---------------------------------------------------------------------------


def resolve_dotted_path(obj: Any, dotted_path: str) -> Any:
    """Walk *obj* along a dotted key path (e.g. ``'a.b.c'``).

    Raises :class:`KeyError` with the full path when any segment is missing.
    """
    current: Any = obj
    for part in dotted_path.split("."):
        if not isinstance(current, dict) or part not in current:
            raise KeyError(dotted_path)
        current = current[part]
    return current


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------


def sample_std(values: list[float]) -> float:
    """Sample standard deviation of *values*.

    Returns ``0.0`` when fewer than 2 values are provided.
    """
    if len(values) < 2:
        return 0.0
    avg = mean(values)
    variance = sum((v - avg) ** 2 for v in values) / max(1, len(values) - 1)
    return math.sqrt(max(0.0, variance))
