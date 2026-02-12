from __future__ import annotations

import json
from pathlib import Path

from temporalci.utils import (
    as_bool,
    as_int,
    atomic_write_json,
    clamp,
    dedupe_prompts,
    is_number,
    normalize_prompt,
    read_json_dict,
    resolve_path,
    safe_write_json,
    utc_now,
    utc_now_iso,
)


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------


def test_utc_now_returns_aware_datetime() -> None:
    dt = utc_now()
    assert dt.tzinfo is not None


def test_utc_now_iso_roundtrips() -> None:
    iso = utc_now_iso()
    assert "T" in iso
    assert iso.endswith("+00:00")


# ---------------------------------------------------------------------------
# is_number
# ---------------------------------------------------------------------------


def test_is_number_accepts_int_and_float() -> None:
    assert is_number(0) is True
    assert is_number(3.14) is True
    assert is_number(-1) is True


def test_is_number_rejects_bool_and_str() -> None:
    assert is_number(True) is False
    assert is_number(False) is False
    assert is_number("42") is False
    assert is_number(None) is False


# ---------------------------------------------------------------------------
# clamp
# ---------------------------------------------------------------------------


def test_clamp_within_bounds() -> None:
    assert clamp(0.5) == 0.5


def test_clamp_below_lo() -> None:
    assert clamp(-1.0) == 0.0
    assert clamp(-5.0, lo=2.0, hi=10.0) == 2.0


def test_clamp_above_hi() -> None:
    assert clamp(2.0) == 1.0
    assert clamp(15.0, lo=2.0, hi=10.0) == 10.0


# ---------------------------------------------------------------------------
# as_bool
# ---------------------------------------------------------------------------


def test_as_bool_with_actual_bool() -> None:
    assert as_bool(True, default=False) is True
    assert as_bool(False, default=True) is False


def test_as_bool_with_truthy_strings() -> None:
    for val in ("1", "true", "True", "YES", "y", "on", "ON"):
        assert as_bool(val, default=False) is True


def test_as_bool_with_falsy_strings() -> None:
    for val in ("0", "false", "False", "NO", "n", "off", "OFF"):
        assert as_bool(val, default=True) is False


def test_as_bool_with_unrecognized_string_returns_default() -> None:
    assert as_bool("maybe", default=True) is True
    assert as_bool("maybe", default=False) is False


def test_as_bool_with_number() -> None:
    assert as_bool(1, default=False) is True
    assert as_bool(0, default=True) is False
    assert as_bool(3.5, default=False) is True


def test_as_bool_with_none_returns_default() -> None:
    assert as_bool(None, default=True) is True
    assert as_bool(None, default=False) is False


# ---------------------------------------------------------------------------
# as_int
# ---------------------------------------------------------------------------


def test_as_int_valid() -> None:
    assert as_int("10", default=5, minimum=0) == 10
    assert as_int(42, default=5, minimum=0) == 42


def test_as_int_below_minimum() -> None:
    assert as_int(-5, default=10, minimum=0) == 0
    assert as_int(2, default=10, minimum=5) == 5


def test_as_int_invalid_returns_default() -> None:
    assert as_int("abc", default=7, minimum=0) == 7
    assert as_int(None, default=7, minimum=0) == 7


# ---------------------------------------------------------------------------
# resolve_path
# ---------------------------------------------------------------------------


def test_resolve_path_absolute_passthrough(tmp_path: Path) -> None:
    target = tmp_path / "file.txt"
    target.write_text("data", encoding="utf-8")
    result = resolve_path(str(target), suite_dir=tmp_path / "suite")
    assert result == target


def test_resolve_path_relative_prefers_suite_dir(tmp_path: Path) -> None:
    suite_dir = tmp_path / "suite"
    suite_dir.mkdir()
    (suite_dir / "data.txt").write_text("from suite", encoding="utf-8")
    result = resolve_path("data.txt", suite_dir=suite_dir)
    assert result == (suite_dir / "data.txt").resolve()


def test_resolve_path_fallback_when_nothing_exists(tmp_path: Path) -> None:
    suite_dir = tmp_path / "suite"
    suite_dir.mkdir()
    result = resolve_path("missing.txt", suite_dir=suite_dir)
    # Falls back to suite-relative candidate
    assert result == (suite_dir / "missing.txt").resolve()


# ---------------------------------------------------------------------------
# normalize_prompt / dedupe_prompts
# ---------------------------------------------------------------------------


def test_normalize_prompt() -> None:
    assert normalize_prompt("  Hello   World  ") == "hello world"
    assert normalize_prompt("NoCHANGE") == "nochange"


def test_dedupe_prompts_preserves_order() -> None:
    prompts = ["A cat", "a  cat", "A dog", "A CAT"]
    result = dedupe_prompts(prompts)
    assert result == ["A cat", "A dog"]


def test_dedupe_prompts_empty() -> None:
    assert dedupe_prompts([]) == []


# ---------------------------------------------------------------------------
# JSON I/O
# ---------------------------------------------------------------------------


def test_atomic_write_json_creates_file(tmp_path: Path) -> None:
    path = tmp_path / "out.json"
    atomic_write_json(path, {"key": "value"})
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload == {"key": "value"}


def test_atomic_write_json_overwrites(tmp_path: Path) -> None:
    path = tmp_path / "out.json"
    atomic_write_json(path, {"v": 1})
    atomic_write_json(path, {"v": 2})
    assert json.loads(path.read_text(encoding="utf-8"))["v"] == 2


def test_atomic_write_json_creates_parent_dirs(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "deep" / "out.json"
    atomic_write_json(path, {"ok": True})
    assert path.exists()


def test_read_json_dict_returns_dict(tmp_path: Path) -> None:
    path = tmp_path / "data.json"
    path.write_text('{"a": 1}', encoding="utf-8")
    result = read_json_dict(path)
    assert result == {"a": 1}


def test_read_json_dict_returns_none_for_missing(tmp_path: Path) -> None:
    assert read_json_dict(tmp_path / "missing.json") is None


def test_read_json_dict_returns_none_for_list(tmp_path: Path) -> None:
    path = tmp_path / "list.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")
    assert read_json_dict(path) is None


def test_read_json_dict_returns_none_for_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("{broken", encoding="utf-8")
    assert read_json_dict(path) is None


def test_safe_write_json_returns_true(tmp_path: Path) -> None:
    path = tmp_path / "ok.json"
    assert safe_write_json(path, {"ok": True}) is True
    assert path.exists()


def test_safe_write_json_returns_false_on_error() -> None:
    # Path that can't be written (invalid parent)
    bad_path = Path("/\x00/invalid/path.json")
    assert safe_write_json(bad_path, {"ok": True}) is False
