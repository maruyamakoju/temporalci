from __future__ import annotations

from datetime import datetime
from pathlib import Path

from temporalci.autopilot_utils import atomic_write_json
from temporalci.autopilot_utils import pid_exists
from temporalci.autopilot_utils import read_json_dict
from temporalci.autopilot_utils import safe_write_json
from temporalci.autopilot_utils import terminate_pid
from temporalci.autopilot_utils import utc_now_iso


def test_utc_now_iso_is_parseable() -> None:
    raw = utc_now_iso()
    parsed = datetime.fromisoformat(raw)
    assert parsed.tzinfo is not None


def test_read_json_dict_handles_missing_and_invalid(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    assert read_json_dict(missing) is None

    invalid = tmp_path / "broken.json"
    invalid.write_text("{not json", encoding="utf-8")
    assert read_json_dict(invalid) is None


def test_atomic_write_and_safe_write_json(tmp_path: Path) -> None:
    path = tmp_path / "payload.json"
    atomic_write_json(path, {"state": "one"})
    payload = read_json_dict(path)
    assert payload is not None
    assert payload["state"] == "one"

    ok = safe_write_json(path, {"state": "two"})
    assert ok is True
    payload = read_json_dict(path)
    assert payload is not None
    assert payload["state"] == "two"


def test_pid_exists_and_terminate_invalid_pid() -> None:
    assert pid_exists(0) is False
    assert terminate_pid(0) is False
