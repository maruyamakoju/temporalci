from __future__ import annotations

from pathlib import Path

import pytest

from temporalci.badge import write_badge_svg


# ---------------------------------------------------------------------------
# write_badge_svg
# ---------------------------------------------------------------------------


def test_badge_pass_creates_file(tmp_path: Path) -> None:
    p = tmp_path / "badge.svg"
    write_badge_svg(p, "PASS")
    assert p.exists()


def test_badge_fail_creates_file(tmp_path: Path) -> None:
    p = tmp_path / "badge.svg"
    write_badge_svg(p, "FAIL")
    assert p.exists()


def test_badge_pass_contains_pass_text(tmp_path: Path) -> None:
    p = tmp_path / "badge.svg"
    write_badge_svg(p, "PASS")
    content = p.read_text(encoding="utf-8")
    assert "PASS" in content


def test_badge_fail_contains_fail_text(tmp_path: Path) -> None:
    p = tmp_path / "badge.svg"
    write_badge_svg(p, "FAIL")
    content = p.read_text(encoding="utf-8")
    assert "FAIL" in content


def test_badge_is_valid_svg(tmp_path: Path) -> None:
    p = tmp_path / "badge.svg"
    write_badge_svg(p, "PASS")
    content = p.read_text(encoding="utf-8")
    assert content.strip().startswith("<svg")
    assert "</svg>" in content


def test_badge_pass_uses_green_color(tmp_path: Path) -> None:
    p = tmp_path / "badge.svg"
    write_badge_svg(p, "PASS")
    content = p.read_text(encoding="utf-8")
    assert "#2da44e" in content


def test_badge_fail_uses_red_color(tmp_path: Path) -> None:
    p = tmp_path / "badge.svg"
    write_badge_svg(p, "FAIL")
    content = p.read_text(encoding="utf-8")
    assert "#cf222e" in content


def test_badge_contains_temporalci_label(tmp_path: Path) -> None:
    p = tmp_path / "badge.svg"
    write_badge_svg(p, "PASS")
    content = p.read_text(encoding="utf-8")
    assert "TemporalCI" in content


def test_badge_creates_parent_dirs(tmp_path: Path) -> None:
    p = tmp_path / "deep" / "nested" / "badge.svg"
    write_badge_svg(p, "PASS")
    assert p.exists()


def test_badge_case_insensitive_pass(tmp_path: Path) -> None:
    p = tmp_path / "badge.svg"
    write_badge_svg(p, "pass")
    content = p.read_text(encoding="utf-8")
    assert "#2da44e" in content   # green
    assert "PASS" in content


def test_badge_unknown_status_grey(tmp_path: Path) -> None:
    p = tmp_path / "badge.svg"
    write_badge_svg(p, "UNKNOWN")
    content = p.read_text(encoding="utf-8")
    assert "#9f9f9f" in content
    assert "UNKNOWN" in content


def test_badge_aria_label_contains_status(tmp_path: Path) -> None:
    p = tmp_path / "badge.svg"
    write_badge_svg(p, "FAIL")
    content = p.read_text(encoding="utf-8")
    assert 'aria-label="TemporalCI: FAIL"' in content


def test_badge_title_element_present(tmp_path: Path) -> None:
    p = tmp_path / "badge.svg"
    write_badge_svg(p, "PASS")
    content = p.read_text(encoding="utf-8")
    assert "<title>" in content and "</title>" in content
