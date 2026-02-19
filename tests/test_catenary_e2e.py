"""End-to-end integration test for the catenary inspection pipeline.

Exercises the full path:
  directory prompt source → frame_archive adapter → catenary_vegetation metric → gate evaluation

Uses synthetic PIL images so no real data or external dependencies are needed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

try:
    from PIL import Image

    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

pytestmark = pytest.mark.skipif(not _HAS_PIL, reason="Pillow not installed")


def _create_suite_yaml(suite_dir: Path, frames_dir: Path) -> Path:
    yaml_path = suite_dir / "suite.yaml"
    yaml_path.write_text(
        f"""\
version: 1
project: "test-catenary"
suite_name: "e2e-veg"

models:
  - name: "cam-left"
    adapter: "frame_archive"
    params:
      archive_dir: "{frames_dir.as_posix()}"
      camera: "left"

tests:
  - id: "veg-check"
    type: "inspection"
    prompt_source:
      kind: "directory"
      path: "{frames_dir.as_posix()}"
      pattern: "*.jpg"
    seeds: [0]

metrics:
  - name: "catenary_vegetation"
    params:
      proximity_threshold: 0.05

gates:
  - metric: "catenary_vegetation.score"
    op: ">="
    value: 0.7
  - metric: "catenary_vegetation.dims.vegetation_proximity"
    op: "<="
    value: 0.15
""",
        encoding="utf-8",
    )
    return yaml_path


class TestCatenaryE2E:
    def test_safe_frames_pass_gate(self, tmp_path: Path) -> None:
        """Blue-sky frames should pass all gates."""
        from temporalci.config import load_suite
        from temporalci.engine import run_suite

        frames = tmp_path / "frames"
        frames.mkdir()
        for i in range(3):
            Image.new("RGB", (100, 100), (100, 130, 230)).save(str(frames / f"sky_{i}.jpg"))

        suite_dir = tmp_path / "suite"
        suite_dir.mkdir()
        yaml_path = _create_suite_yaml(suite_dir, frames)

        suite = load_suite(yaml_path)
        result = run_suite(
            suite=suite,
            artifacts_dir=str(tmp_path / "artifacts"),
            baseline_mode="none",
        )

        assert result["status"] == "PASS"
        assert result["gate_failed"] is False
        assert result["sample_count"] == 3
        score = result["metrics"]["catenary_vegetation"]["score"]
        assert score >= 0.7

    def test_green_frames_fail_gate(self, tmp_path: Path) -> None:
        """All-green frames should trigger gate failure."""
        from temporalci.config import load_suite
        from temporalci.engine import run_suite

        frames = tmp_path / "frames"
        frames.mkdir()
        for i in range(3):
            Image.new("RGB", (100, 100), (20, 150, 20)).save(str(frames / f"veg_{i}.jpg"))

        suite_dir = tmp_path / "suite"
        suite_dir.mkdir()
        yaml_path = _create_suite_yaml(suite_dir, frames)

        suite = load_suite(yaml_path)
        result = run_suite(
            suite=suite,
            artifacts_dir=str(tmp_path / "artifacts"),
            baseline_mode="none",
        )

        assert result["status"] == "FAIL"
        assert result["gate_failed"] is True
        prox = result["metrics"]["catenary_vegetation"]["dims"]["vegetation_proximity"]
        assert prox > 0.15
        alerts = result["metrics"]["catenary_vegetation"]["alert_frames"]
        assert len(alerts) == 3

    def test_mixed_frames_score_between(self, tmp_path: Path) -> None:
        """Mix of safe and vegetation frames should produce intermediate scores."""
        from temporalci.config import load_suite
        from temporalci.engine import run_suite

        frames = tmp_path / "frames"
        frames.mkdir()
        # 2 safe frames, 1 green frame
        Image.new("RGB", (100, 100), (100, 130, 230)).save(str(frames / "a_safe.jpg"))
        Image.new("RGB", (100, 100), (100, 130, 230)).save(str(frames / "b_safe.jpg"))
        Image.new("RGB", (100, 100), (20, 150, 20)).save(str(frames / "c_green.jpg"))

        suite_dir = tmp_path / "suite"
        suite_dir.mkdir()
        yaml_path = _create_suite_yaml(suite_dir, frames)

        suite = load_suite(yaml_path)
        result = run_suite(
            suite=suite,
            artifacts_dir=str(tmp_path / "artifacts"),
            baseline_mode="none",
        )

        score = result["metrics"]["catenary_vegetation"]["score"]
        # Score should be between full-green (~0.2) and full-blue (~0.95)
        assert 0.4 < score < 0.95
        assert result["sample_count"] == 3

    def test_artifacts_generated(self, tmp_path: Path) -> None:
        """Pipeline should generate report, badge, and index artifacts."""
        from temporalci.config import load_suite
        from temporalci.engine import run_suite

        frames = tmp_path / "frames"
        frames.mkdir()
        Image.new("RGB", (100, 100), (100, 130, 230)).save(str(frames / "f.jpg"))

        suite_dir = tmp_path / "suite"
        suite_dir.mkdir()
        yaml_path = _create_suite_yaml(suite_dir, frames)

        suite = load_suite(yaml_path)
        result = run_suite(
            suite=suite,
            artifacts_dir=str(tmp_path / "artifacts"),
            baseline_mode="none",
        )

        run_dir = Path(result["run_dir"])
        assert (run_dir / "run.json").exists()
        assert (run_dir / "report.html").exists()

        model_root = run_dir.parent
        assert (model_root / "badge.svg").exists()

        suite_root = model_root.parent
        assert (suite_root / "index.html").exists()

    def test_sample_limit(self, tmp_path: Path) -> None:
        """sample_limit should cap the number of processed frames."""
        from temporalci.config import load_suite
        from temporalci.engine import run_suite

        frames = tmp_path / "frames"
        frames.mkdir()
        for i in range(10):
            Image.new("RGB", (100, 100), (100, 130, 230)).save(str(frames / f"f_{i:02d}.jpg"))

        suite_dir = tmp_path / "suite"
        suite_dir.mkdir()
        yaml_path = _create_suite_yaml(suite_dir, frames)

        suite = load_suite(yaml_path)
        result = run_suite(
            suite=suite,
            artifacts_dir=str(tmp_path / "artifacts"),
            baseline_mode="none",
            sample_limit=3,
        )

        assert result["sample_count"] == 3
