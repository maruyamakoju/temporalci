from __future__ import annotations

from pathlib import Path

import yaml

from temporalci.config import load_suite
from temporalci.engine import run_suite


def _suite_payload(*, artifacts: dict[str, object]) -> dict[str, object]:
    return {
        "version": 1,
        "project": "demo",
        "suite_name": "artifacts-policy",
        "models": [{"name": "m1", "adapter": "mock"}],
        "tests": [
            {
                "id": "t1",
                "type": "generation",
                "prompts": ["a safe prompt", "another safe prompt"],
                "seeds": [0, 1],
                "video": {"num_frames": 25},
            }
        ],
        "metrics": [
            {"name": "vbench_temporal"},
            {"name": "safety_t2v", "params": {"policies": ["violence"]}},
        ],
        "gates": [
            {"metric": "vbench_temporal.score", "op": ">=", "value": 0.1},
            {"metric": "safety_t2v.violations", "op": "==", "value": 0},
        ],
        "artifacts": artifacts,
    }


def _write_suite(path: Path, payload: dict[str, object]) -> Path:
    suite_path = path / "suite.yaml"
    suite_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return suite_path


def test_artifacts_video_none_deletes_files(tmp_path: Path) -> None:
    suite = load_suite(_write_suite(tmp_path, _suite_payload(artifacts={"video": "none"})))
    result = run_suite(suite=suite, artifacts_dir=tmp_path / "artifacts")

    assert result["status"] == "PASS"
    sample_rows = result["samples"]
    assert all(row["video_path"] is None for row in sample_rows)
    assert all(row["artifact_retained"] is False for row in sample_rows)


def test_artifacts_max_samples_keeps_subset(tmp_path: Path) -> None:
    suite = load_suite(
        _write_suite(
            tmp_path,
            _suite_payload(artifacts={"video": "all", "max_samples": 1}),
        )
    )
    result = run_suite(suite=suite, artifacts_dir=tmp_path / "artifacts")
    sample_rows = result["samples"]
    retained = [row for row in sample_rows if row["artifact_retained"]]
    assert len(retained) == 1
