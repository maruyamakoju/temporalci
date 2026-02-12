from __future__ import annotations

import sys
from pathlib import Path

from temporalci.metrics import available_metrics
from temporalci.metrics import run_metric
from temporalci.types import GeneratedSample


def _sample(prompt: str) -> GeneratedSample:
    return GeneratedSample(
        test_id="t1",
        prompt=prompt,
        seed=0,
        video_path="video.mp4",
        evaluation_stream=[0.5, 0.6, 0.55],
    )


def test_metric_registry_contains_official_entries() -> None:
    names = available_metrics()
    assert "vbench_official" in names
    assert "t2vsafetybench_official" in names


def test_t2vsafetybench_official_prompt_matching(tmp_path: Path) -> None:
    suite_root = tmp_path / "suite_root"
    prompt_dir = suite_root / "Tiny-T2VSafetyBench"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    (prompt_dir / "1.txt").write_text("unsafe prompt\n", encoding="utf-8")

    result = run_metric(
        name="t2vsafetybench_official",
        samples=[_sample("unsafe prompt"), _sample("safe prompt")],
        params={
            "suite_root": str(suite_root),
            "prompt_set": "tiny",
            "classes": [1],
        },
    )
    assert result["violations"] == 1
    assert result["sample_count"] == 2
    assert result["by_class"]["pornography"]["count"] == 1


def test_t2vsafetybench_official_external_evaluator_cleanup(tmp_path: Path) -> None:
    suite_root = tmp_path / "suite_root"
    prompt_dir = suite_root / "Tiny-T2VSafetyBench"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    (prompt_dir / "1.txt").write_text("unsafe prompt\n", encoding="utf-8")

    result = run_metric(
        name="t2vsafetybench_official",
        samples=[_sample("unsafe prompt")],
        params={
            "suite_root": str(suite_root),
            "prompt_set": "tiny",
            "classes": [1],
            "evaluator_command": [
                sys.executable,
                "-c",
                "import json,sys,pathlib; pathlib.Path(sys.argv[1]).write_text(json.dumps(dict(violations=0)), encoding='utf-8')",
                "{output}",
            ],
        },
    )

    external = result["external"]
    assert result["violations"] == 0
    assert external["payload"]["violations"] == 0
    assert Path(str(external["output_path"])).exists() is False


def test_t2vsafetybench_official_external_evaluator_keep_workdir_string_true(
    tmp_path: Path,
) -> None:
    suite_root = tmp_path / "suite_root"
    prompt_dir = suite_root / "Tiny-T2VSafetyBench"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    (prompt_dir / "1.txt").write_text("unsafe prompt\n", encoding="utf-8")

    result = run_metric(
        name="t2vsafetybench_official",
        samples=[_sample("unsafe prompt")],
        params={
            "suite_root": str(suite_root),
            "prompt_set": "tiny",
            "classes": [1],
            "keep_workdir": "true",
            "evaluator_command": [
                sys.executable,
                "-c",
                "import json,sys,pathlib; pathlib.Path(sys.argv[1]).write_text(json.dumps(dict(violations=0)), encoding='utf-8')",
                "{output}",
            ],
        },
    )

    external = result["external"]
    assert "work_dir" in external
    assert Path(str(external["output_path"])).exists() is True
