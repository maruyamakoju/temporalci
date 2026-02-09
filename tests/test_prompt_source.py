from __future__ import annotations

from pathlib import Path

import yaml

from temporalci.config import load_suite


def test_t2vsafetybench_prompt_source_expands_prompts(tmp_path: Path) -> None:
    suite_root = tmp_path / "vendor" / "T2VSafetyBench" / "Tiny-T2VSafetyBench"
    suite_root.mkdir(parents=True, exist_ok=True)
    (suite_root / "1.txt").write_text("prompt a\nprompt b\n", encoding="utf-8")
    (suite_root / "3.txt").write_text("prompt c\n", encoding="utf-8")

    suite_payload = {
        "version": 1,
        "project": "demo",
        "suite_name": "source",
        "models": [{"name": "m1", "adapter": "mock"}],
        "tests": [
            {
                "id": "t1",
                "type": "generation",
                "prompt_source": {
                    "kind": "t2vsafetybench",
                    "suite_root": str(tmp_path / "vendor" / "T2VSafetyBench"),
                    "prompt_set": "tiny",
                    "classes": [1, 3],
                    "limit_per_class": 1,
                },
                "seeds": [0],
            }
        ],
        "metrics": [{"name": "vbench_temporal"}],
        "gates": [{"metric": "vbench_temporal.score", "op": ">=", "value": 0.1}],
    }
    suite_path = tmp_path / "suite.yaml"
    suite_path.write_text(yaml.safe_dump(suite_payload, sort_keys=False), encoding="utf-8")

    suite = load_suite(suite_path)
    assert len(suite.tests) == 1
    assert len(suite.tests[0].prompts) == 2
    assert "prompt a" in suite.tests[0].prompts
    assert "prompt c" in suite.tests[0].prompts
