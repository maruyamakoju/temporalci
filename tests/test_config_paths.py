from __future__ import annotations

from pathlib import Path

import yaml

from temporalci.config import load_suite


def test_load_suite_resolves_init_image_paths_relative_to_suite(tmp_path: Path) -> None:
    assets_dir = tmp_path / "assets" / "init"
    assets_dir.mkdir(parents=True, exist_ok=True)
    image_path = assets_dir / "seed.png"
    image_path.write_bytes(b"png-placeholder")

    suite_payload = {
        "version": 1,
        "project": "demo",
        "suite_name": "path-resolution",
        "models": [
            {
                "name": "svd",
                "adapter": "diffusers_img2vid",
                "params": {
                    "checkpoint": "stabilityai/stable-video-diffusion-img2vid-xt",
                    "init_image": "assets/init/seed.png",
                    "init_images": ["assets/init/seed.png"],
                },
            }
        ],
        "tests": [
            {
                "id": "core",
                "type": "generation",
                "prompts": ["a demo prompt"],
                "seeds": [0],
                "video": {
                    "init_image": "assets/init/seed.png",
                    "init_images": ["assets/init/seed.png"],
                },
            }
        ],
        "metrics": [{"name": "vbench_temporal"}],
        "gates": [{"metric": "vbench_temporal.score", "op": ">=", "value": 0.1}],
    }
    suite_path = tmp_path / "suite.yaml"
    suite_path.write_text(yaml.safe_dump(suite_payload, sort_keys=False), encoding="utf-8")

    suite = load_suite(suite_path)

    expected = str(image_path.resolve())
    assert suite.models[0].params["init_image"] == expected
    assert suite.models[0].params["init_images"] == [expected]
    assert suite.tests[0].video["init_image"] == expected
    assert suite.tests[0].video["init_images"] == [expected]
