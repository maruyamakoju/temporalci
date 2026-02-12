from __future__ import annotations

from temporalci.adapters.diffusers_img2vid import DiffusersImg2VidAdapter


def test_diffusers_adapter_parses_boolean_string_params() -> None:
    adapter = DiffusersImg2VidAdapter(
        model_name="svd",
        params={
            "checkpoint": "dummy-checkpoint",
            "use_safetensors": "false",
            "disable_progress_bar": "false",
        },
    )
    assert adapter.use_safetensors is False
    assert adapter.disable_progress_bar is False
