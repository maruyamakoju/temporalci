from __future__ import annotations

import subprocess
import sys


def test_coordinator_cli_module_help_works() -> None:
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "temporalci.coordinator.cli", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0
    assert "TemporalCI distributed coordinator/worker helper" in result.stdout
