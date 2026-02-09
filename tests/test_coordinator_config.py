from __future__ import annotations

from temporalci.coordinator.config import CoordinatorSettings


def test_coordinator_settings_include_lease_and_heartbeat(monkeypatch: object) -> None:
    monkeypatch.setenv("TEMPORALCI_TASK_LEASE_SEC", "10")
    monkeypatch.setenv("TEMPORALCI_HEARTBEAT_INTERVAL_SEC", "1")

    settings = CoordinatorSettings.from_env()
    assert settings.task_lease_sec == 30
    assert settings.heartbeat_interval_sec == 5
