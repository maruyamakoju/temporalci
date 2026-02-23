from __future__ import annotations

import json
from pathlib import Path

import pytest

from temporalci.contracts.campaign import CampaignManifest, CampaignProvenance


def test_campaign_manifest_json_contains_schema_version(tmp_path: Path) -> None:
    manifest = CampaignManifest(
        campaign_id="cmp-123",
        generated_at_utc="2026-02-23T00:00:00+00:00",
        window={"started_at": "2026-02-23T00:00:00+00:00", "finished_at": "2026-02-23T00:01:00+00:00"},
        config={"project": "temporalci-demo"},
        summary={"jobs_total": 2},
        submitted_runs=[],
        terminal_runs=[],
        submit_errors=[],
        poll_errors=[],
        sales_packs=[],
        validation={"required": False, "passed": True, "qualifying_pairs": 0, "checked_pairs": []},
        provenance={"path": "provenance.json"},
        artifacts={"zip_path": None, "zip_entries": 0, "zip_mode": None, "provenance_path": "provenance.json"},
    )
    path = manifest.write_json(tmp_path / "campaign_manifest.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "campaign_manifest.v1"
    assert payload["campaign_id"] == "cmp-123"


def test_campaign_provenance_json_contains_schema_version(tmp_path: Path) -> None:
    provenance = CampaignProvenance(
        generated_at_utc="2026-02-23T00:00:00+00:00",
        campaign_id="cmp-123",
        temporalci={"version": "0.0.0-test"},
        runtime={"python_version": "3.10"},
        git={"available": False, "commit": None, "dirty": None, "branch": None},
        config={"project": "temporalci-demo"},
        suite_snapshots=[],
        artifacts={"campaign_manifest": {"path": "campaign_manifest.json", "sha256": "abc"}},
    )
    path = provenance.write_json(tmp_path / "provenance.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "campaign_provenance.v1"
    assert payload["campaign_id"] == "cmp-123"


def test_campaign_manifest_from_file_round_trip(tmp_path: Path) -> None:
    manifest = CampaignManifest(
        campaign_id="cmp-123",
        generated_at_utc="2026-02-23T00:00:00+00:00",
        window={"started_at": "2026-02-23T00:00:00+00:00", "finished_at": "2026-02-23T00:01:00+00:00"},
        config={"project": "temporalci-demo"},
        summary={"jobs_total": 2},
        submitted_runs=[],
        terminal_runs=[],
        submit_errors=[],
        poll_errors=[],
        sales_packs=[],
        validation={"required": False, "passed": True, "qualifying_pairs": 0, "checked_pairs": []},
        provenance={"path": "provenance.json"},
        artifacts={"zip_path": None, "zip_entries": 0, "zip_mode": None, "provenance_path": "provenance.json"},
    )
    path = manifest.write_json(tmp_path / "campaign_manifest.json")
    loaded = CampaignManifest.from_file(path)
    assert loaded.schema_version == "campaign_manifest.v1"
    assert loaded.campaign_id == "cmp-123"


def test_campaign_manifest_from_dict_rejects_missing_required_key() -> None:
    with pytest.raises(ValueError):
        CampaignManifest.from_dict(
            {
                "schema_version": "campaign_manifest.v1",
                "campaign_id": "cmp-123",
                "generated_at_utc": "2026-02-23T00:00:00+00:00",
                # window is intentionally missing
                "config": {},
                "summary": {},
                "submitted_runs": [],
                "terminal_runs": [],
                "submit_errors": [],
                "poll_errors": [],
                "sales_packs": [],
                "validation": {},
                "provenance": {},
                "artifacts": {},
            }
        )


def test_campaign_provenance_from_dict_rejects_non_object_runtime() -> None:
    with pytest.raises(ValueError):
        CampaignProvenance.from_dict(
            {
                "schema_version": "campaign_provenance.v1",
                "generated_at_utc": "2026-02-23T00:00:00+00:00",
                "campaign_id": "cmp-123",
                "temporalci": {"version": "0.0.0-test"},
                "runtime": "invalid",
                "git": None,
                "config": {},
                "suite_snapshots": [],
                "artifacts": {},
            }
        )
