from __future__ import annotations

import json
import os
import zipfile
from pathlib import Path

from temporalci.services.campaign_artifact_service import (
    CampaignArtifactBuildRequest,
    build_campaign_artifacts,
)


def test_build_campaign_artifacts_writes_manifest_provenance_and_zip(tmp_path: Path) -> None:
    output_dir = tmp_path / "campaign"
    pair_dir = output_dir / "sales_packs" / "suite-a" / "base__vs__target"
    pair_dir.mkdir(parents=True, exist_ok=True)
    (pair_dir / "manifest.json").write_text("{}", encoding="utf-8")
    (pair_dir / "SALES_PACK.pdf").write_bytes(b"%PDF-1.4\n%demo\n")
    (pair_dir / "SALES_PACK_SUMMARY.md").write_text("# summary\n", encoding="utf-8")
    (pair_dir / "01_proof_pack.json").write_text(
        json.dumps({"comparison": {"rows": [], "condition_rows": []}}),
        encoding="utf-8",
    )
    (pair_dir / "03_failure_casebook.json").write_text("[]", encoding="utf-8")

    summary_path = output_dir / "campaign_manifest.json"
    provenance_path = output_dir / "provenance.json"

    result = build_campaign_artifacts(
        CampaignArtifactBuildRequest(
            output_dir=output_dir,
            summary_path=summary_path,
            provenance_path=provenance_path,
            campaign_id="cmp-123",
            coordinator_url="http://localhost:8080",
            project="temporalci-demo",
            suite_name="regression_core",
            suite_paths=["suites/regression_core.yaml"],
            model_names=["base", "target"],
            runs_per_combination=1,
            max_inflight=2,
            poll_interval_sec=1.0,
            request_timeout_sec=10,
            max_wait_sec=60.0,
            pair_mode="all",
            baseline_model=None,
            output_root=None,
            priority=None,
            min_vram_gb=None,
            max_gpu_util_percent=None,
            max_gpu_memory_used_gb=None,
            require_sig=True,
            require_sig_min=1,
            require_metric="temporal_structure_v1",
            require_metric_delta_abs=0.1,
            require_metric_sig=True,
            require_metric_sig_min=1,
            window_started_at="2026-02-23T00:00:00+00:00",
            window_finished_at="2026-02-23T00:01:00+00:00",
            jobs_total=2,
            jobs_pending=0,
            jobs_inflight=0,
            terminal_failures=0,
            stop_reason=None,
            timeout_sec=None,
            timed_out_run_ids=[],
            cancellation_requested=False,
            suite_display_by_path={"suites/regression_core.yaml": "regression_core.yaml"},
            suite_snapshot_records=[],
            submitted_runs=[],
            terminal_runs=[],
            submit_errors=[],
            poll_errors=[],
            sales_pack_results=[
                {
                    "suite_path": "suites/regression_core.yaml",
                    "suite_archive_id": "suite-a",
                    "pair_slug": "base__vs__target",
                    "output_dir": str(pair_dir),
                    "exit_code": 0,
                }
            ],
            validation_required=True,
            validation_passed=True,
            qualifying_pairs=1,
            validation_pairs=[],
            temporalci_version="0.0.0-test",
            git_info={"available": False, "commit": None, "dirty": None, "branch": None},
            zip_enabled=True,
            zip_mode="full",
            zip_output=str(output_dir / "bundle.zip"),
        )
    )

    assert result.summary_path.exists()
    assert result.provenance_path.exists()
    assert result.zip_path is not None and result.zip_path.exists()
    assert int(result.zip_entries) >= 3
    assert isinstance(result.zip_sha256, str) and len(result.zip_sha256) == 64

    summary_payload = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary_payload["schema_version"] == "campaign_manifest.v1"
    assert summary_payload["artifacts"]["zip_path"] == "bundle.zip"
    assert int(summary_payload["artifacts"]["zip_entries"]) >= 3

    provenance_payload = json.loads(result.provenance_path.read_text(encoding="utf-8"))
    assert provenance_payload["schema_version"] == "campaign_provenance.v1"
    assert provenance_payload["artifacts"]["campaign_zip"]["sha256"] == result.zip_sha256

    with zipfile.ZipFile(result.zip_path, "r") as zf:
        names = zf.namelist()
    assert "campaign_manifest.json" in names
    assert "provenance.json" in names
    assert any(name.endswith("/manifest.json") for name in names)


def test_build_campaign_artifacts_zip_is_deterministic_for_same_payload(tmp_path: Path) -> None:
    output_dir = tmp_path / "campaign_deterministic"
    pair_dir = output_dir / "sales_packs" / "suite-a" / "base__vs__target"
    pair_dir.mkdir(parents=True, exist_ok=True)
    (pair_dir / "manifest.json").write_text("{}", encoding="utf-8")
    (pair_dir / "SALES_PACK.pdf").write_bytes(b"%PDF-1.4\n%demo\n")
    (pair_dir / "SALES_PACK_SUMMARY.md").write_text("# summary\n", encoding="utf-8")
    (pair_dir / "01_proof_pack.json").write_text(
        json.dumps({"comparison": {"rows": [], "condition_rows": []}}),
        encoding="utf-8",
    )
    (pair_dir / "03_failure_casebook.json").write_text("[]", encoding="utf-8")

    request = CampaignArtifactBuildRequest(
        output_dir=output_dir,
        summary_path=output_dir / "campaign_manifest.json",
        provenance_path=output_dir / "provenance.json",
        campaign_id="cmp-deterministic",
        coordinator_url="http://localhost:8080",
        project="temporalci-demo",
        suite_name="regression_core",
        suite_paths=["suites/regression_core.yaml"],
        model_names=["base", "target"],
        runs_per_combination=1,
        max_inflight=2,
        poll_interval_sec=1.0,
        request_timeout_sec=10,
        max_wait_sec=60.0,
        pair_mode="all",
        baseline_model=None,
        output_root=None,
        priority=None,
        min_vram_gb=None,
        max_gpu_util_percent=None,
        max_gpu_memory_used_gb=None,
        require_sig=True,
        require_sig_min=1,
        require_metric="temporal_structure_v1",
        require_metric_delta_abs=0.1,
        require_metric_sig=True,
        require_metric_sig_min=1,
        window_started_at="2026-02-23T00:00:00+00:00",
        window_finished_at="2026-02-23T00:01:00+00:00",
        jobs_total=2,
        jobs_pending=0,
        jobs_inflight=0,
        terminal_failures=0,
        stop_reason=None,
        timeout_sec=None,
        timed_out_run_ids=[],
        cancellation_requested=False,
        suite_display_by_path={"suites/regression_core.yaml": "regression_core.yaml"},
        suite_snapshot_records=[],
        submitted_runs=[],
        terminal_runs=[],
        submit_errors=[],
        poll_errors=[],
        sales_pack_results=[
            {
                "suite_path": "suites/regression_core.yaml",
                "suite_archive_id": "suite-a",
                "pair_slug": "base__vs__target",
                "output_dir": str(pair_dir),
                "exit_code": 0,
            }
        ],
        validation_required=True,
        validation_passed=True,
        qualifying_pairs=1,
        validation_pairs=[],
        temporalci_version="0.0.0-test",
        git_info={"available": False, "commit": None, "dirty": None, "branch": None},
        generated_at_utc="2026-02-23T00:00:00+00:00",
        zip_enabled=True,
        zip_mode="full",
        zip_output=str(output_dir / "bundle.zip"),
    )

    result_first = build_campaign_artifacts(request)
    first_zip_bytes = result_first.zip_path.read_bytes() if result_first.zip_path is not None else b""
    assert result_first.zip_sha256 is not None

    for source in sorted([p for p in pair_dir.rglob("*") if p.is_file()], key=lambda p: str(p)):
        os.utime(source, (1_700_000_100, 1_700_000_100))

    result_second = build_campaign_artifacts(request)
    second_zip_bytes = result_second.zip_path.read_bytes() if result_second.zip_path is not None else b""
    assert result_second.zip_sha256 is not None
    assert result_first.zip_sha256 == result_second.zip_sha256
    assert first_zip_bytes == second_zip_bytes
