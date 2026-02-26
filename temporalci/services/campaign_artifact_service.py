from __future__ import annotations

import hashlib
import json
import platform
import re
import shutil
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from temporalci.contracts.campaign import CampaignManifest, CampaignProvenance


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


_SAFE_SLUG_RE = re.compile(r"[^a-z0-9._-]+")


def safe_slug(value: str, *, fallback: str = "item") -> str:
    text = str(value or "").strip().lower()
    text = text.replace(" ", "-")
    text = _SAFE_SLUG_RE.sub("-", text).strip("-.")
    if text == "":
        return fallback
    return text


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_suite_basename(value: str | None) -> str:
    text = str(value or "").replace("\\", "/").strip()
    name = Path(text).name.strip() if text != "" else ""
    return name or "suite.yaml"


def _path_for_manifest(path: Path | str | None, *, root: Path) -> str | None:
    if path is None:
        return None
    path_obj = Path(str(path)).expanduser()
    try:
        return str(path_obj.resolve().relative_to(root.resolve())).replace("\\", "/")
    except Exception:
        text = str(path_obj).replace("\\", "/")
        return Path(text).name or text


def _display_suite_path(raw: str | None, *, suite_display_by_path: dict[str, str]) -> str:
    text = str(raw or "").strip()
    if text == "":
        return "unknown"
    return suite_display_by_path.get(text, _safe_suite_basename(text))


def _looks_like_local_abs_path(text: str) -> bool:
    raw = str(text or "").strip()
    if raw == "":
        return False
    lowered = raw.lower()
    if lowered.startswith(("http://", "https://", "s3://", "gs://", "az://", "file://")):
        return False
    if len(raw) >= 3 and raw[1] == ":" and raw[2] in {"\\", "/"}:
        return True
    if raw.startswith("\\\\"):
        return True
    if raw.startswith("/"):
        return True
    return False


def _redact_local_path(text: str) -> str:
    normalized = str(text or "").replace("\\", "/").strip()
    name = Path(normalized).name.strip()
    if name != "":
        return name
    return "redacted-path"


_PATH_KEYWORDS = (
    "path",
    "paths",
    "uri",
    "ref",
    "output_dir",
    "summary",
    "report",
    "artifact",
    "thumbnail",
)

_ZIP_FIXED_DATETIME = (1980, 1, 1, 0, 0, 0)
_ZIP_FILE_EXTERNAL_ATTR = 0o100644 << 16
_ZIP_MODES = {"pdf", "full"}


def _sanitize_for_distribution(value: Any, *, key_hint: str | None = None) -> Any:
    if isinstance(value, dict):
        return {str(k): _sanitize_for_distribution(v, key_hint=str(k)) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_for_distribution(v, key_hint=key_hint) for v in value]
    if isinstance(value, str):
        key_text = str(key_hint or "").lower()
        is_path_key = any(token in key_text for token in _PATH_KEYWORDS)
        if _looks_like_local_abs_path(value):
            return _redact_local_path(value)
        if is_path_key:
            normalized = value.replace("\\", "/")
            if ":" in normalized and not normalized.startswith(
                ("http://", "https://", "s3://", "gs://", "az://")
            ):
                return _redact_local_path(normalized)
            return normalized
    return value


def _load_json_any(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _normalize_arcname(text: str) -> str:
    return str(text or "").replace("\\", "/").lstrip("/")


def _write_distribution_json(zf: zipfile.ZipFile, *, source: Path, arcname: str) -> bool:
    payload = _load_json_any(source)
    if payload is None:
        return False
    sanitized = _sanitize_for_distribution(payload)
    encoded = json.dumps(sanitized, ensure_ascii=True, indent=2, sort_keys=True).encode("utf-8")
    info = zipfile.ZipInfo(filename=_normalize_arcname(arcname), date_time=_ZIP_FIXED_DATETIME)
    info.compress_type = zf.compression
    info.create_system = 3
    info.external_attr = _ZIP_FILE_EXTERNAL_ATTR
    zf.writestr(info, encoded)
    return True


def _write_distribution_file(zf: zipfile.ZipFile, *, source: Path, arcname: str) -> bool:
    if not source.exists() or not source.is_file():
        return False
    info = zipfile.ZipInfo(filename=_normalize_arcname(arcname), date_time=_ZIP_FIXED_DATETIME)
    info.compress_type = zf.compression
    info.create_system = 3
    info.external_attr = _ZIP_FILE_EXTERNAL_ATTR
    with zf.open(info, "w") as dst, source.open("rb") as src:
        shutil.copyfileobj(src, dst, length=1024 * 1024)
    return True


def _build_suite_map_entries(suite_snapshot_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for row in suite_snapshot_records:
        entries.append(
            {
                "display_path": row.get("display_path"),
                "suite_slug": row.get("suite_slug"),
                "suite_archive_id": row.get("suite_archive_id"),
                "source_basename": row.get("source_basename"),
                "snapshot_path": row.get("snapshot_path"),
                "sha256": row.get("sha256"),
                "bytes": row.get("bytes"),
                "exists": bool(row.get("exists")),
                "error": row.get("error"),
            }
        )
    return entries


def _sanitize_run_record(
    record: dict[str, Any], *, suite_display_by_path: dict[str, str]
) -> dict[str, Any]:
    out = dict(record)
    out["suite_path"] = _display_suite_path(
        str(out.get("suite_path") or ""), suite_display_by_path=suite_display_by_path
    )
    if "suite_display_path" in out:
        raw_display = str(out.get("suite_display_path") or out.get("suite_path") or "")
        out["suite_display_path"] = _display_suite_path(
            raw_display, suite_display_by_path=suite_display_by_path
        )
    return out


def _build_runtime_info(runtime_info: dict[str, Any] | None) -> dict[str, Any]:
    if isinstance(runtime_info, dict):
        return dict(runtime_info)
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
    }


def _sales_pack_sort_key(row: dict[str, Any]) -> tuple[str, str, str]:
    suite_archive_id = safe_slug(str(row.get("suite_archive_id") or "suite"))
    pair_slug = safe_slug(str(row.get("pair_slug") or row.get("pair") or "pair"))
    output_dir = str(row.get("output_dir") or "")
    return (suite_archive_id, pair_slug, output_dir)


@dataclass(slots=True)
class CampaignArtifactBuildRequest:
    output_dir: str | Path
    summary_path: str | Path
    provenance_path: str | Path
    campaign_id: str
    coordinator_url: str
    project: str
    suite_name: str
    suite_paths: list[str]
    model_names: list[str]
    runs_per_combination: int
    max_inflight: int
    poll_interval_sec: float
    request_timeout_sec: int
    max_wait_sec: float
    pair_mode: str
    baseline_model: str | None
    output_root: str | None
    priority: int | None
    min_vram_gb: float | None
    max_gpu_util_percent: float | None
    max_gpu_memory_used_gb: float | None
    require_sig: bool
    require_sig_min: int
    require_metric: str | None
    require_metric_delta_abs: float | None
    require_metric_sig: bool
    require_metric_sig_min: int
    window_started_at: str
    window_finished_at: str
    jobs_total: int
    jobs_pending: int
    jobs_inflight: int
    terminal_failures: int
    stop_reason: str | None
    timeout_sec: float | None
    timed_out_run_ids: list[str]
    cancellation_requested: bool
    suite_display_by_path: dict[str, str]
    suite_snapshot_records: list[dict[str, Any]]
    submitted_runs: list[dict[str, Any]]
    terminal_runs: list[dict[str, Any]]
    submit_errors: list[dict[str, Any]]
    poll_errors: list[dict[str, Any]]
    sales_pack_results: list[dict[str, Any]]
    validation_required: bool
    validation_passed: bool
    qualifying_pairs: int
    validation_pairs: list[dict[str, Any]]
    temporalci_version: str | None
    git_info: dict[str, Any] | None
    generated_at_utc: str | None = None
    zip_enabled: bool = True
    zip_mode: str = "pdf"
    zip_output: str | None = None
    runtime_info: dict[str, Any] | None = None


@dataclass(slots=True)
class CampaignArtifactBuildResult:
    summary_path: Path
    provenance_path: Path
    zip_path: Path | None
    zip_entries: int
    manifest: dict[str, Any]
    zip_sha256: str | None


def _require_text(value: str | None, *, field_name: str) -> str:
    text = str(value or "").strip()
    if text == "":
        raise ValueError(f"{field_name} is required")
    return text


def _normalize_zip_mode(value: str | None) -> str:
    mode = str(value or "pdf").strip().lower()
    if mode not in _ZIP_MODES:
        raise ValueError(f"zip_mode must be one of {sorted(_ZIP_MODES)}, got: {value!r}")
    return mode


def _validate_build_request(request: CampaignArtifactBuildRequest) -> None:
    _require_text(request.campaign_id, field_name="campaign_id")
    _require_text(request.project, field_name="project")
    _require_text(request.suite_name, field_name="suite_name")
    _require_text(request.window_started_at, field_name="window_started_at")
    _require_text(request.window_finished_at, field_name="window_finished_at")
    _normalize_zip_mode(request.zip_mode)


def build_campaign_artifacts(request: CampaignArtifactBuildRequest) -> CampaignArtifactBuildResult:
    _validate_build_request(request)
    campaign_id = _require_text(request.campaign_id, field_name="campaign_id")
    project = _require_text(request.project, field_name="project")
    suite_name = _require_text(request.suite_name, field_name="suite_name")
    zip_mode = _normalize_zip_mode(request.zip_mode)

    output_dir = Path(str(request.output_dir)).expanduser().resolve()
    summary_path = Path(str(request.summary_path)).expanduser().resolve()
    provenance_path = Path(str(request.provenance_path)).expanduser().resolve()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    provenance_path.parent.mkdir(parents=True, exist_ok=True)

    suite_map_entries = _build_suite_map_entries(list(request.suite_snapshot_records or []))

    submitted_runs_manifest = [
        _sanitize_run_record(row, suite_display_by_path=request.suite_display_by_path)
        for row in list(request.submitted_runs or [])
    ]
    terminal_runs_manifest = [
        _sanitize_run_record(row, suite_display_by_path=request.suite_display_by_path)
        for row in list(request.terminal_runs or [])
    ]

    sorted_sales_pack_rows = sorted(
        list(request.sales_pack_results or []), key=_sales_pack_sort_key
    )

    sales_pack_results_manifest: list[dict[str, Any]] = []
    for row in sorted_sales_pack_rows:
        sanitized = dict(row)
        sanitized["suite_path"] = _display_suite_path(
            str(row.get("suite_path") or ""),
            suite_display_by_path=request.suite_display_by_path,
        )
        output_path = str(row.get("output_dir") or "")
        output_rel = _path_for_manifest(output_path, root=output_dir)
        sanitized["output_dir"] = str(output_rel or _redact_local_path(output_path))
        sanitized.pop("output_dir_rel", None)
        sales_pack_results_manifest.append(sanitized)

    validation_pairs_manifest: list[dict[str, Any]] = []
    for row in list(request.validation_pairs or []):
        sanitized = dict(row)
        proof_value = sanitized.get("proof_path")
        if isinstance(proof_value, str) and proof_value.strip() != "":
            sanitized["proof_path"] = str(
                _path_for_manifest(proof_value, root=output_dir) or proof_value
            )
        sanitized["suite_path"] = _display_suite_path(
            str(sanitized.get("suite_path") or ""),
            suite_display_by_path=request.suite_display_by_path,
        )
        validation_pairs_manifest.append(sanitized)

    terminal_counts: dict[str, int] = {}
    for terminal in terminal_runs_manifest:
        label = str(terminal.get("display_status") or "UNKNOWN")
        terminal_counts[label] = int(terminal_counts.get(label, 0)) + 1
    terminal_counts = dict(sorted(terminal_counts.items(), key=lambda item: str(item[0])))

    pack_errors = sum(
        1 for row in sales_pack_results_manifest if int(row.get("exit_code") or 0) != 0
    )
    suite_snapshot_rel_paths = sorted(
        {
            str(row.get("snapshot_path") or "").replace("\\", "/")
            for row in suite_map_entries
            if str(row.get("snapshot_path") or "").strip() != ""
        }
    )
    suite_snapshot_count = len(suite_snapshot_rel_paths)
    generated_at_text = str(request.generated_at_utc or "").strip()
    generated_at = generated_at_text if generated_at_text != "" else now_utc_iso()
    provenance_rel_path = str(
        _path_for_manifest(provenance_path, root=output_dir) or "provenance.json"
    )

    manifest_model = CampaignManifest(
        campaign_id=campaign_id,
        generated_at_utc=generated_at,
        window={
            "started_at": str(request.window_started_at),
            "finished_at": str(request.window_finished_at),
        },
        config={
            "coordinator_url": str(request.coordinator_url),
            "project": project,
            "suite_name": suite_name,
            "suite_paths": [
                _display_suite_path(path, suite_display_by_path=request.suite_display_by_path)
                for path in list(request.suite_paths or [])
            ],
            "suite_path_map": suite_map_entries,
            "model_names": list(request.model_names or []),
            "runs_per_combination": int(request.runs_per_combination),
            "max_inflight": int(request.max_inflight),
            "poll_interval_sec": float(request.poll_interval_sec),
            "request_timeout_sec": int(request.request_timeout_sec),
            "max_wait_sec": float(request.max_wait_sec),
            "pair_mode": str(request.pair_mode),
            "baseline_model": (
                None if request.baseline_model is None else str(request.baseline_model)
            ),
            "output_root": (None if request.output_root is None else str(request.output_root)),
            "campaign_filter_mode": "campaign_id+suite_path",
            "priority": request.priority,
            "min_vram_gb": request.min_vram_gb,
            "max_gpu_util_percent": request.max_gpu_util_percent,
            "max_gpu_memory_used_gb": request.max_gpu_memory_used_gb,
            "zip": bool(request.zip_enabled),
            "zip_mode": zip_mode,
            "require_sig": bool(request.require_sig),
            "require_sig_min": int(max(1, int(request.require_sig_min))),
            "require_metric": (
                None if request.require_metric is None else str(request.require_metric)
            ),
            "require_metric_delta_abs": request.require_metric_delta_abs,
            "require_metric_sig": bool(request.require_metric_sig),
            "require_metric_sig_min": int(max(1, int(request.require_metric_sig_min))),
        },
        summary={
            "jobs_total": int(request.jobs_total),
            "jobs_submitted": len(submitted_runs_manifest),
            "jobs_terminal": len(terminal_runs_manifest),
            "jobs_pending": int(request.jobs_pending),
            "jobs_inflight": int(request.jobs_inflight),
            "terminal_failures": int(request.terminal_failures),
            "terminal_counts": terminal_counts,
            "submit_error_count": len(list(request.submit_errors or [])),
            "poll_error_count": len(list(request.poll_errors or [])),
            "sales_pack_pairs": len(sales_pack_results_manifest),
            "sales_pack_errors": int(pack_errors),
            "validation_required": bool(request.validation_required),
            "validation_passed": bool(request.validation_passed),
            "validation_qualifying_pairs": int(request.qualifying_pairs),
            "validation_checked_pairs": len(validation_pairs_manifest),
            "stop_reason": request.stop_reason,
            "timeout_sec": (None if request.timeout_sec is None else float(request.timeout_sec)),
            "timed_out_run_ids": list(request.timed_out_run_ids or []),
            "cancellation_requested": bool(request.cancellation_requested),
        },
        submitted_runs=submitted_runs_manifest,
        terminal_runs=terminal_runs_manifest,
        submit_errors=list(request.submit_errors or []),
        poll_errors=list(request.poll_errors or []),
        sales_packs=sales_pack_results_manifest,
        validation={
            "required": bool(request.validation_required),
            "passed": bool(request.validation_passed),
            "qualifying_pairs": int(request.qualifying_pairs),
            "checked_pairs": validation_pairs_manifest,
        },
        provenance={
            "path": provenance_rel_path,
        },
        artifacts={
            "zip_path": None,
            "zip_entries": 0,
            "zip_mode": None,
            "provenance_path": provenance_rel_path,
            "suite_snapshot_count": int(suite_snapshot_count),
        },
    )
    summary_path = manifest_model.write_json(summary_path)

    provenance_model = CampaignProvenance(
        generated_at_utc=generated_at,
        campaign_id=campaign_id,
        temporalci={"version": request.temporalci_version},
        runtime=_build_runtime_info(request.runtime_info),
        git=(None if request.git_info is None else dict(request.git_info)),
        config={
            "project": project,
            "suite_name": suite_name,
            "suite_paths": [
                _display_suite_path(path, suite_display_by_path=request.suite_display_by_path)
                for path in list(request.suite_paths or [])
            ],
            "model_names": list(request.model_names or []),
            "runs_per_combination": int(request.runs_per_combination),
            "pair_mode": str(request.pair_mode),
            "baseline_model": (
                None if request.baseline_model is None else str(request.baseline_model)
            ),
            "campaign_filter_mode": "campaign_id+suite_path",
            "require_sig": bool(request.require_sig),
            "require_sig_min": int(max(1, int(request.require_sig_min))),
            "require_metric": (
                None if request.require_metric is None else str(request.require_metric)
            ),
            "require_metric_delta_abs": request.require_metric_delta_abs,
            "require_metric_sig": bool(request.require_metric_sig),
            "require_metric_sig_min": int(max(1, int(request.require_metric_sig_min))),
        },
        suite_snapshots=suite_map_entries,
        artifacts={
            "campaign_manifest": {
                "path": str(_path_for_manifest(summary_path, root=output_dir) or summary_path.name),
                "sha256": _sha256_file(summary_path),
            }
        },
    )
    provenance_path = provenance_model.write_json(provenance_path)

    zip_path: Path | None = None
    zip_entries = 0
    zip_sha256: str | None = None
    if bool(request.zip_enabled):
        if isinstance(request.zip_output, str) and request.zip_output.strip() != "":
            zip_path = Path(request.zip_output).expanduser().resolve()
        else:
            zip_path = output_dir / f"{safe_slug(campaign_id)}.zip"
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            written_arcnames: set[str] = set()

            def _write_file_unique(source: Path, arcname: str) -> bool:
                normalized = _normalize_arcname(arcname)
                if normalized == "" or normalized in written_arcnames:
                    return False
                if not _write_distribution_file(zf, source=source, arcname=normalized):
                    return False
                written_arcnames.add(normalized)
                return True

            def _write_json_unique(source: Path, arcname: str) -> bool:
                normalized = _normalize_arcname(arcname)
                if normalized == "" or normalized in written_arcnames:
                    return False
                if _write_distribution_json(zf, source=source, arcname=normalized):
                    written_arcnames.add(normalized)
                    return True
                return _write_file_unique(source, normalized)

            if _write_json_unique(summary_path, "campaign_manifest.json"):
                zip_entries += 1
            if provenance_path.exists() and provenance_path.is_file():
                if _write_json_unique(provenance_path, "provenance.json"):
                    zip_entries += 1
            if zip_mode == "full":
                for snapshot_rel in suite_snapshot_rel_paths:
                    snapshot_source = output_dir / snapshot_rel
                    if not snapshot_source.exists() or not snapshot_source.is_file():
                        continue
                    if _write_file_unique(snapshot_source, snapshot_rel):
                        zip_entries += 1
            for row in sorted_sales_pack_rows:
                if int(row.get("exit_code") or 0) != 0:
                    continue
                pair_output_dir = Path(str(row.get("output_dir") or "")).expanduser()
                if not pair_output_dir.exists():
                    continue
                suite_archive_id = safe_slug(str(row.get("suite_archive_id") or "suite"))
                pair_slug = safe_slug(str(row.get("pair_slug") or row.get("pair") or "pair"))
                arc_prefix = f"sales_packs/{suite_archive_id}/{pair_slug}"
                rel_names = ["manifest.json", "SALES_PACK.pdf", "SALES_PACK_SUMMARY.md"]
                if zip_mode == "full":
                    rel_names.extend(
                        [
                            "01_proof_pack.json",
                            "01_proof_pack.md",
                            "02_human_correlation.json",
                            "02_human_correlation.md",
                            "03_failure_casebook.json",
                            "03_failure_casebook.md",
                        ]
                    )
                for rel_name in rel_names:
                    source = pair_output_dir / rel_name
                    if not source.exists() or not source.is_file():
                        continue
                    arcname = f"{arc_prefix}/{rel_name}"
                    if rel_name.endswith(".json"):
                        if _write_json_unique(source, arcname):
                            zip_entries += 1
                    elif _write_file_unique(source, arcname):
                        zip_entries += 1
                if zip_mode == "full":
                    assets_root = pair_output_dir / "assets"
                    if assets_root.exists() and assets_root.is_dir():
                        asset_sources = sorted(
                            [source for source in assets_root.rglob("*") if source.is_file()],
                            key=lambda path: str(path).replace("\\", "/"),
                        )
                        for source in asset_sources:
                            try:
                                asset_rel = source.resolve().relative_to(pair_output_dir.resolve())
                                asset_rel_text = str(asset_rel).replace("\\", "/")
                                arcname = f"{arc_prefix}/{asset_rel_text}"
                            except Exception:
                                arcname = f"{arc_prefix}/assets/{source.name}"
                            if _write_file_unique(source, arcname):
                                zip_entries += 1

        zip_sha256 = _sha256_file(zip_path)
        manifest_model.artifacts = {
            "zip_path": str(_path_for_manifest(zip_path, root=output_dir) or zip_path.name),
            "zip_entries": int(zip_entries),
            "zip_mode": zip_mode,
            "provenance_path": str(
                _path_for_manifest(provenance_path, root=output_dir) or "provenance.json"
            ),
            "suite_snapshot_count": int(suite_snapshot_count),
            "zip_sha256": zip_sha256,
        }
        summary_path = manifest_model.write_json(summary_path)

    artifacts = dict(provenance_model.artifacts or {})
    artifacts["campaign_manifest"] = {
        "path": str(_path_for_manifest(summary_path, root=output_dir) or summary_path.name),
        "sha256": _sha256_file(summary_path),
    }
    if zip_path is not None and zip_sha256 is not None:
        artifacts["campaign_zip"] = {
            "path": str(_path_for_manifest(zip_path, root=output_dir) or zip_path.name),
            "sha256": zip_sha256,
            "entries": int(zip_entries),
        }
    provenance_model.artifacts = artifacts
    provenance_path = provenance_model.write_json(provenance_path)

    return CampaignArtifactBuildResult(
        summary_path=summary_path,
        provenance_path=provenance_path,
        zip_path=zip_path,
        zip_entries=int(zip_entries),
        manifest=manifest_model.to_dict(),
        zip_sha256=zip_sha256,
    )
