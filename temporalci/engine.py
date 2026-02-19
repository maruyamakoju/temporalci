from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path
from typing import Any

from temporalci.adapters import build_adapter
from temporalci.baseline import (
    _average_metrics_dicts,
    _load_previous_run,
    _read_tags,
    _validate_baseline_mode,
    _write_tag,
)
from temporalci.config import select_model
from temporalci.gate_eval import (
    _apply_windowed_gates,
    _build_legacy_series_key,
    _compare,
    _compute_regressions,
    _evaluate_gates,
    _extract_metric_series,
    _load_recent_runs,
    _paired_deltas_for_gate,
    _read_sprt_params,
    _resolve_metric_path,
    _run_sprt,
    _sample_std,
    _split_metric_path,
)
from temporalci.trend import load_model_runs, write_trend_report
from temporalci.badge import write_badge_svg
from temporalci.compare import write_compare_report
from temporalci.index import write_suite_index
from temporalci.metrics import run_metric
from temporalci.report import write_html_report
from temporalci.types import GeneratedSample, RunResult, SuiteSpec
from temporalci.utils import (
    atomic_write_json,
    normalize_prompt,
    read_json_dict,
    utc_now,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_RUN_DIR_RETRY_LIMIT = 20


# ---------------------------------------------------------------------------
# Git metadata
# ---------------------------------------------------------------------------


def _capture_git_metadata() -> dict[str, Any]:
    """Return current git commit/branch/dirty status.

    Returns an empty dict if git is unavailable or the working directory is not
    inside a git repository — callers should treat absence of the key as "unknown".
    """
    import subprocess

    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        dirty_out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return {"commit": commit, "branch": branch, "dirty": bool(dirty_out)}
    except Exception:  # noqa: BLE001
        return {}

# ---------------------------------------------------------------------------
# Run directory helpers
# ---------------------------------------------------------------------------


def _new_run_id() -> str:
    return utc_now().strftime("%Y%m%dT%H%M%S%fZ")


def _create_run_dir(model_root: Path) -> tuple[str, Path]:
    """Allocate a unique timestamp-based run directory under *model_root*."""
    for _ in range(_RUN_DIR_RETRY_LIMIT):
        run_id = _new_run_id()
        run_dir = model_root / run_id
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
            return run_id, run_dir
        except FileExistsError:
            continue
    raise RuntimeError("failed to allocate unique run directory after retries")


# ---------------------------------------------------------------------------
# Artifact retention
# ---------------------------------------------------------------------------


def _safe_unlink(path: Path) -> bool:
    try:
        path.unlink()
        return True
    except (FileNotFoundError, OSError):
        return False


def _build_sample_rows_with_retention(
    *,
    samples: list[GeneratedSample],
    status: str,
    artifacts_cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    video_policy = str(artifacts_cfg.get("video", "all")).strip().lower()
    max_samples_raw = artifacts_cfg.get("max_samples")
    max_samples = int(max_samples_raw) if max_samples_raw is not None else None

    if video_policy == "none":
        retain_limit = 0
    elif video_policy == "failures_only" and status == "PASS":
        retain_limit = 0
    else:
        retain_limit = len(samples)

    if max_samples is not None:
        retain_limit = min(retain_limit, max_samples)

    rows: list[dict[str, Any]] = []
    for idx, sample in enumerate(samples):
        keep = idx < retain_limit
        local_path = Path(sample.video_path)
        local_exists = local_path.exists() and local_path.is_file()
        deleted = False
        if local_exists and not keep:
            deleted = _safe_unlink(local_path)

        rows.append(
            {
                "test_id": sample.test_id,
                "prompt": sample.prompt,
                "seed": sample.seed,
                "video_path": sample.video_path if keep else None,
                "artifact_retained": keep,
                "artifact_deleted": deleted,
                "metadata": sample.metadata,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Sample generation
# ---------------------------------------------------------------------------


def _build_sample_id(
    *,
    test_id: str,
    prompt: str,
    seed: int,
    video_cfg: dict[str, Any],
) -> str:
    payload = {
        "test_id": test_id,
        "prompt": normalize_prompt(prompt),
        "seed": int(seed),
        "video_cfg": video_cfg,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha1(encoded.encode("utf-8")).hexdigest()[:16]


def _iter_jobs(suite: SuiteSpec):
    """Yield (test, prompt, seed) tuples for all combinations in the suite."""
    for test in suite.tests:
        for prompt in test.prompts:
            for seed in test.seeds:
                yield test, prompt, seed


def _generate_samples(
    *,
    suite: SuiteSpec,
    adapter: Any,
    videos_dir: Path,
    sample_limit: int | None = None,
    progress_callback=None,
    workers: int = 1,
    retry: int = 1,
    inter_sample_delay: float = 0.0,
    adapter_timeout: float | None = None,
) -> tuple[list[GeneratedSample], int]:
    """Run the adapter for every (test, prompt, seed) combination.

    Returns ``(samples, skipped_count)`` where *skipped_count* is the number
    of jobs that failed permanently after all *retry* attempts.

    Parameters
    ----------
    workers:
        Number of parallel threads (default ``1`` = sequential).  When > 1
        the adapter must be thread-safe.
    retry:
        Maximum adapter call attempts per sample (default ``1`` = no retry).
        On permanent failure the sample is skipped and counted in
        *skipped_count*.
    inter_sample_delay:
        Seconds to wait between dispatching each job when *workers* > 1
        (default ``0``).  Set to e.g. ``3.0`` when using a GPU adapter to
        avoid saturating VRAM and freezing the system.
    adapter_timeout:
        Maximum seconds allowed for a single sample generation attempt.
        ``None`` (default) means no limit.  On timeout the sample is
        abandoned (daemon thread) and counted as skipped.
    progress_callback:
        Optional ``(current, total, test_id, prompt, seed)`` callable fired
        before each sample. Thread-safe when *workers* > 1.
    """
    import concurrent.futures
    import threading
    from itertools import islice

    jobs = list(islice(_iter_jobs(suite), sample_limit))
    total = len(jobs)

    _lock = threading.Lock()
    _started = 0

    def _fire_progress(test_id: str, prompt: str, seed: int) -> None:
        nonlocal _started
        if progress_callback is None:
            return
        with _lock:
            _started += 1
            current = _started
        try:
            progress_callback(current, total, test_id, prompt, seed)
        except Exception:  # noqa: BLE001
            pass

    def _run_one(test: Any, prompt: str, seed: int) -> GeneratedSample | None:
        _fire_progress(test.id, prompt, int(seed))
        effective_video_cfg = dict(test.video)
        if "encode" in suite.artifacts and "encode" not in effective_video_cfg:
            effective_video_cfg["encode"] = suite.artifacts["encode"]
        for _attempt in range(max(retry, 1)):
            try:
                sample = adapter.generate(
                    test_id=test.id,
                    prompt=prompt,
                    seed=seed,
                    video_cfg=effective_video_cfg,
                    output_dir=videos_dir,
                )
                sample_id = _build_sample_id(
                    test_id=test.id,
                    prompt=prompt,
                    seed=int(seed),
                    video_cfg=effective_video_cfg,
                )
                metadata = dict(sample.metadata)
                metadata.setdefault("sample_id", sample_id)
                sample.metadata = metadata
                return sample
            except Exception:  # noqa: BLE001
                pass
        return None  # all attempts exhausted

    def _run_one_timed(test: Any, prompt: str, seed: int) -> GeneratedSample | None:
        """Wrap _run_one with a per-sample wall-clock timeout.

        Uses a daemon thread so an overdue adapter call does not block the
        process indefinitely.  Returns ``None`` on timeout (counts as skipped).
        """
        import threading as _threading

        _result: list[GeneratedSample | None] = [None]
        _done = _threading.Event()

        def _target() -> None:
            _result[0] = _run_one(test, prompt, int(seed))
            _done.set()

        t = _threading.Thread(target=_target, daemon=True)
        t.start()
        finished = _done.wait(timeout=adapter_timeout)
        return _result[0] if finished else None

    _dispatch = _run_one_timed if (adapter_timeout is not None and adapter_timeout > 0) else _run_one

    if workers <= 1:
        raw: list[GeneratedSample | None] = [
            _dispatch(test, prompt, int(seed)) for test, prompt, seed in jobs
        ]
    else:
        import time as _time

        raw = [None] * len(jobs)
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            future_to_idx: dict[concurrent.futures.Future[Any], int] = {}
            for i, (test, prompt, seed) in enumerate(jobs):
                if i > 0 and inter_sample_delay > 0:
                    _time.sleep(inter_sample_delay)
                future_to_idx[pool.submit(_dispatch, test, prompt, int(seed))] = i
            for future in concurrent.futures.as_completed(future_to_idx):
                raw[future_to_idx[future]] = future.result()

    samples: list[GeneratedSample] = [s for s in raw if s is not None]
    skipped = sum(1 for s in raw if s is None)
    return samples, skipped


# ---------------------------------------------------------------------------
# Run artifact writing
# ---------------------------------------------------------------------------


def _dispatch_webhook(url: str, payload: dict[str, Any]) -> None:
    """Fire-and-forget POST to a webhook URL. Failures are non-fatal."""
    try:
        data = json.dumps(payload, default=str).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10):
            pass
    except Exception:  # noqa: BLE001
        pass


def _load_alert_state(model_root: Path) -> dict[str, Any]:
    """Load persistent alert state from model_root/alert_state.json."""
    data = read_json_dict(model_root / "alert_state.json")
    return data if data is not None else {"state": "passing"}


def _save_alert_state(model_root: Path, data: dict[str, Any]) -> None:
    try:
        atomic_write_json(model_root / "alert_state.json", data)
    except Exception:  # noqa: BLE001
        pass


def _maybe_dispatch_webhook(
    *,
    url: str,
    model_root: Path,
    run_id: str,
    timestamp: str,
    status: str,
    webhook_payload: dict[str, Any],
) -> None:
    """Dispatch webhook only on state transitions: new failure or recovery.

    Repeated failures do not re-fire; recovery fires once when status returns to PASS.
    Adds ``event_type`` field (``"new_failure"`` | ``"recovery"``) to the payload.
    """
    alert_state = _load_alert_state(model_root)
    prev_state = str(alert_state.get("state", "passing"))
    now_failing = status != "PASS"
    was_failing = prev_state == "failing"
    new_state = "failing" if now_failing else "passing"
    should_dispatch = (not was_failing and now_failing) or (was_failing and not now_failing)

    _save_alert_state(
        model_root,
        {
            "state": new_state,
            "last_run_id": run_id,
            "last_change_run_id": run_id if should_dispatch else alert_state.get("last_change_run_id"),
            "last_change_timestamp": timestamp if should_dispatch else alert_state.get("last_change_timestamp"),
        },
    )
    if should_dispatch:
        event_type = "new_failure" if now_failing else "recovery"
        _dispatch_webhook(url, {**webhook_payload, "event_type": event_type})


def _write_run_artifacts(
    *,
    run_dir: Path,
    model_root: Path,
    run_id: str,
    timestamp: str,
    status: str,
    samples: list[GeneratedSample],
    payload: dict[str, Any],
) -> None:
    """Persist run.json, report.html, latest_run.txt, and runs.jsonl."""
    run_json = run_dir / "run.json"
    run_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_html_report(run_dir / "report.html", payload)

    latest_file = model_root / "latest_run.txt"
    latest_file.write_text(run_id, encoding="utf-8")

    index_entry = {
        "run_id": run_id,
        "timestamp_utc": timestamp,
        "status": status,
        "sample_count": len(samples),
    }
    with (model_root / "runs.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(index_entry) + "\n")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_suite(
    *,
    suite: SuiteSpec,
    model_name: str | None = None,
    artifacts_dir: str | Path = "artifacts",
    fail_on_regression: bool = True,
    fail_on_skip: bool = False,
    baseline_mode: str = "latest_pass",
    webhook_url: str | None = None,
    sample_limit: int | None = None,
    tag: str | None = None,
    progress_callback=None,
    workers: int = 1,
    retry: int = 1,
    inter_sample_delay: float = 0.0,
    notify_on: str = "change",
    env: str | None = None,
    adapter_timeout: float | None = None,
) -> RunResult:
    """Execute a full suite run and return the result payload.

    Parameters
    ----------
    fail_on_regression:
        Treat regression vs baseline as FAIL (default ``True``).
    fail_on_skip:
        Treat any skipped sample (retry exhaustion) as FAIL (default ``False``).
    tag:
        If set, save this tag → run_id mapping in ``model_root/tags.json``
        after the run, enabling ``--baseline-mode tag:<name>`` in later runs.
    progress_callback:
        Optional callable ``(current, total, test_id, prompt, seed)`` fired
        before each sample is generated.  Useful for live progress display.
    workers:
        Number of parallel sample-generation threads (default ``1``).
    retry:
        Max adapter call attempts per sample before skipping (default ``1``).
    inter_sample_delay:
        Seconds between dispatching each job when *workers* > 1 (default
        ``0``).  Use e.g. ``3.0`` with GPU adapters to prevent VRAM
        saturation and system freezes.
    notify_on:
        When to fire the webhook.  ``"change"`` (default) fires only on state
        transitions (new failure / recovery).  ``"always"`` fires on every run.
    env:
        Optional environment label (e.g. ``"staging"``, ``"prod"``) stored in
        the run payload for tracking and filtering.  No effect on gate logic.
    adapter_timeout:
        Per-sample wall-clock timeout in seconds.  ``None`` = unlimited.
        Timed-out samples are counted as skipped (see *fail_on_skip*).
    """
    _validate_baseline_mode(baseline_mode)

    model = select_model(suite, model_name)
    adapter = build_adapter(model)

    timestamp = utc_now().isoformat()
    model_root = Path(artifacts_dir) / suite.project / suite.suite_name / model.name
    run_id, run_dir = _create_run_dir(model_root=model_root)
    videos_dir = run_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    # 1. Generate samples
    samples, skipped_count = _generate_samples(
        suite=suite,
        adapter=adapter,
        videos_dir=videos_dir,
        sample_limit=sample_limit,
        progress_callback=progress_callback,
        workers=workers,
        retry=retry,
        inter_sample_delay=inter_sample_delay,
        adapter_timeout=adapter_timeout,
    )

    # 2. Evaluate metrics
    metrics_payload: dict[str, Any] = {}
    for metric in suite.metrics:
        metric_params = dict(metric.params)
        if "keep_workdir" in suite.artifacts and "keep_workdir" not in metric_params:
            metric_params["keep_workdir"] = suite.artifacts["keep_workdir"]
        metrics_payload[metric.name] = run_metric(
            name=metric.name,
            samples=samples,
            params=metric_params,
        )

    # 3. Load baseline
    baseline = _load_previous_run(
        model_root=model_root, current_run_id=run_id, baseline_mode=baseline_mode
    )
    baseline_metrics = baseline.get("metrics") if isinstance(baseline, dict) else None
    baseline_run_id = baseline.get("run_id") if isinstance(baseline, dict) else None

    # 4. Evaluate gates
    gates = _evaluate_gates(
        suite.gates,
        metrics_payload,
        baseline_metrics=baseline_metrics if isinstance(baseline_metrics, dict) else None,
    )

    # 4b. Windowed gate override: suppress transient failures that haven't hit min_failures
    _max_window = max(
        (g.window for g in suite.gates if getattr(g, "window", 0) > 0),
        default=0,
    )
    if _max_window > 0:
        _recent = _load_recent_runs(model_root, current_run_id=run_id, n=_max_window - 1)
        gates = _apply_windowed_gates(suite.gates, gates, _recent)

    gate_failed = any(not gate.get("passed", False) for gate in gates)

    # 5. Regression comparison
    regressions = _compute_regressions(
        gates=gates, current_metrics=metrics_payload, baseline_metrics=baseline_metrics
    )
    regression_failed = any(item["regressed"] for item in regressions)

    # 6. Determine final status
    should_fail = (
        gate_failed
        or (fail_on_regression and regression_failed)
        or (fail_on_skip and skipped_count > 0)
    )
    status = "FAIL" if should_fail else "PASS"

    # 7. Apply retention policy
    sample_rows = _build_sample_rows_with_retention(
        samples=samples, status=status, artifacts_cfg=suite.artifacts
    )

    # 8. Build result payload
    git_meta = _capture_git_metadata()
    payload: dict[str, Any] = {
        "run_id": run_id,
        "timestamp_utc": timestamp,
        "project": suite.project,
        "suite_name": suite.suite_name,
        "model_name": model.name,
        "status": status,
        "sample_count": len(samples),
        "skipped_count": skipped_count,
        "metrics": metrics_payload,
        "gates": gates,
        "gate_failed": gate_failed,
        "regressions": regressions,
        "regression_failed": regression_failed,
        "baseline_run_id": baseline_run_id,
        "baseline_mode": baseline_mode,
        "artifacts_policy": suite.artifacts,
        "samples": sample_rows,
        **({"git": git_meta} if git_meta else {}),
        **({"env": env} if env else {}),
    }

    # 9. Write artifacts
    _write_run_artifacts(
        run_dir=run_dir,
        model_root=model_root,
        run_id=run_id,
        timestamp=timestamp,
        status=status,
        samples=samples,
        payload=payload,
    )

    payload["run_dir"] = str(run_dir)

    # 10. Auto badge — write model_root/badge.svg (non-fatal)
    try:
        write_badge_svg(model_root / "badge.svg", status)
    except Exception:  # noqa: BLE001
        pass

    # 10b. Tag — persist tag → run_id mapping (non-fatal)
    if tag:
        try:
            _write_tag(model_root, tag, run_id)
        except Exception:  # noqa: BLE001
            pass

    # 11. Auto trend report — regenerate model-level trend_report.html (non-fatal)
    try:
        trend_runs = load_model_runs(model_root)
        if len(trend_runs) >= 2:
            write_trend_report(model_root / "trend_report.html", trend_runs)
    except Exception:  # noqa: BLE001
        pass

    # 12. Auto compare report — write compare_report.html into run dir when baseline exists (non-fatal)
    if baseline is not None:
        try:
            write_compare_report(run_dir / "compare_report.html", baseline, payload)
        except Exception:  # noqa: BLE001
            pass

    # 13. Auto suite index (non-fatal)
    try:
        suite_root = Path(artifacts_dir) / suite.project / suite.suite_name
        write_suite_index(suite_root, project=suite.project, suite_name=suite.suite_name)
    except Exception:  # noqa: BLE001
        pass

    # 14. Webhook — fire on state transitions (default) or every run (notify_on="always")
    if webhook_url:
        _wh_payload = {
            "run_id": run_id,
            "status": status,
            "project": suite.project,
            "suite_name": suite.suite_name,
            "model_name": model.name,
            "timestamp_utc": timestamp,
            "gate_failed": gate_failed,
            "regression_failed": regression_failed,
            "run_dir": str(run_dir),
            "gate_failures": [
                {k: v for k, v in g.items() if k not in ("sprt", "llr_history")}
                for g in gates if not g.get("passed")
            ],
            "top_regressions": [
                r for r in regressions if r.get("regressed")
            ][:5],
        }
        try:
            if notify_on == "always":
                _dispatch_webhook(webhook_url, {
                    **_wh_payload,
                    "event_type": "failure" if status != "PASS" else "pass",
                })
            else:
                _maybe_dispatch_webhook(
                    url=webhook_url,
                    model_root=model_root,
                    run_id=run_id,
                    timestamp=timestamp,
                    status=status,
                    webhook_payload=_wh_payload,
                )
        except Exception:  # noqa: BLE001
            pass

    return payload  # type: ignore[return-value]
