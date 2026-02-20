from __future__ import annotations

import argparse
import json
import os as _os
from pathlib import Path
from typing import Any, cast

from temporalci.adapters import available_adapters
from temporalci.constants import BASELINE_MODES
from temporalci.config import SuiteValidationError  # ConfigError alias
from temporalci.config import load_suite
from temporalci.engine import run_suite
from temporalci.metrics import available_metrics
from temporalci.sprt_calibration import sprt_main
from temporalci.trend import load_model_runs, write_trend_report
from temporalci._cli_impl import (
    _cmd_alert,
    _cmd_annotate,
    _cmd_compare,
    _cmd_doctor,
    _cmd_export,
    _cmd_history,
    _cmd_init,
    _cmd_metrics_show,
    _cmd_prune,
    _cmd_repair_index,
    _cmd_report,
    _cmd_status,
    _cmd_summary,
    _cmd_tag,
    _cmd_tune_gates,
    _print_summary,
)


def _build_parser(config: dict[str, Any] | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="temporalci", description="TemporalCI CLI")
    parser.add_argument(
        "--config",
        default=None,
        metavar="PATH",
        help="Project config file (default: auto-detect .temporalci.yaml in CWD)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_cmd = sub.add_parser("run", help="Run a suite")
    run_cmd.add_argument("suite", help="Path to suite yaml")
    run_cmd.add_argument("--model", help="Model name in suite", default=None)
    run_cmd.add_argument("--artifacts-dir", default="artifacts", help="Artifact root directory")
    run_cmd.add_argument(
        "--ignore-regression",
        action="store_true",
        help="Do not fail CI on regression against previous run",
    )
    run_cmd.add_argument(
        "--baseline-mode",
        default="latest_pass",
        metavar="MODE",
        help=(
            f"How baseline run is chosen. Built-in: {sorted(BASELINE_MODES)}. "
            "Also accepts 'tag:<name>' to use a tagged run."
        ),
    )
    run_cmd.add_argument(
        "--tag",
        default=None,
        metavar="NAME",
        help="Save this tag → run_id mapping after the run (for use with --baseline-mode tag:<name>)",
    )
    run_cmd.add_argument(
        "--print-json",
        action="store_true",
        help="Print full run payload as JSON",
    )
    run_cmd.add_argument(
        "--no-progress",
        action="store_true",
        help="Suppress live sample-generation progress output",
    )
    run_cmd.add_argument(
        "--webhook-url",
        default=None,
        metavar="URL",
        help="HTTP endpoint to POST on gate/regression failure (Slack, Discord, etc.)",
    )
    run_cmd.add_argument(
        "--prune-keep-last",
        type=int,
        default=None,
        metavar="N",
        help="After run, automatically prune old runs keeping last N (default: no pruning)",
    )
    run_cmd.add_argument(
        "--sample",
        type=int,
        default=None,
        metavar="N",
        dest="sample_limit",
        help="Run only the first N samples (smoke-test mode, default: all)",
    )
    run_cmd.add_argument(
        "--workers",
        type=int,
        default=1,
        metavar="N",
        help="Number of parallel sample-generation threads (default: 1 = sequential)",
    )
    run_cmd.add_argument(
        "--retry",
        type=int,
        default=1,
        metavar="N",
        help="Max adapter call attempts per sample before skipping it (default: 1 = no retry)",
    )
    run_cmd.add_argument(
        "--fail-on-skip",
        action="store_true",
        help="Treat any skipped sample (retry exhaustion) as run FAIL",
    )
    run_cmd.add_argument(
        "--inter-sample-delay",
        type=float,
        default=0.0,
        metavar="SECONDS",
        dest="inter_sample_delay",
        help=(
            "Seconds to sleep between dispatching each job when --workers > 1 "
            "(default: 0). Use e.g. 3.0 with GPU adapters to avoid VRAM saturation."
        ),
    )
    run_cmd.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help=(
            "Validate suite + adapter and print expected sample counts "
            "without generating any videos or writing artifacts."
        ),
    )
    run_cmd.add_argument(
        "--output-json",
        default=None,
        metavar="PATH",
        dest="output_json",
        help="Write run payload(s) to a JSON file after completion.",
    )
    run_cmd.add_argument(
        "--fail-fast",
        action="store_true",
        dest="fail_fast",
        help="Stop multi-model run immediately after the first model failure.",
    )
    run_cmd.add_argument(
        "--model-workers",
        type=int,
        default=1,
        metavar="N",
        dest="model_workers",
        help=(
            "Number of models to run in parallel (default: 1 = sequential). "
            "Each model still uses --workers threads for its own sample generation. "
            "Progress output is suppressed when > 1."
        ),
    )
    run_cmd.add_argument(
        "--notify-on",
        choices=["change", "always"],
        default="change",
        dest="notify_on",
        help=(
            "When to fire the webhook: 'change' (default) fires only on "
            "new failure / recovery transitions; 'always' fires on every run."
        ),
    )
    run_cmd.add_argument(
        "--ci",
        action="store_true",
        help=(
            "Enable CI mode: suppress progress, add GitHub Actions annotations "
            "(auto-detected from CI / GITHUB_ACTIONS env vars)."
        ),
    )
    run_cmd.add_argument(
        "--env",
        default=None,
        metavar="NAME",
        help="Environment label stored in run payload (e.g. 'staging', 'prod')",
    )
    run_cmd.add_argument(
        "--adapter-timeout",
        type=float,
        default=None,
        metavar="SECONDS",
        dest="adapter_timeout",
        help=(
            "Per-sample generation timeout in seconds.  "
            "Timed-out samples are skipped (use --fail-on-skip to treat as FAIL)."
        ),
    )
    run_cmd.add_argument(
        "--include-model",
        action="append",
        default=None,
        metavar="NAME",
        dest="include_models",
        help="Only run this model (repeatable; ignored if --model is set)",
    )
    run_cmd.add_argument(
        "--exclude-model",
        action="append",
        default=None,
        metavar="NAME",
        dest="exclude_models",
        help="Skip this model (repeatable; ignored if --model is set)",
    )
    run_cmd.add_argument(
        "--gate-override",
        action="append",
        default=None,
        metavar="SPEC",
        dest="gate_overrides",
        help=(
            "Override a gate threshold without editing YAML: 'METRIC OP VALUE' "
            "(e.g. 'vbench_temporal.score >= 0.65'). Repeatable. "
            "Existing gates with matching metric+op have their value replaced; "
            "non-matching specs are added as new gates."
        ),
    )
    run_cmd.add_argument(
        "--watch",
        type=float,
        default=None,
        metavar="SECONDS",
        help=("Continuous mode: re-run the suite every SECONDS seconds. Press Ctrl+C to stop."),
    )

    # Apply .temporalci.yaml defaults to run subcommand (CLI args still override)
    _cfg_run = (config or {}).get("run", {})
    _RUN_CFG_KEYS: dict[str, type] = {
        "workers": int,
        "retry": int,
        "inter_sample_delay": float,
        "artifacts_dir": str,
        "baseline_mode": str,
        "webhook_url": str,
        "tag": str,
        "sample_limit": int,
        "prune_keep_last": int,
        "fail_on_skip": bool,
        "fail_fast": bool,
        "no_progress": bool,
        "model_workers": int,
        "notify_on": str,
        "env": str,
    }
    _run_defaults: dict[str, Any] = {}
    for _k, _typ in _RUN_CFG_KEYS.items():
        if _k in _cfg_run and _cfg_run[_k] is not None:
            try:
                _run_defaults[_k] = _typ(_cfg_run[_k])
            except (TypeError, ValueError):
                pass
    if _run_defaults:
        run_cmd.set_defaults(**_run_defaults)

    validate_cmd = sub.add_parser("validate", help="Validate a suite file")
    validate_cmd.add_argument("suite", help="Path to suite yaml")

    list_cmd = sub.add_parser("list", help="List available adapters and metrics")
    list_cmd.add_argument(
        "--json",
        action="store_true",
        help="Print as JSON",
    )

    sprt_cmd = sub.add_parser("sprt", help="SPRT calibration/apply/check utilities")
    sprt_cmd.add_argument(
        "sprt_args",
        nargs=argparse.REMAINDER,
        help="Pass-through args for 'calibrate|apply|check'",
    )

    status_cmd = sub.add_parser(
        "status", help="Show recent run history for a model or suite in the terminal"
    )
    status_target = status_cmd.add_mutually_exclusive_group(required=True)
    status_target.add_argument(
        "--model-root",
        default=None,
        metavar="DIR",
        help="Model artifact directory (e.g. artifacts/project/suite/model)",
    )
    status_target.add_argument(
        "--suite-root",
        default=None,
        metavar="DIR",
        help="Suite artifact directory to show all models (e.g. artifacts/project/suite)",
    )
    status_cmd.add_argument(
        "--last-n",
        type=int,
        default=10,
        metavar="N",
        help="Number of recent runs to show (default: 10)",
    )
    status_cmd.add_argument(
        "--verbose",
        action="store_true",
        help="Show all metric paths including nested dims (default: top-level only)",
    )
    status_cmd.add_argument(
        "--output-format",
        choices=["text", "json", "csv"],
        default="text",
        dest="output_format",
        help="Output format: text (default), json, or csv",
    )
    status_cmd.add_argument(
        "--output",
        default=None,
        metavar="PATH",
        help="Output file path (required for --output-format csv)",
    )

    compare_cmd = sub.add_parser(
        "compare",
        help="Compare two runs side-by-side (explicit or auto from --model-root)",
    )
    compare_cmd.add_argument(
        "run_a",
        nargs="?",
        default=None,
        metavar="RUN_A",
        help="Run-ID mode: first (baseline) run ID (requires --model-root)",
    )
    compare_cmd.add_argument(
        "run_b",
        nargs="?",
        default=None,
        metavar="RUN_B",
        help="Run-ID mode: second (candidate) run ID (requires --model-root)",
    )
    compare_cmd.add_argument(
        "--model-root",
        default=None,
        metavar="DIR",
        help="Auto mode: pick latest and latest-pass runs from this model directory",
    )
    compare_cmd.add_argument(
        "--baseline",
        default=None,
        metavar="PATH",
        help="Explicit mode: path to baseline run.json",
    )
    compare_cmd.add_argument(
        "--candidate",
        default=None,
        metavar="PATH",
        help="Explicit mode: path to candidate run.json",
    )
    compare_cmd.add_argument(
        "--output",
        default="compare_report.html",
        metavar="PATH",
        help="Output HTML path (default: compare_report.html)",
    )

    prune_cmd = sub.add_parser("prune", help="Delete old run directories to free disk space")
    prune_target = prune_cmd.add_mutually_exclusive_group(required=True)
    prune_target.add_argument(
        "--model-root",
        default=None,
        metavar="DIR",
        help="Model artifact directory (e.g. artifacts/project/suite/model)",
    )
    prune_target.add_argument(
        "--suite-root",
        default=None,
        metavar="DIR",
        help="Suite directory — prune all models inside (e.g. artifacts/project/suite)",
    )
    prune_cmd.add_argument(
        "--keep-last",
        type=int,
        default=20,
        metavar="N",
        help="Number of most recent runs to keep per model (default: 20)",
    )
    prune_cmd.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be deleted without actually deleting",
    )

    trend_cmd = sub.add_parser("trend", help="Generate a trend report from run history")
    trend_cmd.add_argument(
        "--model-root",
        required=True,
        metavar="DIR",
        help="Model artifact directory (e.g. artifacts/project/suite/model)",
    )
    trend_cmd.add_argument(
        "--output",
        default="trend_report.html",
        metavar="PATH",
        help="Output HTML path (default: trend_report.html)",
    )
    trend_cmd.add_argument(
        "--last-n",
        type=int,
        default=30,
        metavar="N",
        help="Number of recent runs to include (default: 30)",
    )
    trend_cmd.add_argument(
        "--title",
        default="TemporalCI Trend Report",
        metavar="TITLE",
        help="Report title",
    )

    export_cmd = sub.add_parser(
        "export", help="Export run history to CSV or JSONL for external analysis"
    )
    export_target = export_cmd.add_mutually_exclusive_group(required=True)
    export_target.add_argument(
        "--model-root",
        default=None,
        metavar="DIR",
        help="Single model directory (e.g. artifacts/project/suite/model)",
    )
    export_target.add_argument(
        "--suite-root",
        default=None,
        metavar="DIR",
        help="Suite directory — export all models with a model_name column",
    )
    export_cmd.add_argument(
        "--output",
        required=True,
        metavar="PATH",
        help="Output file path (.csv or .jsonl)",
    )
    export_cmd.add_argument(
        "--format",
        choices=["csv", "jsonl"],
        default="csv",
        dest="fmt",
        help="Output format: csv (default) or jsonl",
    )
    export_cmd.add_argument(
        "--last-n",
        type=int,
        default=0,
        metavar="N",
        help="Export only last N runs (default: all)",
    )

    report_cmd = sub.add_parser(
        "report", help="Regenerate HTML report(s) from existing run.json files"
    )
    report_target = report_cmd.add_mutually_exclusive_group(required=True)
    report_target.add_argument(
        "--run-dir",
        default=None,
        metavar="DIR",
        help="Single run directory containing run.json",
    )
    report_target.add_argument(
        "--model-root",
        default=None,
        metavar="DIR",
        help="Regenerate reports for all runs under this model directory",
    )
    report_cmd.add_argument(
        "--output",
        default=None,
        metavar="PATH",
        help="Output report path (default: report.html inside run-dir)",
    )

    init_cmd = sub.add_parser("init", help="Scaffold a new suite.yaml with sensible defaults")
    init_cmd.add_argument(
        "--project",
        default="my-project",
        metavar="NAME",
        help="Project name (default: my-project)",
    )
    init_cmd.add_argument(
        "--adapter",
        default="mock",
        metavar="ADAPTER",
        help="Adapter name (default: mock). Run 'temporalci list' to see available.",
    )
    init_cmd.add_argument(
        "--metric",
        default="vbench_temporal",
        metavar="METRIC",
        help="Metric name (default: vbench_temporal).",
    )
    init_cmd.add_argument(
        "--output",
        default="suite.yaml",
        metavar="PATH",
        help="Output path (default: suite.yaml)",
    )
    init_cmd.add_argument(
        "--force",
        action="store_true",
        help="Overwrite output file if it already exists",
    )

    annotate_cmd = sub.add_parser("annotate", help="Attach a text note to an existing run")
    annotate_cmd.add_argument(
        "--model-root",
        required=True,
        metavar="DIR",
        help="Model artifact directory (e.g. artifacts/project/suite/model)",
    )
    annotate_cmd.add_argument(
        "--run-id",
        required=True,
        metavar="ID",
        help="Run ID to annotate (directory name under model-root)",
    )
    annotate_cmd.add_argument(
        "--note",
        required=True,
        metavar="TEXT",
        help="Annotation text to attach to the run",
    )

    repair_cmd = sub.add_parser(
        "repair-index",
        help="Rebuild runs.jsonl from existing run.json files (use after manual pruning or corruption)",
    )
    repair_cmd.add_argument(
        "--model-root",
        required=True,
        metavar="DIR",
        help="Model artifact directory (e.g. artifacts/project/suite/model)",
    )
    repair_cmd.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be written without modifying runs.jsonl",
    )

    sub.add_parser("doctor", help="Print an environment diagnostic summary")

    tag_cmd = sub.add_parser("tag", help="Manage baseline tags for a model")
    tag_cmd.add_argument(
        "action",
        choices=["list", "show", "delete", "set"],
        help="list all tags · show a tag's run ID · delete a tag · set a tag to a run ID",
    )
    tag_cmd.add_argument(
        "--model-root",
        required=True,
        metavar="DIR",
        help="Model artifact directory (e.g. artifacts/project/suite/model)",
    )
    tag_cmd.add_argument(
        "--name",
        default=None,
        metavar="NAME",
        help="Tag name (required for show, delete, set)",
    )
    tag_cmd.add_argument(
        "--run-id",
        default=None,
        metavar="ID",
        dest="run_id",
        help="Run ID to associate with the tag (required for set)",
    )

    alert_cmd = sub.add_parser(
        "alert",
        help="Exit non-zero if the model (or any model in a suite) is currently failing",
    )
    alert_target = alert_cmd.add_mutually_exclusive_group(required=True)
    alert_target.add_argument(
        "--model-root",
        default=None,
        metavar="DIR",
        help="Model artifact directory",
    )
    alert_target.add_argument(
        "--suite-root",
        default=None,
        metavar="DIR",
        help="Suite directory — check all models inside",
    )

    metrics_show_cmd = sub.add_parser(
        "metrics-show",
        help="Show detailed metrics, gates and regressions for a specific run",
    )
    metrics_show_cmd.add_argument(
        "--model-root",
        required=True,
        metavar="DIR",
        help="Model artifact directory (e.g. artifacts/project/suite/model)",
    )
    metrics_show_cmd.add_argument(
        "--run-id",
        default=None,
        metavar="ID",
        dest="run_id",
        help="Run ID to inspect (default: latest run)",
    )
    metrics_show_cmd.add_argument(
        "--verbose",
        action="store_true",
        help="Show per-sample data and full SPRT details",
    )

    tune_cmd = sub.add_parser(
        "tune-gates",
        help="Suggest gate thresholds from historical PASS-run metrics",
    )
    tune_cmd.add_argument(
        "--model-root",
        required=True,
        metavar="DIR",
        help="Model artifact directory (e.g. artifacts/project/suite/model)",
    )
    tune_cmd.add_argument(
        "--percentile",
        type=float,
        default=5.0,
        metavar="PCT",
        help="Use this percentile of PASS-run scores as the threshold (default: 5)",
    )
    tune_cmd.add_argument(
        "--last-n",
        type=int,
        default=30,
        metavar="N",
        help="Only consider the last N runs (default: 30)",
    )
    tune_cmd.add_argument(
        "--metric",
        default=None,
        metavar="NAME",
        help="Only tune this metric (default: all metrics found in history)",
    )

    summary_cmd = sub.add_parser(
        "summary",
        help="Show status overview for all projects / suites / models under an artifacts directory",
    )
    summary_cmd.add_argument(
        "--artifacts-dir",
        required=True,
        metavar="DIR",
        help="Artifact root directory (e.g. 'artifacts')",
    )
    summary_cmd.add_argument(
        "--output-format",
        choices=["text", "json"],
        default="text",
        dest="output_format",
        help="Output format: text (default) or json",
    )

    history_cmd = sub.add_parser(
        "history",
        help="Show run history with optional filtering by status or date",
    )
    history_cmd.add_argument(
        "--model-root",
        required=True,
        metavar="DIR",
        help="Model artifact directory (e.g. artifacts/project/suite/model)",
    )
    history_cmd.add_argument(
        "--last-n",
        type=int,
        default=20,
        metavar="N",
        help="Maximum number of runs to show (default: 20)",
    )
    history_cmd.add_argument(
        "--status",
        choices=["PASS", "FAIL", "all"],
        default="all",
        help="Filter by run status: PASS, FAIL, or all (default: all)",
    )
    history_cmd.add_argument(
        "--since",
        default=None,
        metavar="DATE",
        help="Show only runs on or after this date (YYYY-MM-DD)",
    )
    history_cmd.add_argument(
        "--output-format",
        choices=["text", "json"],
        default="text",
        dest="output_format",
        help="Output format: text (default) or json",
    )

    heatmap_cmd = sub.add_parser(
        "heatmap",
        help="Generate vegetation detection heatmap overlays for inspection frames",
    )
    heatmap_cmd.add_argument(
        "--frame-dir",
        required=True,
        metavar="DIR",
        help="Directory containing source frames",
    )
    heatmap_cmd.add_argument(
        "--output-dir",
        required=True,
        metavar="DIR",
        help="Output directory for heatmap PNGs",
    )
    heatmap_cmd.add_argument(
        "--pattern",
        default="*.jpg",
        help="Glob pattern for frame files (default: *.jpg)",
    )
    heatmap_cmd.add_argument(
        "--alpha",
        type=float,
        default=0.45,
        help="Overlay opacity 0-1 (default: 0.45)",
    )

    insp_cmd = sub.add_parser(
        "inspection-report",
        help="Generate an HTML inspection report with embedded heatmap thumbnails",
    )
    insp_cmd.add_argument(
        "--run-dir",
        required=True,
        metavar="DIR",
        help="Run artifact directory containing run.json",
    )
    insp_cmd.add_argument(
        "--frame-dir",
        required=True,
        metavar="DIR",
        help="Directory containing source frames",
    )
    insp_cmd.add_argument(
        "--output",
        default=None,
        metavar="PATH",
        help="Output HTML path (default: <run-dir>/inspection_report.html)",
    )
    insp_cmd.add_argument(
        "--pattern",
        default="*.jpg",
        help="Glob pattern for frame files (default: *.jpg)",
    )

    # ── clearance ──────────────────────────────────────────────────────
    clr_cmd = sub.add_parser(
        "clearance",
        help="Run 3-layer vision pipeline (segmentation + depth + wire detection)",
    )
    clr_cmd.add_argument(
        "frame_dir",
        help="Directory containing inspection frames",
    )
    clr_cmd.add_argument(
        "--output-dir",
        default="clearance_output",
        help="Output directory for multi-panel visualizations (default: clearance_output)",
    )
    clr_cmd.add_argument(
        "--pattern",
        default="*.jpg",
        help="Glob pattern for frame files (default: *.jpg)",
    )
    clr_cmd.add_argument(
        "--device",
        default="auto",
        help="Inference device: auto, cpu, or cuda (default: auto)",
    )
    clr_cmd.add_argument(
        "--skip-depth",
        action="store_true",
        help="Skip depth estimation (faster, 2-layer mode)",
    )
    clr_cmd.add_argument(
        "--json",
        action="store_true",
        dest="print_json",
        help="Print results as JSON",
    )

    # ── inspect (video → frames → analysis) ─────────────────────────
    inspect_cmd = sub.add_parser(
        "inspect",
        help="Process a video file: extract frames and run 3-layer vision pipeline",
    )
    inspect_cmd.add_argument(
        "video",
        help="Path to input video file",
    )
    inspect_cmd.add_argument(
        "--output-dir",
        default="inspection_output",
        help="Output directory for frames and results (default: inspection_output)",
    )
    inspect_cmd.add_argument(
        "--fps",
        type=float,
        default=1.0,
        help="Frame extraction rate in frames per second (default: 1.0)",
    )
    inspect_cmd.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to extract (default: all)",
    )
    inspect_cmd.add_argument(
        "--device",
        default="auto",
        help="Inference device: auto, cpu, or cuda (default: auto)",
    )
    inspect_cmd.add_argument(
        "--skip-depth",
        action="store_true",
        help="Skip depth estimation (faster, 2-layer mode)",
    )
    inspect_cmd.add_argument(
        "--json",
        action="store_true",
        dest="print_json",
        help="Print results as JSON",
    )

    route_map_cmd = sub.add_parser(
        "route-map",
        help="Generate an interactive Leaflet.js route map from inspection run results",
    )
    route_map_cmd.add_argument(
        "--run-dir",
        required=True,
        metavar="DIR",
        help="Run artifact directory containing run.json",
    )
    route_map_cmd.add_argument(
        "--output",
        default="route_map.html",
        metavar="PATH",
        help="Output HTML path (default: route_map.html)",
    )
    route_map_cmd.add_argument(
        "--title",
        default="Catenary Inspection Route Map",
        metavar="TITLE",
        help="Map title (default: Catenary Inspection Route Map)",
    )

    # ── temporal-diff ──────────────────────────────────────────────────
    tdiff_cmd = sub.add_parser(
        "temporal-diff",
        help="Compare two inspection runs to detect vegetation change over time",
    )
    tdiff_cmd.add_argument(
        "--before",
        required=True,
        metavar="DIR",
        help="Run directory for the earlier (before) inspection",
    )
    tdiff_cmd.add_argument(
        "--after",
        required=True,
        metavar="DIR",
        help="Run directory for the later (after) inspection",
    )
    tdiff_cmd.add_argument(
        "--output",
        default="temporal_diff.html",
        metavar="PATH",
        help="Output HTML path (default: temporal_diff.html)",
    )
    tdiff_cmd.add_argument(
        "--before-label",
        default="Before",
        metavar="LABEL",
        help="Label for the before run (e.g. '2025-06 Summer')",
    )
    tdiff_cmd.add_argument(
        "--after-label",
        default="After",
        metavar="LABEL",
        help="Label for the after run (e.g. '2025-12 Winter')",
    )
    tdiff_cmd.add_argument(
        "--json",
        action="store_true",
        dest="print_json",
        help="Print results as JSON",
    )

    # ── dashboard ──────────────────────────────────────────────────────
    dash_cmd = sub.add_parser(
        "dashboard",
        help="Generate an interactive HTML dashboard from inspection results",
    )
    dash_cmd.add_argument(
        "--run-dir",
        required=True,
        metavar="DIR",
        help="Run artifact directory containing run.json",
    )
    dash_cmd.add_argument(
        "--output",
        default="dashboard.html",
        metavar="PATH",
        help="Output HTML path (default: dashboard.html)",
    )
    dash_cmd.add_argument(
        "--title",
        default="Catenary Inspection Dashboard",
        metavar="TITLE",
        help="Dashboard title",
    )

    # ── km-report ──────────────────────────────────────────────────
    km_report_cmd = sub.add_parser(
        "km-report",
        help="Generate km-based risk report from multi-camera fusion results",
    )
    km_report_cmd.add_argument(
        "--run-dir",
        required=True,
        metavar="DIR",
        help="Run artifact directory containing run.json",
    )
    km_report_cmd.add_argument(
        "--output",
        default="km_report.html",
        metavar="PATH",
        help="Output HTML path (default: km_report.html)",
    )
    km_report_cmd.add_argument(
        "--bin-size",
        type=float,
        default=0.5,
        metavar="KM",
        dest="bin_size",
        help="Km bin size for aggregation (default: 0.5)",
    )
    km_report_cmd.add_argument(
        "--title",
        default="Km-based Inspection Report",
        metavar="TITLE",
        help="Report title",
    )
    km_report_cmd.add_argument(
        "--budget",
        type=float,
        default=None,
        metavar="KM",
        help="Maintenance budget in km (prints priority list when set)",
    )
    km_report_cmd.add_argument(
        "--json",
        action="store_true",
        dest="print_json",
        help="Print aggregated km data as JSON instead of generating HTML",
    )

    serve_cmd = sub.add_parser(
        "serve",
        help="Launch live web dashboard for real-time video inspection",
    )
    serve_cmd.add_argument(
        "--video",
        required=True,
        metavar="PATH",
        help="Path to input video file",
    )
    serve_cmd.add_argument(
        "--port",
        type=int,
        default=8421,
        metavar="PORT",
        help="HTTP port (default: 8421)",
    )
    serve_cmd.add_argument(
        "--host",
        default="0.0.0.0",
        metavar="HOST",
        help="Bind address (default: 0.0.0.0)",
    )
    serve_cmd.add_argument(
        "--fps",
        type=float,
        default=1.0,
        help="Extraction FPS (default: 1.0)",
    )
    serve_cmd.add_argument(
        "--max-frames",
        type=int,
        default=0,
        dest="max_frames",
        help="Max frames to process (0=all)",
    )
    serve_cmd.add_argument("--device", default="auto", help="Torch device")
    serve_cmd.add_argument(
        "--skip-depth",
        action="store_true",
        dest="skip_depth",
        help="Skip depth estimation (faster)",
    )
    serve_cmd.add_argument(
        "--output-dir",
        default="serve_output",
        dest="output_dir",
        help="Output directory for frames/panels (default: serve_output)",
    )

    return parser


def _apply_gate_overrides(suite: Any, overrides: list[str]) -> Any:
    """Parse 'METRIC OP VALUE' override strings and patch suite.gates.

    Existing gates whose ``metric`` and ``op`` match are value-replaced.
    Non-matching specs are appended as new gates.
    Returns a new SuiteSpec (original is unchanged).
    """
    import dataclasses
    from temporalci.types import GateSpec

    new_gates = list(suite.gates)
    for spec_str in overrides:
        parts = spec_str.strip().split(None, 2)
        if len(parts) != 3:
            raise ValueError(f"--gate-override must be 'METRIC OP VALUE', got: {spec_str!r}")
        metric, op, value_str = parts
        try:
            value: Any = float(value_str)
        except ValueError:
            value = value_str

        matched = False
        for i, gate in enumerate(new_gates):
            if gate.metric == metric and gate.op == op:
                new_gates[i] = dataclasses.replace(gate, value=value)
                matched = True
                break
        if not matched:
            new_gates.append(GateSpec(metric=metric, op=op, value=value))

    return dataclasses.replace(suite, gates=new_gates)


def _load_project_config(config_path: str | None = None) -> dict[str, Any]:
    """Load CLI defaults from .temporalci.yaml (or *config_path*)."""
    import yaml as _yaml

    if config_path is not None:
        path = Path(config_path)
        if not path.exists():
            return {}
        try:
            data = _yaml.safe_load(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:  # noqa: BLE001
            return {}

    for name in (".temporalci.yaml", ".temporalci.yml"):
        path = Path(name)
        if path.exists():
            try:
                data = _yaml.safe_load(path.read_text(encoding="utf-8"))
                return data if isinstance(data, dict) else {}
            except Exception:  # noqa: BLE001
                return {}
    return {}


def main(argv: list[str] | None = None) -> int:
    # Pre-parse to find --config before building the full parser with config defaults
    _pre = argparse.ArgumentParser(add_help=False)
    _pre.add_argument("--config", default=None)
    _pre_args, _ = _pre.parse_known_args(argv)
    project_cfg = _load_project_config(_pre_args.config)

    parser = _build_parser(project_cfg)
    args = parser.parse_args(argv)

    if args.command == "list":
        payload = {
            "adapters": available_adapters(),
            "metrics": available_metrics(),
        }
        if args.json:
            print(json.dumps(payload, indent=2))
        else:
            print("adapters:", ", ".join(payload["adapters"]))
            print("metrics:", ", ".join(payload["metrics"]))
        return 0

    if args.command == "validate":
        try:
            suite = load_suite(Path(args.suite))
        except SuiteValidationError as exc:
            print(f"config error: {exc}")
            return 1
        print(
            f"valid suite: project={suite.project} suite={suite.suite_name} "
            f"models={len(suite.models)} tests={len(suite.tests)}"
        )
        return 0

    if args.command == "sprt":
        if not args.sprt_args:
            print("usage: temporalci sprt <calibrate|apply|check> ...")
            return 1
        try:
            return sprt_main(args.sprt_args)
        except Exception as exc:  # noqa: BLE001
            print(f"runtime error: {exc}")
            return 1

    if args.command == "status":
        return _cmd_status(args)

    if args.command == "compare":
        return _cmd_compare(args)

    if args.command == "prune":
        return _cmd_prune(args)

    if args.command == "export":
        return _cmd_export(args)

    if args.command == "init":
        return _cmd_init(args)

    if args.command == "annotate":
        return _cmd_annotate(args)

    if args.command == "report":
        return _cmd_report(args)

    if args.command == "repair-index":
        return _cmd_repair_index(args)

    if args.command == "alert":
        return _cmd_alert(args)

    if args.command == "doctor":
        return _cmd_doctor(args)

    if args.command == "tag":
        return _cmd_tag(args)

    if args.command == "history":
        return _cmd_history(args)

    if args.command == "tune-gates":
        return _cmd_tune_gates(args)

    if args.command == "summary":
        return _cmd_summary(args)

    if args.command == "metrics-show":
        return _cmd_metrics_show(args)

    if args.command == "inspection-report":
        try:
            from temporalci.inspection_report import write_inspection_report

            run_dir = Path(args.run_dir)
            run_json = run_dir / "run.json"
            if not run_json.exists():
                print(f"run.json not found in {run_dir}")
                return 1
            run_data = json.loads(run_json.read_text(encoding="utf-8"))
            output = Path(args.output) if args.output else run_dir / "inspection_report.html"
            write_inspection_report(
                output,
                run_data=run_data,
                frame_dir=args.frame_dir,
                pattern=args.pattern,
            )
            print(f"inspection report: {output}")
            return 0
        except Exception as exc:  # noqa: BLE001
            print(f"runtime error: {exc}")
            return 1

    if args.command == "heatmap":
        try:
            from temporalci.heatmap import generate_heatmaps

            results = generate_heatmaps(
                args.frame_dir,
                args.output_dir,
                pattern=args.pattern,
                overlay_alpha=args.alpha,
            )
            for r in results:
                prox = r["green_ratio_quarter"]
                cov = r["green_ratio_half"]
                print(f"  {r['source_frame']:20s}  prox={prox:.4f}  cov={cov:.4f}")
            print(f"\n{len(results)} heatmaps written to {args.output_dir}")
            return 0
        except Exception as exc:  # noqa: BLE001
            print(f"runtime error: {exc}")
            return 1

    if args.command == "inspect":
        try:
            from temporalci.vision.video import process_video

            video_path = Path(args.video)
            if not video_path.is_file():
                print(f"error: video file not found: {video_path}")
                return 1

            result = process_video(
                video_path,
                output_dir=args.output_dir,
                fps=args.fps,
                max_frames=args.max_frames,
                device=args.device,
                skip_depth=args.skip_depth,
            )

            if args.print_json:
                print(json.dumps(result, indent=2, default=str))
            else:
                meta = result.get("_meta", {})
                print(f"\n{'=' * 60}")
                print(f"  Video: {meta.get('video_path', args.video)}")
                print(f"  Frames extracted: {meta.get('frames_extracted', '?')}")
                print(f"  Score: {result.get('score', 0.0):.4f}")
                print(f"  Samples: {result.get('sample_count', 0)}")
                for dim, val in result.get("dims", {}).items():
                    print(f"  {dim}: {val:.6f}")
                alerts = result.get("alert_frames", [])
                print(f"  Alert frames: {len(alerts)}")
                for a in alerts:
                    print(
                        f"    {a['prompt']:20s}  risk={a['risk_level']}  "
                        f"score={a['risk_score']:.2f}  clearance={a['clearance_px']:.0f}px"
                    )
                print(f"{'=' * 60}")
                print(f"Output directory: {args.output_dir}/")

            return 0
        except Exception as exc:  # noqa: BLE001
            print(f"runtime error: {exc}")
            return 1

    if args.command == "clearance":
        try:
            from temporalci.metrics.catenary_clearance import evaluate
            from temporalci.types import GeneratedSample

            frame_dir = Path(args.frame_dir)
            if not frame_dir.is_dir():
                print(f"error: not a directory: {frame_dir}")
                return 1

            frames = sorted(frame_dir.glob(args.pattern))
            if not frames:
                print(f"no frames matching '{args.pattern}' in {frame_dir}")
                return 1

            samples = [
                GeneratedSample(
                    test_id="clearance",
                    prompt=f.stem,
                    seed=0,
                    video_path=str(f),
                    evaluation_stream=[],
                )
                for f in frames
            ]

            print(f"Running 3-layer vision pipeline on {len(samples)} frames...")
            print(f"  Device: {args.device}")
            print(f"  Depth: {'skip' if args.skip_depth else 'enabled'}")

            result = evaluate(
                samples,
                params={
                    "device": args.device,
                    "skip_depth": str(args.skip_depth).lower(),
                    "output_dir": args.output_dir,
                },
            )

            if args.print_json:
                print(json.dumps(result, indent=2, default=str))
            else:
                print(f"\n{'=' * 60}")
                print(f"  Score: {result['score']:.4f}")
                print(f"  Samples: {result['sample_count']}")
                for dim, val in result["dims"].items():
                    print(f"  {dim}: {val:.6f}")
                alerts = result.get("alert_frames", [])
                print(f"  Alert frames: {len(alerts)}")
                for a in alerts:
                    print(
                        f"    {a['prompt']:20s}  risk={a['risk_level']}  "
                        f"score={a['risk_score']:.2f}  clearance={a['clearance_px']:.0f}px"
                    )
                print(f"{'=' * 60}")
                print(f"Multi-panel visualizations: {args.output_dir}/")

            return 0
        except Exception as exc:  # noqa: BLE001
            print(f"runtime error: {exc}")
            return 1

    if args.command == "route-map":
        try:
            from temporalci.route_map import generate_route_map

            run_dir = Path(args.run_dir)
            run_json = run_dir / "run.json"
            if not run_json.exists():
                print(f"run.json not found in {run_dir}")
                return 1
            run_data = json.loads(run_json.read_text(encoding="utf-8"))
            per_sample = run_data.get("per_sample", [])
            if not per_sample:
                print(f"no per_sample data in {run_json}")
                return 1
            output = Path(args.output)
            generate_route_map(per_sample, output, title=args.title)
            geo_count = sum(
                1 for s in per_sample if s.get("lat") is not None and s.get("lon") is not None
            )
            print(f"route map: {output} ({geo_count}/{len(per_sample)} geo-located)")
            return 0
        except Exception as exc:  # noqa: BLE001
            print(f"runtime error: {exc}")
            return 1

    if args.command == "temporal-diff":
        try:
            from temporalci.temporal_diff import temporal_diff, write_temporal_diff_report

            before_dir = Path(args.before)
            after_dir = Path(args.after)

            for label, d in [("before", before_dir), ("after", after_dir)]:
                rj = d / "run.json"
                if not rj.exists():
                    print(f"run.json not found in {label} dir: {d}")
                    return 1

            before_data = json.loads((before_dir / "run.json").read_text(encoding="utf-8"))
            after_data = json.loads((after_dir / "run.json").read_text(encoding="utf-8"))

            # Extract metric results (may be nested under metrics key)
            before_result = before_data.get("metrics", {}).get("catenary_clearance", before_data)
            after_result = after_data.get("metrics", {}).get("catenary_clearance", after_data)

            if args.print_json:
                diff = temporal_diff(before_result, after_result)
                print(json.dumps(diff, indent=2, default=str))
            else:
                output = Path(args.output)
                diff = write_temporal_diff_report(
                    output,
                    before_result,
                    after_result,
                    before_label=args.before_label,
                    after_label=args.after_label,
                )
                summary = diff["summary"]
                print(f"\n{'=' * 60}")
                print(f"  Temporal Diff: {args.before_label} vs {args.after_label}")
                print(f"  Matched frames: {summary['matched_count']}")
                print(f"  Avg risk delta: {summary.get('avg_risk_delta', 0):+.4f}")
                print(f"  Degraded: {summary.get('degraded_count', 0)}")
                print(f"  Improved: {summary.get('improved_count', 0)}")
                print(f"  Stable: {summary.get('stable_count', 0)}")
                print(f"  Hotspots: {len(diff['hotspots'])}")
                print(f"{'=' * 60}")
                print(f"Report: {output}")

            return 0
        except Exception as exc:  # noqa: BLE001
            print(f"runtime error: {exc}")
            return 1

    if args.command == "dashboard":
        try:
            from temporalci.dashboard import generate_dashboard

            run_dir = Path(args.run_dir)
            run_json = run_dir / "run.json"
            if not run_json.exists():
                print(f"run.json not found in {run_dir}")
                return 1
            run_data = json.loads(run_json.read_text(encoding="utf-8"))

            # Try to extract catenary_clearance results or use top-level
            metrics = run_data.get("metrics", {})
            results = metrics.get("catenary_clearance", run_data)

            output = Path(args.output)
            generate_dashboard(results, output, title=args.title)
            print(f"dashboard: {output}")
            return 0
        except Exception as exc:  # noqa: BLE001
            print(f"runtime error: {exc}")
            return 1

    if args.command == "km-report":
        try:
            from temporalci.fusion import (
                aggregate_by_km,
                generate_km_report,
                prioritize_maintenance,
            )

            run_dir = Path(args.run_dir)
            run_json = run_dir / "run.json"
            if not run_json.exists():
                print(f"run.json not found in {run_dir}")
                return 1
            run_data = json.loads(run_json.read_text(encoding="utf-8"))

            # Extract per_sample data (may be nested under metrics)
            metrics = run_data.get("metrics", {})
            result = metrics.get("catenary_clearance", run_data)
            per_sample = result.get("per_sample", [])
            if not per_sample:
                print(f"no per_sample data in {run_json}")
                return 1

            km_bins = aggregate_by_km(per_sample, bin_size_km=args.bin_size)
            if not km_bins:
                print("no frames with km data found")
                return 1

            if args.print_json:
                km_payload: dict[str, Any] = {"km_bins": km_bins}
                if args.budget is not None:
                    km_payload["priority"] = prioritize_maintenance(km_bins, budget_km=args.budget)
                print(json.dumps(km_payload, indent=2))
            else:
                output = Path(args.output)
                generate_km_report(km_bins, output, title=args.title)
                print(
                    f"km report: {output} ({len(km_bins)} bins, "
                    f"{sum(b['frame_count'] for b in km_bins)} frames)"
                )

                if args.budget is not None:
                    priority = prioritize_maintenance(km_bins, budget_km=args.budget)
                    print(f"\npriority maintenance ({args.budget:.1f} km budget):")
                    for seg in priority:
                        print(
                            f"  km {seg['km_start']:.1f}-{seg['km_end']:.1f}  "
                            f"risk={seg['avg_risk']:.4f}  urgency={seg['urgency']}"
                        )
                    if not priority:
                        print("  (no segments fit within budget)")

            return 0
        except Exception as exc:  # noqa: BLE001
            print(f"runtime error: {exc}")
            return 1

    if args.command == "trend":
        try:
            model_root = Path(args.model_root)
            runs = load_model_runs(model_root, last_n=args.last_n)
            if not runs:
                print(f"no runs found in {model_root}")
                return 1
            out = Path(args.output)
            write_trend_report(out, runs, title=args.title)
            print(f"trend report: {out} ({len(runs)} runs)")
            return 0
        except Exception as exc:  # noqa: BLE001
            print(f"runtime error: {exc}")
            return 1

    if args.command == "serve":
        try:
            from temporalci.server import create_app, run_server

            video = Path(args.video)
            if not video.is_file():
                print(f"video not found: {video}")
                return 1

            app = create_app(
                video_path=str(video),
                fps=args.fps,
                max_frames=args.max_frames,
                device=args.device,
                skip_depth=args.skip_depth,
                output_dir=args.output_dir,
            )
            print(f"Starting live dashboard at http://{args.host}:{args.port}")
            print(f"Video: {video}")
            print("Press Ctrl+C to stop")
            run_server(app, host=args.host, port=args.port)
            return 0
        except ImportError as exc:
            print(f"missing dependency: {exc}")
            return 1
        except Exception as exc:  # noqa: BLE001
            print(f"runtime error: {exc}")
            return 1

    if args.command != "run":
        parser.error(f"unknown command: {args.command}")

    # CI mode: auto-detect from flag or environment variables
    _ci_flag = getattr(args, "ci", False)
    _ci_mode = _ci_flag or bool(_os.environ.get("CI")) or bool(_os.environ.get("GITHUB_ACTIONS"))
    _github_actions = bool(_os.environ.get("GITHUB_ACTIONS"))

    # Build progress callback unless --no-progress or --print-json or CI mode
    no_progress = getattr(args, "no_progress", False)
    use_progress = not no_progress and not args.print_json and not _ci_mode

    def _make_progress_cb(model_label: str | None = None) -> Any:
        prefix = f"[{model_label}] " if model_label else ""

        def _cb(current: int, total: int, test_id: str, prompt: str, seed: int) -> None:
            short = (prompt[:55] + "…") if len(prompt) > 55 else prompt
            print(f'{prefix}[{current}/{total}] {test_id}  "{short}"  seed={seed}', flush=True)

        return _cb

    adapter_timeout = getattr(args, "adapter_timeout", None)
    gate_overrides = getattr(args, "gate_overrides", None) or []

    try:
        suite = load_suite(Path(args.suite))
    except SuiteValidationError as exc:
        print(f"config error: {exc}")
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"runtime error: {exc}")
        return 1

    # Apply gate overrides before running
    if gate_overrides:
        try:
            suite = _apply_gate_overrides(suite, gate_overrides)
        except ValueError as exc:
            print(f"config error: {exc}")
            return 1

    # Multi-model mode: when --model is omitted, run all models in suite
    model_names = [args.model] if args.model else [m.name for m in suite.models]

    # Apply --include-model / --exclude-model filters (only in multi-model mode)
    if not args.model:
        include_models = getattr(args, "include_models", None)
        exclude_models = getattr(args, "exclude_models", None)
        if include_models:
            model_names = [m for m in model_names if m in include_models]
        if exclude_models:
            model_names = [m for m in model_names if m not in exclude_models]
        if not model_names:
            print("config error: no models match --include-model / --exclude-model filters")
            return 1

    multi = len(model_names) > 1
    prune_keep = getattr(args, "prune_keep_last", None)
    sample_limit = getattr(args, "sample_limit", None)
    tag = getattr(args, "tag", None)
    workers = getattr(args, "workers", 1) or 1
    retry = getattr(args, "retry", 1) or 1
    fail_on_skip = getattr(args, "fail_on_skip", False)
    inter_sample_delay = getattr(args, "inter_sample_delay", 0.0) or 0.0
    fail_fast = getattr(args, "fail_fast", False)
    model_workers = getattr(args, "model_workers", 1) or 1
    notify_on = getattr(args, "notify_on", "change") or "change"
    env = getattr(args, "env", None) or None
    watch_interval = getattr(args, "watch", None)

    # ── P1: dry-run ─────────────────────────────────────────────────────────
    if getattr(args, "dry_run", False):
        from temporalci.adapters import build_adapter
        from temporalci.config import select_model

        print(
            f"dry-run: {len(model_names)} model(s)  project={suite.project}  suite={suite.suite_name}"
        )
        all_ok = True
        for mname in model_names:
            print(f"\n  ── {mname} ──")
            model = select_model(suite, mname)
            jobs_total = sum(len(t.prompts) * len(t.seeds) for t in suite.tests)
            if sample_limit is not None:
                jobs_total = min(jobs_total, sample_limit)
            try:
                build_adapter(model)
                adapter_status = "ok"
            except Exception as exc:  # noqa: BLE001
                adapter_status = f"ERROR: {exc}"
                all_ok = False
            metric_names = ", ".join(m.name for m in suite.metrics)
            print(f"    adapter:  {model.adapter}  ({adapter_status})")
            print(f"    samples:  {jobs_total}")
            print(f"    metrics:  {metric_names}")
            print(f"    gates:    {len(suite.gates)}")
        return 0 if all_ok else 1
    # ────────────────────────────────────────────────────────────────────────

    def _run_one_model(mname: str) -> tuple[str, dict[str, Any] | None, Exception | None]:
        pcb = (
            _make_progress_cb(mname if multi else None)
            if use_progress and model_workers <= 1
            else None
        )
        try:
            r = run_suite(
                suite=suite,
                model_name=mname,
                artifacts_dir=args.artifacts_dir,
                fail_on_regression=not args.ignore_regression,
                fail_on_skip=fail_on_skip,
                baseline_mode=args.baseline_mode,
                webhook_url=args.webhook_url,
                sample_limit=sample_limit,
                tag=tag,
                progress_callback=pcb,
                workers=workers,
                retry=retry,
                inter_sample_delay=inter_sample_delay,
                notify_on=notify_on,
                env=env,
                adapter_timeout=adapter_timeout,
            )
            return mname, cast("dict[str, Any]", r), None
        except SuiteValidationError as exc:
            return mname, None, exc
        except Exception as exc:  # noqa: BLE001
            return mname, None, exc

    def _print_model_result(
        mname: str,
        result: dict[str, Any] | None,
        exc: Exception | None,
    ) -> None:
        if multi:
            print(f"\n── {mname} ──")
        if exc is not None:
            if isinstance(exc, SuiteValidationError):
                print(f"config error: {exc}")
                if _github_actions:
                    print(f"::error::TemporalCI config error [{mname}]: {exc}")
            else:
                print(f"runtime error [{mname}]: {exc}")
                if _github_actions:
                    print(f"::error::TemporalCI runtime error [{mname}]: {exc}")
        elif result is not None:
            if args.print_json:
                print(json.dumps(result, indent=2))
            else:
                _print_summary(result)
            # GitHub Actions annotations for gate failures and regressions
            if _github_actions:
                for gate in result.get("gates") or []:
                    if not isinstance(gate, dict) or gate.get("passed"):
                        continue
                    print(
                        f"::error::TemporalCI gate FAILED [{mname}]: "
                        f"{gate.get('metric')} {gate.get('op')} {gate.get('value')} "
                        f"(actual={gate.get('actual')})"
                    )
                for reg in result.get("regressions") or []:
                    if not isinstance(reg, dict) or not reg.get("regressed"):
                        continue
                    print(
                        f"::warning::TemporalCI regression [{mname}]: "
                        f"{reg.get('metric')} baseline={reg.get('baseline')} "
                        f"current={reg.get('current')} delta={reg.get('delta')}"
                    )
            if prune_keep is not None:
                try:
                    from temporalci.prune import prune_model_runs

                    run_dir = Path(str(result.get("run_dir", "")))
                    prune_result = prune_model_runs(run_dir.parent, keep_last=prune_keep)
                    mb = prune_result["bytes_freed"] / 1_048_576
                    print(
                        f"prune: kept={prune_result['kept']}  deleted={prune_result['deleted']}"
                        f"  freed={mb:.2f} MB"
                    )
                except Exception as exc2:  # noqa: BLE001
                    print(f"prune warning: {exc2}")

    def _run_iteration() -> int:
        results: list[tuple[str, dict[str, Any] | None, Exception | None]] = []

        if model_workers > 1:
            import concurrent.futures as _cf

            _order = {n: i for i, n in enumerate(model_names)}
            with _cf.ThreadPoolExecutor(max_workers=model_workers) as pool:
                _raw = list(pool.map(_run_one_model, model_names))
            results = sorted(_raw, key=lambda t: _order.get(t[0], 0))
            for mname, result, exc in results:
                _print_model_result(mname, result, exc)
        else:
            for mname in model_names:
                mname_out, result, exc = _run_one_model(mname)
                results.append((mname_out, result, exc))
                _print_model_result(mname_out, result, exc)
                if exc is not None and fail_fast:
                    break
                if result is not None and fail_fast and result.get("status") != "PASS":
                    if multi:
                        print(f"  fail-fast: stopping after {mname_out} failed")
                    break

        # ── --output-json ────────────────────────────────────────────────────
        output_json_path = getattr(args, "output_json", None)
        if output_json_path:
            try:
                out_p = Path(output_json_path)
                out_p.parent.mkdir(parents=True, exist_ok=True)
                if multi:
                    data: Any = [r[1] for r in results if r[1] is not None]
                else:
                    data = results[0][1] if results and results[0][1] is not None else {}
                out_p.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
                print(f"→ json: {out_p}")
            except Exception as exc:  # noqa: BLE001
                print(f"output-json warning: {exc}")

        # Multi-model summary table
        if multi:
            print("\n── summary ──")
            n_pass = 0
            for mname, res, err in results:
                if err is not None:
                    print(f"  {mname}: ERROR  {err}")
                elif res is not None and res.get("status") == "PASS":
                    n_pass += 1
                    print(f"  {mname}: PASS  samples={res['sample_count']}")
                else:
                    s = res.get("status", "FAIL") if res else "FAIL"
                    print(f"  {mname}: {s}  gate_failed={res.get('gate_failed') if res else '?'}")
            print(f"\n{n_pass}/{len(results)} models passed.")
            all_ok = n_pass == len(results)
            if _github_actions:
                level = "notice" if all_ok else "error"
                print(f"::{level}::TemporalCI: {n_pass}/{len(results)} models passed")
            return 0 if all_ok else 2

        # Single-model exit code
        single_result = results[0][1] if results else None
        return 0 if (single_result and single_result.get("status") == "PASS") else 2

    # ── --watch: continuous mode ─────────────────────────────────────────────
    if watch_interval is None:
        return _run_iteration()

    import time as _watch_time
    import datetime as _watch_dt

    _watch_iter = 0
    while True:
        try:
            _watch_iter += 1
            _now = _watch_dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"\n{'─' * 60}")
            print(f"watch  iteration={_watch_iter}  {_now}")
            print(f"{'─' * 60}")
            _run_iteration()
            print(f"\nwatch: next run in {watch_interval:.0f}s  (Ctrl+C to stop)")
            _watch_time.sleep(watch_interval)
        except KeyboardInterrupt:
            print("\nwatch: stopped.")
            return 0
