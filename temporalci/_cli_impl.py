"""CLI command handler implementations extracted from ``cli.py``.

All ``_cmd_*`` functions, ``_print_summary``, and related helpers live here
so that ``cli.py`` can stay focused on parser construction and ``main()``.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from temporalci.compare import compare_runs, format_compare_text, write_compare_report
from temporalci.trend import load_model_runs, write_trend_report


# ---------------------------------------------------------------------------
# Run summary printer (used by main() after run_suite)
# ---------------------------------------------------------------------------


def _print_summary(result: dict[str, Any]) -> None:
    print(f"status={result['status']} run_id={result['run_id']}")
    print(f"run_dir={result['run_dir']}")
    skipped = result.get("skipped_count", 0)
    skipped_str = f"  skipped={skipped}" if skipped else ""
    print(f"model={result['model_name']} samples={result['sample_count']}{skipped_str}")
    print(f"gate_failed={result['gate_failed']} regression_failed={result['regression_failed']}")

    gates = result.get("gates", [])
    if isinstance(gates, list):
        for gate in gates:
            if not isinstance(gate, dict):
                continue
            status = "PASS" if gate.get("passed") else "FAIL"
            print(
                "gate "
                f"{gate.get('metric')} {gate.get('op')} {gate.get('value')} "
                f"actual={gate.get('actual')} => {status}"
            )

    regressions = result.get("regressions", [])
    if isinstance(regressions, list) and regressions:
        for item in regressions:
            if not isinstance(item, dict):
                continue
            status = "REGRESSED" if item.get("regressed") else "OK"
            print(
                "regression "
                f"{item.get('metric')} baseline={item.get('baseline')} "
                f"current={item.get('current')} delta={item.get('delta')} => {status}"
            )

    # Artifact paths
    run_dir = Path(str(result.get("run_dir", "")))
    if run_dir.exists():
        for fname, label in [
            ("report.html", "report"),
            ("compare_report.html", "compare"),
        ]:
            p = run_dir / fname
            if p.exists():
                print(f"→ {label}: {p}")
        model_root = run_dir.parent
        trend = model_root / "trend_report.html"
        if trend.exists():
            print(f"→ trend:   {trend}")
        index = model_root.parent / "index.html"
        if index.exists():
            print(f"→ index:   {index}")


# ---------------------------------------------------------------------------
# doctor
# ---------------------------------------------------------------------------


def _cmd_doctor(_args: Any) -> int:
    """Handler for ``temporalci doctor``."""
    import subprocess
    import sys

    issues = 0

    def _ok(msg: str) -> None:
        print(f"[ok]   {msg}")

    def _warn(msg: str) -> None:
        nonlocal issues
        issues += 1
        print(f"[warn] {msg}")

    # Python version
    major, minor, micro = sys.version_info[:3]
    py_ver = f"{major}.{minor}.{micro}"
    if major >= 3 and minor >= 11:
        _ok(f"Python {py_ver}")
    else:
        _warn(f"Python {py_ver}  (3.11+ recommended)")

    # git
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
        _ok(f"git: {commit} ({branch})")
    except Exception:  # noqa: BLE001
        _warn("git: not available or not in a git repository")

    # Adapters
    from temporalci.adapters import available_adapters
    adapters = available_adapters()
    _ok(f"adapters ({len(adapters)}): {', '.join(adapters)}")

    # Metrics
    from temporalci.metrics import available_metrics
    metrics_list = available_metrics()
    _ok(f"metrics ({len(metrics_list)}): {', '.join(metrics_list)}")

    # Config file
    cfg_found = False
    for cfg_name in (".temporalci.yaml", ".temporalci.yml"):
        cfg_path = Path(cfg_name)
        if cfg_path.exists():
            try:
                import yaml as _yaml
                data = _yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
                run_cfg = data.get("run", {}) if isinstance(data, dict) else {}
                summary = "  ".join(f"{k}={v}" for k, v in list(run_cfg.items())[:5])
                _ok(f"config: {cfg_path}" + (f"  [{summary}]" if summary else ""))
            except Exception as exc:  # noqa: BLE001
                _warn(f"config: {cfg_path} found but parse error: {exc}")
            cfg_found = True
            break
    if not cfg_found:
        print("[info] config: no .temporalci.yaml in current directory")

    print()
    if issues == 0:
        print("all checks passed.")
    else:
        print(f"{issues} warning(s).")
    return 0


# ---------------------------------------------------------------------------
# tag
# ---------------------------------------------------------------------------


def _cmd_tag(args: Any) -> int:
    """Handler for ``temporalci tag``."""
    model_root = Path(args.model_root)
    if not model_root.exists():
        print(f"config error: model-root not found: {model_root}")
        return 1

    tags_file = model_root / "tags.json"

    def _load() -> dict[str, str]:
        if not tags_file.exists():
            return {}
        try:
            data = json.loads(tags_file.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (json.JSONDecodeError, OSError):
            return {}

    def _save(tags: dict[str, str]) -> None:
        tags_file.write_text(json.dumps(tags, indent=2), encoding="utf-8")

    if args.action == "list":
        tags = _load()
        if not tags:
            print("no tags set")
            return 0
        col = max(len(k) for k in tags)
        for k, v in sorted(tags.items()):
            print(f"  {k:<{col}}  →  {v}")
        return 0

    if args.action == "show":
        if not args.name:
            print("config error: --name is required for 'show'")
            return 1
        tags = _load()
        run_id = tags.get(args.name)
        if not run_id:
            print(f"tag '{args.name}' not found")
            return 1
        print(f"{args.name}  →  {run_id}")
        run_json = model_root / run_id / "run.json"
        if run_json.exists():
            try:
                payload = json.loads(run_json.read_text(encoding="utf-8"))
                status = payload.get("status", "?")
                ts = str(payload.get("timestamp_utc", "?"))[:19]
                print(f"  status={status}  timestamp={ts}")
            except Exception:  # noqa: BLE001
                pass
        return 0

    if args.action == "delete":
        if not args.name:
            print("config error: --name is required for 'delete'")
            return 1
        tags = _load()
        if args.name not in tags:
            print(f"tag '{args.name}' not found")
            return 1
        del tags[args.name]
        _save(tags)
        print(f"deleted tag '{args.name}'")
        return 0

    if args.action == "set":
        if not args.name:
            print("config error: --name is required for 'set'")
            return 1
        if not args.run_id:
            print("config error: --run-id is required for 'set'")
            return 1
        tags = _load()
        tags[args.name] = args.run_id
        _save(tags)
        print(f"set tag '{args.name}'  →  {args.run_id}")
        return 0

    return 0


# ---------------------------------------------------------------------------
# alert
# ---------------------------------------------------------------------------


def _cmd_alert(args: Any) -> int:
    """Handler for ``temporalci alert``."""

    def _read_state(model_root: Path) -> tuple[str, str, str]:
        """Return (state, last_run_id, last_change_run_id)."""
        state_file = model_root / "alert_state.json"
        if not state_file.exists():
            return "passing", "", ""
        try:
            data = json.loads(state_file.read_text(encoding="utf-8"))
            return (
                str(data.get("state", "passing")),
                str(data.get("last_run_id", "")),
                str(data.get("last_change_run_id", "")),
            )
        except Exception:  # noqa: BLE001
            return "unknown", "", ""

    if getattr(args, "suite_root", None):
        from temporalci.index import discover_models

        suite_root = Path(args.suite_root)
        if not suite_root.exists():
            print(f"config error: suite-root not found: {suite_root}")
            return 1
        models = discover_models(suite_root)
        if not models:
            print(f"no models found in {suite_root}")
            return 0
        any_failing = False
        for name, model_root in models:
            state, last_run, _ = _read_state(model_root)
            tag = "FAILING" if state == "failing" else "ok"
            suffix = f"  last_run={last_run}" if last_run else ""
            print(f"  {name}: {tag}{suffix}")
            if state == "failing":
                any_failing = True
        return 1 if any_failing else 0

    model_root = Path(args.model_root)
    if not model_root.exists():
        print(f"config error: model-root not found: {model_root}")
        return 1
    state, last_run, last_change = _read_state(model_root)
    suffix = ""
    if last_run:
        suffix = f"  last_run={last_run}"
    if last_change:
        suffix += f"  last_change={last_change}"
    if state == "failing":
        print(f"FAILING{suffix}")
        return 1
    print(f"ok: state={state}{suffix}")
    return 0


# ---------------------------------------------------------------------------
# repair-index
# ---------------------------------------------------------------------------


def _cmd_repair_index(args: Any) -> int:
    """Handler for ``temporalci repair-index``."""
    model_root = Path(args.model_root)
    if not model_root.exists():
        print(f"config error: model-root not found: {model_root}")
        return 1

    run_dirs = sorted(
        d for d in model_root.iterdir()
        if d.is_dir() and (d / "run.json").exists()
    )
    if not run_dirs:
        print(f"no run.json files found in {model_root}")
        return 1

    entries: list[dict[str, Any]] = []
    skipped = 0
    for run_dir in run_dirs:
        try:
            payload = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
            entries.append({
                "run_id": payload.get("run_id", run_dir.name),
                "timestamp_utc": payload.get("timestamp_utc", ""),
                "status": payload.get("status", "UNKNOWN"),
                "sample_count": payload.get("sample_count", 0),
            })
        except Exception as exc:  # noqa: BLE001
            print(f"  warning: skipped {run_dir.name}: {exc}")
            skipped += 1

    index_path = model_root / "runs.jsonl"
    if args.dry_run:
        print(
            f"dry-run: would write {len(entries)} entries to {index_path}"
            + (f"  ({skipped} skipped)" if skipped else "")
        )
        return 0

    with index_path.open("w", encoding="utf-8") as fh:
        for entry in entries:
            fh.write(json.dumps(entry) + "\n")

    print(
        f"repaired: {len(entries)} entries written → {index_path}"
        + (f"  ({skipped} skipped)" if skipped else "")
    )
    return 0


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------


def _cmd_suite_status(args: Any) -> int:
    """Handler for ``temporalci status --suite-root``."""
    from temporalci.index import discover_models

    suite_root = Path(args.suite_root)
    if not suite_root.exists():
        print(f"config error: suite-root not found: {suite_root}")
        return 1

    models = discover_models(suite_root)
    if not models:
        print(f"no models found in {suite_root}")
        return 1

    output_format = getattr(args, "output_format", "text")

    # JSON output: {model_name: [run, ...], ...}
    if output_format == "json":
        payload: dict[str, Any] = {}
        for model_name, model_root in models:
            runs = load_model_runs(model_root, last_n=args.last_n)
            payload[model_name] = runs
        print(json.dumps(payload, indent=2, default=str))
        return 0

    print(f"Suite: {suite_root}")
    print()
    name_w = max(len(name) for name, _ in models)
    name_w = max(name_w, 12)
    header = f"  {'Model':<{name_w}}  {'Runs':>5}  {'Pass':>4}  {'Fail':>4}  {'Status':<6}  Recent"
    print(header)
    print("  " + "-" * (len(header) - 2))

    any_fail = False
    for model_name, model_root in models:
        runs = load_model_runs(model_root, last_n=args.last_n)
        if not runs:
            print(f"  {model_name:<{name_w}}  {'—':>5}  {'—':>4}  {'—':>4}  {'N/A':<6}")
            continue
        statuses = [str(r.get("status", "UNKNOWN")) for r in runs]
        n_pass = statuses.count("PASS")
        n_fail = len(statuses) - n_pass
        latest_status = statuses[-1]
        if latest_status != "PASS":
            any_fail = True
        recent = " ".join("P" if s == "PASS" else "F" for s in statuses[-10:])
        print(f"  {model_name:<{name_w}}  {len(runs):>5}  {n_pass:>4}  {n_fail:>4}  {latest_status:<6}  [{recent}]")

    return 1 if any_fail else 0


def _cmd_status(args: Any) -> int:
    """Handler for ``temporalci status``."""
    if getattr(args, "suite_root", None):
        return _cmd_suite_status(args)

    model_root = Path(args.model_root)
    runs = load_model_runs(model_root, last_n=args.last_n)
    if not runs:
        print(f"no runs found in {model_root}")
        return 1

    output_format = getattr(args, "output_format", "text")

    # JSON output: list of full run dicts
    if output_format == "json":
        print(json.dumps(runs, indent=2, default=str))
        return 0

    # CSV output: delegate to export_runs
    if output_format == "csv":
        from temporalci.export import export_runs
        out_path = Path(args.output) if getattr(args, "output", None) else None
        if out_path is None:
            print("config error: --output PATH is required for --output-format csv")
            return 1
        n = export_runs(model_root, out_path, last_n=args.last_n)
        if n == 0:
            print(f"no runs found in {model_root}")
            return 1
        print(f"csv: {out_path}  ({n} runs)")
        return 0

    # Header
    latest = runs[-1]
    project = str(latest.get("project", ""))
    suite_name = str(latest.get("suite_name", ""))
    model_name = str(latest.get("model_name", ""))
    print(f"Model: {project} / {suite_name} / {model_name}")
    print(f"Path:  {model_root}")

    statuses = [str(r.get("status", "UNKNOWN")) for r in runs]
    n_pass = statuses.count("PASS")
    n_fail = len(statuses) - n_pass
    sparkline = " ".join("P" if s == "PASS" else "F" for s in statuses)
    print(f"Runs:  {len(runs)} total · {n_pass} pass · {n_fail} fail")
    print(f"       [{sparkline}]")
    print()

    # Run table
    col_w = max(len(str(r.get("run_id", ""))) for r in runs)
    col_w = max(col_w, 20)
    has_notes = any(str(r.get("note", "")).strip() for r in runs)
    header = f"  {'#':>4}  {'Run ID':<{col_w}}  {'Timestamp':<19}  {'Status':<6}  Samples"
    if has_notes:
        header += "  Note"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for i, run in enumerate(runs, start=1):
        run_id = str(run.get("run_id", ""))
        ts = str(run.get("timestamp_utc", ""))[:19]
        s = str(run.get("status", "UNKNOWN"))
        samples = str(run.get("sample_count", ""))
        line = f"  {i:>4}  {run_id:<{col_w}}  {ts:<19}  {s:<6}  {samples}"
        if has_notes:
            note = str(run.get("note", "")).strip()[:40]
            line += f"  {note}"
        print(line)

    # Latest metrics — collect all scalar paths and compare to previous run
    def _collect_flat_metrics(obj: Any) -> dict[str, float]:
        result: dict[str, float] = {}

        def _walk(o: Any, prefix: str) -> None:
            if not isinstance(o, dict):
                return
            for k, v in o.items():
                path = f"{prefix}.{k}" if prefix else k
                if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v)):
                    result[path] = float(v)
                elif isinstance(v, dict):
                    _walk(v, path)

        _walk(obj, "")
        return result

    all_metrics = _collect_flat_metrics(latest.get("metrics") or {})
    prev_metrics = _collect_flat_metrics(runs[-2].get("metrics") or {}) if len(runs) >= 2 else {}

    if all_metrics:
        print()
        verbose = getattr(args, "verbose", False)
        # Default: only paths with ≤1 dot (e.g. "vbench_temporal.score" not "...dims.motion_smoothness")
        display = all_metrics if verbose else {k: v for k, v in all_metrics.items() if k.count(".") <= 1}
        if display:
            label = "Latest metrics (all paths):" if verbose else "Latest metrics:"
            print(label)
            col_m = max(len(k) for k in display)
            for k, v in sorted(display.items(), key=lambda kv: (kv[0].count("."), kv[0])):
                if k in prev_metrics:
                    delta = v - prev_metrics[k]
                    if abs(delta) < 1e-9:
                        arrow = "  →"
                    elif delta > 0:
                        arrow = f"  ↑ +{delta:.6f}"
                    else:
                        arrow = f"  ↓ {delta:.6f}"
                else:
                    arrow = ""
                print(f"  {k:<{col_m}}  {v:.6f}{arrow}")
            hidden = len(all_metrics) - len(display)
            if not verbose and hidden > 0:
                print(f"  … {hidden} more paths hidden (use --verbose to show all)")

    return 0


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------


def _cmd_compare(args: Any) -> int:
    """Handler for ``temporalci compare``.

    Run-ID mode (``RUN_A RUN_B --model-root``): loads run.json from two run
    directories inside ``model_root`` by their run IDs.

    Auto mode (``--model-root`` only): selects the latest run as candidate and
    the most recent PASS run (other than the candidate) as the baseline.

    Explicit mode (``--baseline`` + ``--candidate``): loads the two specified
    run.json files directly.

    Returns 1 if any metric regressions are detected; 0 otherwise.
    """
    try:
        out = Path(args.output)
        run_a = getattr(args, "run_a", None)
        run_b = getattr(args, "run_b", None)

        if run_a and run_b:
            # Run-ID mode: resolve paths relative to --model-root
            if not getattr(args, "model_root", None):
                print("compare: --model-root is required when using run-id mode")
                return 1
            model_root = Path(args.model_root)
            b_path = model_root / run_a / "run.json"
            c_path = model_root / run_b / "run.json"
            if not b_path.exists():
                print(f"config error: run.json not found for run '{run_a}': {b_path}")
                return 1
            if not c_path.exists():
                print(f"config error: run.json not found for run '{run_b}': {c_path}")
                return 1
            baseline_run = json.loads(b_path.read_text(encoding="utf-8"))
            candidate_run = json.loads(c_path.read_text(encoding="utf-8"))
            print(f"comparing: {run_a}  →  {run_b}")
            cmp = write_compare_report(out, baseline_run, candidate_run)
        elif getattr(args, "model_root", None) and not run_a:
            # Auto mode: derive runs from model_root
            model_root = Path(args.model_root)
            runs = load_model_runs(model_root, last_n=200)
            if not runs:
                print(f"no runs found in {model_root}")
                return 1
            candidate_run = runs[-1]
            candidate_id = candidate_run.get("run_id")
            baseline_run: dict[str, Any] | None = None
            for run in reversed(runs[:-1]):
                if run.get("status") == "PASS":
                    baseline_run = run
                    break
            if baseline_run is None:
                # Fallback: use second-to-last if only one run or no PASS found
                if len(runs) >= 2:
                    baseline_run = runs[-2]
                else:
                    print(f"need at least 2 runs in {model_root} for auto compare")
                    return 1
            b_id = str(baseline_run.get("run_id", ""))
            c_id = str(candidate_id or "")
            print(f"auto-selected: baseline={b_id}  candidate={c_id}")
            cmp = write_compare_report(out, baseline_run, candidate_run)
        elif getattr(args, "baseline", None) and getattr(args, "candidate", None):
            # Explicit mode
            b_path = Path(args.baseline)
            c_path = Path(args.candidate)
            if not b_path.exists():
                print(f"config error: baseline not found: {b_path}")
                return 1
            if not c_path.exists():
                print(f"config error: candidate not found: {c_path}")
                return 1
            baseline_run = json.loads(b_path.read_text(encoding="utf-8"))
            candidate_run = json.loads(c_path.read_text(encoding="utf-8"))
            cmp = write_compare_report(out, baseline_run, candidate_run)
        else:
            print("compare: use RUN_A RUN_B --model-root, --model-root (auto), or --baseline + --candidate")
            return 1

        print(format_compare_text(cmp))
        print(f"compare report: {out}")

        # Return 1 if any metric regressions detected
        has_regression = bool(cmp.get("gate_regressions")) or any(
            float(d.get("delta", 0)) < 0
            for d in (cmp.get("metric_deltas") or [])
            if isinstance(d, dict)
        )
        return 1 if has_regression else 0
    except Exception as exc:  # noqa: BLE001
        print(f"runtime error: {exc}")
        return 1


# ---------------------------------------------------------------------------
# prune
# ---------------------------------------------------------------------------


def _prune_one(model_root: Path, keep_last: int, dry_run: bool, label: str = "") -> int:
    """Prune a single model_root. Returns bytes_freed (or 0 on error)."""
    from temporalci.prune import prune_model_runs

    result = prune_model_runs(model_root, keep_last=keep_last, dry_run=dry_run)
    prefix = "[dry-run] " if dry_run else ""
    mb = result["bytes_freed"] / 1_048_576
    tag = f"{label}: " if label else ""
    print(
        f"{prefix}{tag}kept={result['kept']}  deleted={result['deleted']}"
        f"  skipped={result['skipped']}  freed={mb:.2f} MB"
    )
    return result["bytes_freed"]


def _cmd_prune(args: Any) -> int:
    """Handler for ``temporalci prune``."""
    try:
        if getattr(args, "suite_root", None):
            from temporalci.index import discover_models

            suite_root = Path(args.suite_root)
            if not suite_root.exists():
                print(f"config error: suite-root not found: {suite_root}")
                return 1
            models = discover_models(suite_root)
            if not models:
                print(f"no models found in {suite_root}")
                return 1
            total_freed = 0
            for model_name, model_root in models:
                total_freed += _prune_one(
                    model_root, args.keep_last, args.dry_run, label=model_name
                )
            prefix = "[dry-run] " if args.dry_run else ""
            print(f"{prefix}total freed: {total_freed / 1_048_576:.2f} MB")
            return 0
        else:
            model_root = Path(args.model_root)
            if not model_root.exists():
                print(f"config error: model-root not found: {model_root}")
                return 1
            _prune_one(model_root, args.keep_last, args.dry_run)
            return 0
    except ValueError as exc:
        print(f"config error: {exc}")
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"runtime error: {exc}")
        return 1


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------


def _cmd_export(args: Any) -> int:
    """Handler for ``temporalci export``."""
    from temporalci.export import export_runs, export_suite_runs

    try:
        output = Path(args.output)

        if getattr(args, "suite_root", None):
            suite_root = Path(args.suite_root)
            if not suite_root.exists():
                print(f"config error: suite-root not found: {suite_root}")
                return 1
            n = export_suite_runs(suite_root, output, last_n=args.last_n, fmt=args.fmt)
            if n == 0:
                print(f"no runs found in {suite_root}")
                return 1
            print(f"exported {n} runs → {output}  (format: {args.fmt}, all models)")
            return 0

        model_root = Path(args.model_root)
        if not model_root.exists():
            print(f"config error: model-root not found: {model_root}")
            return 1
        n = export_runs(model_root, output, last_n=args.last_n, fmt=args.fmt)
        if n == 0:
            print(f"no runs found in {model_root}")
            return 1
        print(f"exported {n} runs → {output}  (format: {args.fmt})")
        return 0
    except ValueError as exc:
        print(f"config error: {exc}")
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"runtime error: {exc}")
        return 1


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def _cmd_report(args: Any) -> int:
    """Handler for ``temporalci report``."""
    from temporalci.report import write_html_report

    try:
        if getattr(args, "model_root", None):
            model_root = Path(args.model_root)
            if not model_root.exists():
                print(f"config error: model-root not found: {model_root}")
                return 1
            run_dirs = sorted(
                d for d in model_root.iterdir()
                if d.is_dir() and (d / "run.json").exists()
            )
            if not run_dirs:
                print(f"no runs found in {model_root}")
                return 1
            count = 0
            for run_dir in run_dirs:
                try:
                    payload = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
                    out = run_dir / (args.output or "report.html")
                    write_html_report(out, payload)
                    count += 1
                except Exception as exc:  # noqa: BLE001
                    print(f"warning: skipped {run_dir.name}: {exc}")
            print(f"regenerated {count} report(s) in {model_root}")
            return 0

        run_dir = Path(args.run_dir)
        if not run_dir.exists():
            print(f"config error: run-dir not found: {run_dir}")
            return 1
        run_json_path = run_dir / "run.json"
        if not run_json_path.exists():
            print(f"config error: run.json not found in {run_dir}")
            return 1
        payload = json.loads(run_json_path.read_text(encoding="utf-8"))
        out = Path(args.output) if args.output else run_dir / "report.html"
        write_html_report(out, payload)
        print(f"report: {out}")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"runtime error: {exc}")
        return 1


# ---------------------------------------------------------------------------
# init
# ---------------------------------------------------------------------------


_INIT_TEMPLATE = """\
version: 1
project: {project}
suite_name: main-suite

models:
  - name: my-model
    adapter: {adapter}
    params: {{}}

tests:
  - id: quality-check
    prompts:
      - "a person walking in the park"
      - "a car driving on a highway"
    seeds: [0, 1, 2]
    video:
      num_frames: 25

metrics:
  - name: {metric}

gates:
  - metric: {metric}.score
    op: ">="
    value: 0.7
"""


def _cmd_init(args: Any) -> int:
    """Handler for ``temporalci init``."""
    out = Path(args.output)
    if out.exists() and not args.force:
        print(f"config error: '{out}' already exists. Use --force to overwrite.")
        return 1
    try:
        text = _INIT_TEMPLATE.format(
            project=args.project,
            adapter=args.adapter,
            metric=args.metric,
        )
        out.write_text(text, encoding="utf-8")
        print(f"created {out}")
        print(f"  project={args.project}  adapter={args.adapter}  metric={args.metric}")
        print("Edit the file, then run: temporalci validate suite.yaml")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"runtime error: {exc}")
        return 1


# ---------------------------------------------------------------------------
# annotate
# ---------------------------------------------------------------------------


def _cmd_annotate(args: Any) -> int:
    """Handler for ``temporalci annotate``."""
    model_root = Path(args.model_root)
    if not model_root.exists():
        print(f"config error: model-root not found: {model_root}")
        return 1
    run_json_path = model_root / args.run_id / "run.json"
    if not run_json_path.exists():
        print(f"config error: run.json not found: {run_json_path}")
        return 1
    try:
        payload = json.loads(run_json_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            print("config error: run.json is not a JSON object")
            return 1
        payload["note"] = args.note
        run_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"annotated {args.run_id}: {args.note!r}")
        return 0
    except (json.JSONDecodeError, OSError) as exc:
        print(f"runtime error: {exc}")
        return 1


# ---------------------------------------------------------------------------
# metrics-show
# ---------------------------------------------------------------------------


def _cmd_metrics_show(args: Any) -> int:
    """Handler for ``temporalci metrics-show``."""
    import math as _math

    model_root = Path(args.model_root)
    if not model_root.exists():
        print(f"config error: model-root not found: {model_root}")
        return 1

    run_id = getattr(args, "run_id", None)
    if run_id:
        run_json_path = model_root / run_id / "run.json"
        if not run_json_path.exists():
            print(f"config error: run.json not found for run '{run_id}': {run_json_path}")
            return 1
        payload = json.loads(run_json_path.read_text(encoding="utf-8"))
    else:
        # Load latest run
        runs = load_model_runs(model_root, last_n=1)
        if not runs:
            print(f"no runs found in {model_root}")
            return 1
        latest_meta = runs[-1]
        run_id = str(latest_meta.get("run_id", ""))
        run_json_path = model_root / run_id / "run.json"
        if not run_json_path.exists():
            print(f"config error: run.json not found for latest run '{run_id}'")
            return 1
        payload = json.loads(run_json_path.read_text(encoding="utf-8"))

    verbose = getattr(args, "verbose", False)

    # Header
    print(f"Run:       {payload.get('run_id', '')}")
    print(f"Timestamp: {str(payload.get('timestamp_utc', ''))[:19]}")
    status = str(payload.get("status", "?"))
    print(f"Status:    {status}")
    sc = payload.get("sample_count", 0)
    sk = payload.get("skipped_count", 0)
    skip_str = f"  ({sk} skipped)" if sk else ""
    print(f"Samples:   {sc}{skip_str}")
    if payload.get("baseline_run_id"):
        print(f"Baseline:  {payload['baseline_run_id']}  ({payload.get('baseline_mode', '')})")
    if payload.get("env"):
        print(f"Env:       {payload['env']}")
    if isinstance(payload.get("git"), dict):
        g = payload["git"]
        dirty = " (dirty)" if g.get("dirty") else ""
        print(f"Git:       {g.get('commit', '')[:8]} ({g.get('branch', '')}){dirty}")

    # Gates
    gates = payload.get("gates") or []
    if gates:
        print()
        print("Gates:")
        col = max(len(str(g.get("metric", ""))) for g in gates)
        col = max(col, 10)
        for gate in gates:
            passed = gate.get("passed", False)
            mark = "PASS" if passed else "FAIL"
            windowed = "  [windowed]" if gate.get("windowed_pass") else ""
            actual = gate.get("actual", "?")
            print(
                f"  {mark}  {str(gate.get('metric','')):<{col}}  "
                f"{gate.get('op','')} {gate.get('value','')}  "
                f"actual={actual}{windowed}"
            )
            if verbose and "sprt" in gate:
                sprt = gate["sprt"]
                print(f"        SPRT decision={sprt.get('decision')}  "
                      f"llr={sprt.get('llr')}  "
                      f"paired={sprt.get('paired_count')}")

    # Regressions
    regressions = payload.get("regressions") or []
    if regressions:
        print()
        print("Regressions:")
        for reg in regressions:
            mark = "REGRESSED" if reg.get("regressed") else "ok"
            print(
                f"  {mark}  {reg.get('metric','')}  "
                f"baseline={reg.get('baseline')}  "
                f"current={reg.get('current')}  "
                f"delta={reg.get('delta')}"
            )

    # Metrics — flat table
    def _flat(obj: Any, prefix: str = "") -> dict[str, float]:
        result: dict[str, float] = {}
        if not isinstance(obj, dict):
            return result
        for k, v in obj.items():
            path = f"{prefix}.{k}" if prefix else k
            if isinstance(v, (int, float)) and not isinstance(v, bool) and _math.isfinite(float(v)):
                result[path] = float(v)
            elif isinstance(v, dict):
                result.update(_flat(v, path))
        return result

    metrics_raw = payload.get("metrics") or {}
    flat = _flat(metrics_raw)
    if flat:
        print()
        label = "Metrics (all paths):" if verbose else "Metrics:"
        if not verbose:
            flat = {k: v for k, v in flat.items() if k.count(".") <= 1}
        if flat:
            print(label)
            col_m = max(len(k) for k in flat)
            for k, v in sorted(flat.items(), key=lambda kv: (kv[0].count("."), kv[0])):
                print(f"  {k:<{col_m}}  {v:.6f}")

    # Per-sample distribution (verbose only)
    if verbose:
        for metric_name, metric_payload in metrics_raw.items():
            if not isinstance(metric_payload, dict):
                continue
            per_sample = metric_payload.get("per_sample")
            if not isinstance(per_sample, list) or not per_sample:
                continue
            print()
            print(f"Per-sample [{metric_name}]:")
            for i, row in enumerate(per_sample[:20]):
                if not isinstance(row, dict):
                    continue
                sid = str(row.get("sample_id", f"idx:{i}"))[:16]
                score = row.get("score", "?")
                print(f"  {sid}  score={score}")
            if len(per_sample) > 20:
                print(f"  … {len(per_sample) - 20} more (use --verbose)")

    return 0 if status == "PASS" else 1


# ---------------------------------------------------------------------------
# tune-gates
# ---------------------------------------------------------------------------


def _cmd_tune_gates(args: Any) -> int:
    """Handler for ``temporalci tune-gates``."""
    import math as _math

    model_root = Path(args.model_root)
    if not model_root.exists():
        print(f"config error: model-root not found: {model_root}")
        return 1

    runs = load_model_runs(model_root, last_n=args.last_n)
    pass_runs = [r for r in runs if r.get("status") == "PASS"]
    if not pass_runs:
        print(f"no PASS runs found in last {args.last_n} runs of {model_root}")
        return 1

    percentile = float(getattr(args, "percentile", 5.0))
    target_metric = getattr(args, "metric", None)

    # Collect metric scores: {dotted_path: [values]}
    scores: dict[str, list[float]] = {}
    for run in pass_runs:
        metrics_raw = run.get("metrics") or {}

        def _collect(obj: Any, prefix: str) -> None:
            if not isinstance(obj, dict):
                return
            for k, v in obj.items():
                path = f"{prefix}.{k}" if prefix else k
                if isinstance(v, (int, float)) and not isinstance(v, bool) and _math.isfinite(float(v)):
                    if target_metric is None or path.startswith(target_metric):
                        scores.setdefault(path, []).append(float(v))
                elif isinstance(v, dict):
                    _collect(v, path)

        _collect(metrics_raw, "")

    if not scores:
        print("no numeric metric values found in PASS runs")
        return 1

    def _percentile(values: list[float], pct: float) -> float:
        sorted_v = sorted(values)
        idx = (pct / 100.0) * (len(sorted_v) - 1)
        lo = int(idx)
        hi = min(lo + 1, len(sorted_v) - 1)
        frac = idx - lo
        return sorted_v[lo] * (1 - frac) + sorted_v[hi] * frac

    print(f"tune-gates: {model_root}")
    print(f"  PASS runs analyzed: {len(pass_runs)}  (percentile: {percentile:.0f}th)")
    print()
    print("Suggested gates (paste into your suite YAML):")
    print()
    print("gates:")

    # Determine operator: for metrics ending in "violations", "errors", "failures" → lower is better
    _lower_is_better_suffixes = ("violations", "errors", "failures", "skipped")

    for path in sorted(scores):
        vals = scores[path]
        p_val = _percentile(vals, percentile)
        avg_val = sum(vals) / len(vals)
        last_name = path.split(".")[-1]
        is_lower = any(last_name.endswith(s) for s in _lower_is_better_suffixes)
        if is_lower:
            op = "<="
            suggested = _percentile(vals, 100 - percentile)  # upper bound
        else:
            op = ">="
            suggested = p_val
        print(f"  - metric: {path}")
        print(f"    op: \"{op}\"")
        print(f"    value: {suggested:.6f}  # avg={avg_val:.4f}, n={len(vals)}")
    return 0


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------


def _cmd_summary(args: Any) -> int:
    """Handler for ``temporalci summary``."""
    artifacts_dir = Path(args.artifacts_dir)
    if not artifacts_dir.exists():
        print(f"config error: artifacts-dir not found: {artifacts_dir}")
        return 1

    output_format = getattr(args, "output_format", "text")

    # Walk: artifacts_dir / project / suite / model
    tree: dict[str, Any] = {}  # project → suite → model → latest_run_info
    any_fail = False

    for proj_dir in sorted(artifacts_dir.iterdir()):
        if not proj_dir.is_dir():
            continue
        for suite_dir in sorted(proj_dir.iterdir()):
            if not suite_dir.is_dir():
                continue
            for model_dir in sorted(suite_dir.iterdir()):
                if not model_dir.is_dir():
                    continue
                runs = load_model_runs(model_dir, last_n=1)
                if not runs:
                    continue
                latest = runs[-1]
                status = str(latest.get("status", "?"))
                if status != "PASS":
                    any_fail = True
                info: dict[str, Any] = {
                    "project": proj_dir.name,
                    "suite": suite_dir.name,
                    "model": model_dir.name,
                    "status": status,
                    "run_id": str(latest.get("run_id", "")),
                    "timestamp_utc": str(latest.get("timestamp_utc", ""))[:19],
                    "sample_count": latest.get("sample_count", 0),
                    "gate_failed": bool(latest.get("gate_failed")),
                }
                tree.setdefault(proj_dir.name, {}).setdefault(suite_dir.name, {})[model_dir.name] = info

    if not tree:
        print(f"no runs found under {artifacts_dir}")
        return 1

    if output_format == "json":
        print(json.dumps(tree, indent=2))
        return 1 if any_fail else 0

    print(f"Summary: {artifacts_dir}")
    print()
    for proj, suites in tree.items():
        print(f"  [{proj}]")
        for suite, models in suites.items():
            print(f"    {suite}/")
            for model, info in models.items():
                s = info["status"]
                ts = info["timestamp_utc"]
                sc = info.get("sample_count", "")
                marker = "✗" if s != "PASS" else "✓"
                print(f"      {marker} {model:<30} {s:<6}  {ts}  samples={sc}")
        print()

    total = sum(len(m) for s in tree.values() for m in s.values())
    n_pass = sum(
        1 for s in tree.values() for m in s.values()
        for info in m.values() if info["status"] == "PASS"
    )
    print(f"{n_pass}/{total} models passing.")
    return 1 if any_fail else 0


# ---------------------------------------------------------------------------
# history
# ---------------------------------------------------------------------------


def _cmd_history(args: Any) -> int:
    """Handler for ``temporalci history``."""
    import datetime

    model_root = Path(args.model_root)
    runs = load_model_runs(model_root, last_n=args.last_n)
    if not runs:
        print(f"no runs found in {model_root}")
        return 1

    # Filter by status
    status_filter = getattr(args, "status", "all")
    if status_filter != "all":
        runs = [r for r in runs if r.get("status") == status_filter]

    # Filter by --since DATE
    since_str = getattr(args, "since", None)
    if since_str:
        try:
            since_dt = datetime.date.fromisoformat(since_str)
        except ValueError:
            print(f"config error: --since must be YYYY-MM-DD, got: {since_str!r}")
            return 1
        filtered: list[dict[str, Any]] = []
        for r in runs:
            ts = str(r.get("timestamp_utc", ""))[:10]
            try:
                run_date = datetime.date.fromisoformat(ts)
                if run_date >= since_dt:
                    filtered.append(r)
            except ValueError:
                filtered.append(r)  # keep runs with unparseable timestamps
        runs = filtered

    if not runs:
        print("no matching runs found")
        return 1

    output_format = getattr(args, "output_format", "text")
    if output_format == "json":
        print(json.dumps(runs, indent=2, default=str))
        return 0

    # Text table
    latest = runs[-1]
    project = str(latest.get("project", ""))
    suite_name = str(latest.get("suite_name", ""))
    model_name = str(latest.get("model_name", ""))
    print(f"Model:  {project} / {suite_name} / {model_name}")
    print(f"Path:   {model_root}")
    filters = []
    if status_filter != "all":
        filters.append(f"status={status_filter}")
    if since_str:
        filters.append(f"since={since_str}")
    filter_str = f"  [{', '.join(filters)}]" if filters else ""
    print(f"Showing {len(runs)} run(s){filter_str}")
    print()

    col_w = max(len(str(r.get("run_id", ""))) for r in runs)
    col_w = max(col_w, 20)
    header = f"  {'#':>4}  {'Run ID':<{col_w}}  {'Timestamp':<19}  {'Status':<6}  {'Samples':>7}  Metrics"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for i, run in enumerate(runs, start=1):
        run_id = str(run.get("run_id", ""))
        ts = str(run.get("timestamp_utc", ""))[:19]
        s = str(run.get("status", "UNKNOWN"))
        samples = str(run.get("sample_count", ""))
        # Collect top-level metric scores for inline summary
        metrics_raw = run.get("metrics") or {}
        metric_parts: list[str] = []
        for mname, mpayload in list(metrics_raw.items())[:3]:
            if isinstance(mpayload, dict):
                score = mpayload.get("score")
                if isinstance(score, (int, float)) and not isinstance(score, bool):
                    metric_parts.append(f"{mname}={score:.4f}")
        metrics_str = "  ".join(metric_parts)
        print(f"  {i:>4}  {run_id:<{col_w}}  {ts:<19}  {s:<6}  {samples:>7}  {metrics_str}")

    return 0
