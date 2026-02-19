"""CLI: generate a cross-run trend report for a model artifact directory.

Usage
-----
python scripts/trend_report.py \\
    --model-root artifacts/demo-video-model/regression_core/demo_mock_model \\
    --output artifacts/trend_report.html \\
    --last-n 30

Or discover model roots automatically under a project/suite:

python scripts/trend_report.py \\
    --artifacts-dir artifacts \\
    --project demo-video-model \\
    --suite regression_core \\
    --model demo_mock_model \\
    --output artifacts/trend_report.html
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate an HTML trend report from TemporalCI artifact history."
    )

    src = p.add_mutually_exclusive_group()
    src.add_argument(
        "--model-root",
        default=None,
        metavar="DIR",
        help="Direct path to the model run directory (e.g. artifacts/proj/suite/model)",
    )
    src.add_argument(
        "--artifacts-dir",
        default=None,
        metavar="DIR",
        help="Artifact root; combine with --project / --suite / --model",
    )

    p.add_argument("--project", default=None)
    p.add_argument("--suite", default=None)
    p.add_argument("--model", default=None)
    p.add_argument(
        "--last-n",
        type=int,
        default=30,
        help="Maximum number of recent runs to include (default: 30)",
    )
    p.add_argument(
        "--output",
        default="trend_report.html",
        metavar="FILE",
        help="Output HTML file path (default: trend_report.html)",
    )
    p.add_argument(
        "--title",
        default=None,
        help="Report title (auto-generated from model path if omitted)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    # Resolve model_root
    if args.model_root:
        model_root = Path(args.model_root).resolve()
    elif args.artifacts_dir:
        parts = [args.artifacts_dir]
        for segment in (args.project, args.suite, args.model):
            if segment:
                parts.append(segment)
        model_root = Path(*parts).resolve()
    else:
        print("error: provide --model-root or --artifacts-dir with --project/--suite/--model")
        return 1

    if not model_root.exists():
        print(f"error: model root not found: {model_root}")
        return 1

    runs_jsonl = model_root / "runs.jsonl"
    if not runs_jsonl.exists():
        print(f"error: no runs.jsonl found in {model_root}")
        return 1

    # Import here so errors are clear
    from temporalci.trend import load_model_runs, write_trend_report

    runs = load_model_runs(model_root, last_n=args.last_n)
    if not runs:
        print(f"no runs found in {model_root}")
        return 1

    title = args.title or f"TemporalCI Trend — {model_root.name}"
    output = Path(args.output).resolve()

    write_trend_report(output, runs, title=title)
    print(f"wrote trend report ({len(runs)} runs) → {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
