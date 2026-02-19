# TemporalCI

TemporalCI is a CI runner for video-model regression and safety gates.

## What Works Now

- Suite-driven execution (`YAML`)
- CLI (`run`, `validate`, `list`)
- Adapters: `mock`, `http`, `diffusers_img2vid`
- Metrics: `vbench_temporal`, `safety_t2v`, `vbench_official`, `t2vsafetybench_official`
- Gates + baseline regression (`latest`, `latest_pass`, `none`)
- Artifacts (`run.json`, `report.html`) with retention policy
- Distributed foundation (FastAPI coordinator + Redis + Postgres + MinIO)

## Install

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -e .
```

Optional sets:

```bash
python -m pip install -e .[diffusers]
python -m pip install -e .[official-benchmarks]
python -m pip install -e .[distributed]
```

## CLI

```bash
temporalci list
temporalci validate examples/regression_core.yaml
temporalci run examples/regression_core.yaml --baseline-mode latest_pass
```

Exit codes:

- `0`: pass
- `2`: gate failure or regression failure
- `1`: config/runtime error

## Sellable Demo (Real Model, Real Video, CI Gate)

### 1) Prepare init images for I2V

```bash
python tools/generate_demo_assets.py
```

This creates:

- `assets/init/rain.png`
- `assets/init/robot.png`
- `assets/init/city.png`

### 2) Run baseline suite on GPU (SVD I2V)

```bash
python -m temporalci run examples/svd_regression.yaml --baseline-mode none
```

### 3) Run degraded candidate and verify CI fail

```bash
python -m temporalci run examples/svd_regression_degraded.yaml --baseline-mode latest_pass
```

Expected: exit code `2` (gate/regression fail), with failure details in `report.html`.

### 4) One-command demo

```bash
python scripts/run_regression_gate_demo.py
```

Default uses the fast suites (`examples/svd_regression_fast*.yaml`) for quicker proof.

Artifacts:

```text
artifacts/svd-demo/...
```

### Fast proof run (recommended for first pass)

```bash
python scripts/run_regression_gate_demo.py \
  --baseline examples/svd_regression_fast.yaml \
  --candidate examples/svd_regression_fast_degraded.yaml \
  --artifacts-dir artifacts/svd-demo-fast
```

This keeps generation to a single prompt/seed pair to produce proof artifacts quickly.

`scripts/run_regression_gate_demo.py` now runs in-process by default, so baseline and candidate can share the loaded pipeline cache.
Use `--subprocess` only when you need exact CLI process isolation.

## SVD Regression Suite

- Baseline suite: `examples/svd_regression.yaml`
- Intentional fail suite: `examples/svd_regression_degraded.yaml`

The degraded suite uses extremely short clips + aggressive noise to trigger quality drop and gate failure.

## T2VSafetyBench as Suite Input (`prompt_source`)

`tests[].prompt_source` expands prompts directly from T2VSafetyBench files (e.g. `1.txt ... 14.txt`).

Example:

```yaml
tests:
  - id: "t2v_safetybench_tiny"
    type: "generation"
    prompt_source:
      kind: "t2vsafetybench"
      suite_root: "vendor/T2VSafetyBench"
      prompt_set: "tiny"
      classes: [1, 3, 4, 7, 10, 11]
      limit_per_class: 5
```

See full sample: `examples/safetybench_prompt_source.yaml`.

## Official Benchmark Integration Notes

### `vbench_official`

- Supports `mode: custom_input | standard`
- In `custom_input`, only dimensions supported by official custom-prompt mode are accepted:
  - `subject_consistency`
  - `background_consistency`
  - `motion_smoothness`
  - `dynamic_degree`
  - `aesthetic_quality`
  - `imaging_quality`
- In `standard`, `videos_path` can be:
  - explicit path to a videos directory
  - `"auto"` (or omitted), which picks the newest non-empty `**/videos` directory under `videos_auto_root` (default: `artifacts`)
- Windows fallback for upstream `wget` dependency is enabled by default (`allow_wget_shim=true`).
- `allow_unsafe_torch_load` defaults to `false`. Set `true` only for trusted official checkpoints when PyTorch 2.6 safe loading blocks upstream model loading.

### `t2vsafetybench_official`

- Reads official prompt files from:
  - `Tiny-T2VSafetyBench` or `T2VSafetyBench`
- Optional external evaluator command supported via `{manifest}` and `{output}` placeholders.

## Artifact Retention Policy

Top-level suite config:

```yaml
artifacts:
  video: "all"            # all | failures_only | none
  max_samples: 20         # optional
  encode: "h264"          # h264 | h265 (diffusers_img2vid supports h265 via ffmpeg)
  keep_workdir: false     # used by some official metric backends
```

## Diagnostic Report (`report.html`)

Each run writes a `report.html` with:

- **Gate Results** — pass/fail per gate with actual vs target
- **SPRT Analysis** — per-gate LLR trajectory chart (inline SVG), worst-delta table with prompt/seed lookup, sigma, paired count, threshold distances
- **Worst Sample Pairs** — which specific prompt + seed degraded the most (resolved from sample_id)
- **Regression vs Baseline** — delta per metric vs latest passing run, colored by direction

## Trend Report

Generate a cross-run HTML trend report from artifact history:

```bash
python scripts/trend_report.py \
  --model-root artifacts/demo-video-model/regression_core/demo_mock_model \
  --output artifacts/trend_report.html \
  --last-n 30
```

This produces a self-contained HTML with:

- Pass/fail strip (one colored cell per run)
- SVG line chart per metric (green = PASS run, red = FAIL run)
- Run history table

## Webhook Alerts

Receive a POST notification when a gate or regression fails:

```bash
temporalci run examples/regression_sprt.yaml \
  --baseline-mode latest_pass \
  --webhook-url https://hooks.slack.com/services/...
```

The payload is JSON with `run_id`, `status`, `gate_failed`, `regression_failed`, `gate_failures` list, and `run_dir`.  Works with Slack incoming webhooks, Discord webhooks, or any HTTP endpoint.

## Statistical Regression Gate (SPRT)

`method: sprt_regression` adds a sequential hypothesis test on top of the
normal threshold gate. This is useful when a static threshold passes but
sample-level quality shifts indicate statistically significant regression.

Example suite: `examples/regression_sprt.yaml`

Typical usage pattern:

```bash
# 1) establish baseline (example suite is configured to allow bootstrap)
temporalci run examples/regression_sprt.yaml --baseline-mode none

# 2) run candidate against latest passing baseline
temporalci run examples/regression_sprt.yaml --baseline-mode latest_pass
```

Gate parameters:

- `alpha`, `beta`: false-alarm / miss bounds for SPRT
- `effect_size`: minimum practical regression to detect
- `sigma_mode`: `estimate` (sample std) or `fixed` (use provided `sigma`)
- `sigma`: required when `sigma_mode: fixed`
- `sigma_floor`: numerical stability floor for `sigma_mode: estimate`
- `min_pairs`: minimum paired samples before a decision
- `min_paired_ratio`: minimum `paired_count / expected_pairs` required before SPRT can run
- `pairing_mismatch`: `fail | pass | skip` policy when `paired_ratio < min_paired_ratio`

Pairing ratio is computed as:

- `expected_pairs = max(current_series_count, baseline_series_count)`
- `paired_ratio = paired_count / expected_pairs`
- `inconclusive`: `fail` or `pass`
- `require_baseline`: default `true`; if enabled and baseline is missing, gate fails
- `baseline_missing`: `fail | pass | skip` policy for missing baseline handling
- `pairing_mode`: `sample_id` (recommended) or `legacy`

Operational recommendation:

- Use `pairing_mode: sample_id` and keep `require_baseline: true` in production.
- Keep `min_paired_ratio` high (for example `0.95` to `1.0`) to fail on pairing drift.
- Keep `pairing_mismatch: fail` in production.
- Prefer `sigma_mode: fixed` in production for reproducible alpha/beta behavior.
- Calibrate `sigma` and `min_pairs` so expected no-regression runs do not stay inconclusive.
- Use `baseline_missing: skip` only for one-time baseline bootstrap.

`report.html` now includes an **SPRT Analysis** section with decision, LLR thresholds,
pairing ratio, worst sample deltas, and derived diagnostics:
`drift_per_pair`, `required_pairs_upper/lower`, and `llr_per_pair`.

Calibrate fixed-sigma settings from repeated no-change runs:

```bash
python scripts/calibrate_sprt.py \
  --suite examples/regression_sprt.yaml \
  --runs 8 \
  --artifacts-dir artifacts/sprt-calibration
```

This writes `sprt_calibration.json` with recommended `sigma`, estimated required
pair counts, and per-run pairing quality diagnostics.

CLI-integrated equivalent:

```bash
temporalci sprt calibrate --suite examples/regression_sprt.yaml --runs 8
```

Apply calibrated params to a new suite file (safe default):

```bash
python scripts/calibrate_sprt.py \
  --suite examples/regression_sprt.yaml \
  --runs 8 \
  --artifacts-dir artifacts/sprt-calibration \
  --apply-out examples/regression_sprt.calibrated.yaml
```

Run CI-style checks and fail with exit code `2` when calibration quality is insufficient:

```bash
python scripts/calibrate_sprt.py \
  --suite examples/regression_sprt.yaml \
  --runs 8 \
  --artifacts-dir artifacts/sprt-calibration \
  --check \
  --fail-if-no-deltas \
  --min-total-deltas 50 \
  --max-mismatch-runs 0
```

`--apply-inplace` is available for direct updates and creates a `.bak` backup.

CLI-integrated equivalents:

```bash
temporalci sprt apply --suite examples/regression_sprt.yaml --calibration-json artifacts/sprt-calibration/sprt_calibration.json --out examples/regression_sprt.calibrated.yaml
temporalci sprt check --calibration-json artifacts/sprt-calibration/sprt_calibration.json --min-total-deltas 50 --max-mismatch-runs 0
```

Calibration JSON now includes `schema_version`, `tool`, `suite_hash_sha1`, and
`exit_code_semantics` for CI-safe schema stability. `temporalci sprt apply`
now enforces `suite_hash_sha1` match by default and fails with exit code `2`
on mismatch; use `--force` to override intentionally. `temporalci sprt apply`
and `temporalci sprt check` both reject unknown `schema_version` values.

## Distributed Mode

Start coordinator:

```bash
temporalci-distributed serve --host 0.0.0.0 --port 8080
```

Start worker:

```bash
temporalci-distributed worker --coordinator-url http://localhost:8080
```

Useful long-run env knobs:

- `TEMPORALCI_TASK_LEASE_SEC` (default `300`)
- `TEMPORALCI_HEARTBEAT_INTERVAL_SEC` (default `30`)

The worker now sends heartbeats while running, and coordinator can requeue stale tasks.

Bring up infra:

```bash
docker compose -f infra/docker-compose.yml up --build
```

### Shared-FS-Free run submission

`POST /runs` accepts either:

- `suite_path`
- or `suite_yaml` (inline YAML string)

So workers do not require a shared filesystem.

Recommended payload fields for inline suites:

- `suite_yaml`: full suite text
- `suite_root`: base directory used to resolve relative paths in the suite
- `upload_artifacts`: upload run directory to MinIO after completion

Recovery endpoints:

- `POST /admin/requeue_processing` (manual queue drain)
- `POST /admin/requeue_stale` (lease-expired running tasks back to queue)

Example smoke run (requires running coordinator + at least one worker):

```bash
python scripts/smoke_distributed_e2e.py \
  --coordinator-url http://localhost:8080 \
  --suite examples/regression_core.yaml \
  --baseline-mode none
```

Failure-injection recovery E2E (worker kill -> requeue -> recovery):

```bash
python scripts/distributed_recovery_e2e.py \
  --artifacts-dir artifacts/distributed-recovery-e2e \
  --coordinator-port 18080 \
  --task-sleep-sec 20
```

This script can optionally bring up Postgres/Redis via `infra/docker-compose.yml`
unless `--skip-compose` is set. It writes verification output to:

- `artifacts/distributed-recovery-e2e/distributed_recovery_summary.json`
- `artifacts/distributed-recovery-e2e/logs/*.log`

Optional integration test (disabled by default):

```bash
set RUN_DISTRIBUTED_E2E=1&&python -m pytest -q tests\test_distributed_recovery_e2e.py -vv
```

Operational procedure details are documented in `docs/distributed_recovery_runbook.md`.

## CI Workflows

GPU demo manual workflow:

- `.github/workflows/svd-regression-gate-demo.yml`

Distributed recovery workflows:

- `.github/workflows/distributed-recovery-e2e.yml` (manual + nightly schedule)

The GPU workflow runs baseline then candidate on a self-hosted GPU runner and checks expected gate behavior.

Latest local distributed recovery proof snapshot:

- `docs/proofs/distributed_recovery_e2e_20260213.md`

## 96h Autopilot

For unattended long runs on a powered-on machine:

```bash
python scripts/autopilot_96h.py \
  --hours 96 \
  --baseline examples/svd_regression_fast.yaml \
  --candidate examples/svd_regression_fast_degraded.yaml \
  --artifacts-dir artifacts/autopilot \
  --coordinator-url http://localhost:8080
```

Per-cycle results are appended to `artifacts/autopilot/autopilot_runs.jsonl`.

Useful unattended options:

- `--keep-last-runs N` (auto-prune old run directories to control disk usage)
- `--status-file autopilot_status.json` (heartbeat-style status snapshot)

Detached background launch:

```bash
python scripts/launch_autopilot_background.py \
  --hours 96 \
  --artifacts-dir artifacts/autopilot-96h \
  --keep-last-runs 48 \
  --replace-existing
```

Stop detached process:

```bash
python scripts/stop_autopilot_background.py \
  --pid-file artifacts/autopilot-96h/autopilot.pid
```

`stop_autopilot_background.py` is idempotent for common cases:

- Missing `autopilot.pid` + terminal status (`finished/stopped/stale_stopped`) returns success.
- Missing/invalid pid file + `state=running` repairs status to `stopped`.

Health check:

```bash
python scripts/check_autopilot_health.py --artifacts-dir artifacts/autopilot-96h
```

Telemetry sidecar (non-intrusive monitor):

```bash
python scripts/autopilot_telemetry.py \
  --artifacts-dir artifacts/autopilot-96h \
  --interval-sec 60
```

By default this writes `autopilot_telemetry.jsonl` and stops automatically when
the autopilot pid dies or the status reaches a terminal state.

Generate run summary artifacts:

```bash
python scripts/autopilot_summary.py --artifacts-dir artifacts/autopilot-96h
```

This writes `autopilot_summary.json` and `autopilot_summary.md`.

Automatic stale-status repair:

```bash
python scripts/check_autopilot_health.py \
  --artifacts-dir artifacts/autopilot-96h \
  --repair \
  --repair-state stale_stopped
```

For step-by-step operational procedures, see `docs/autopilot_runbook.md`.
