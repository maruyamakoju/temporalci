# Autopilot Runbook

This runbook covers launch, health checks, stale repair, and stop/recovery for detached TemporalCI autopilot runs.

## Paths

- Default artifacts root: `artifacts/autopilot-96h`
- PID metadata: `artifacts/autopilot-96h/autopilot.pid`
- Status: `artifacts/autopilot-96h/autopilot_status.json`
- Cycle log: `artifacts/autopilot-96h/autopilot_runs.jsonl`
- Session log: `artifacts/autopilot-96h/session.log`

## Launch

```bash
python scripts/launch_autopilot_background.py \
  --hours 96 \
  --artifacts-dir artifacts/autopilot-96h \
  --keep-last-runs 48 \
  --replace-existing
```

If you need to disable per-cycle memory cleanup for debugging:

```bash
python scripts/launch_autopilot_background.py \
  --artifacts-dir artifacts/autopilot-96h \
  --skip-memory-cleanup \
  --replace-existing
```

## Health Check

Human-readable:

```bash
python scripts/check_autopilot_health.py --artifacts-dir artifacts/autopilot-96h
```

Machine-readable:

```bash
python scripts/check_autopilot_health.py --artifacts-dir artifacts/autopilot-96h --json
```

Key fields:

- `healthy`: `true` only when `pid_alive=true`, `state=running`, and status age is not stale.
- `stale`: status age exceeded `--max-stale-sec` (default `1800`).
- `last_runs_line`: last JSONL line for quick inspection.

## Stale Repair

Use repair when status is `running` but process is dead/stale:

```bash
python scripts/check_autopilot_health.py \
  --artifacts-dir artifacts/autopilot-96h \
  --repair \
  --repair-state stale_stopped
```

Repair writes terminal metadata to `autopilot_status.json`:

- `state`: `stale_stopped` (or `stopped` if configured)
- `repair_reason`
- `repair_source`
- `finished_at_utc`

## Stop

```bash
python scripts/stop_autopilot_background.py \
  --pid-file artifacts/autopilot-96h/autopilot.pid
```

Behavior:

- If PID is alive: terminates process tree and writes `state=stopped`.
- If PID file exists but process is already dead: marks `state=stopped`.
- If PID file is missing and status is already terminal: returns success.
- If PID file is missing/invalid and status is `running`: repairs status to `stopped`.

## Recovery Playbook

1. Run health check (`--json` preferred).
2. If `healthy=true`: no action.
3. If `state=running` and `pid_alive=false`: run stale repair.
4. If repair succeeds and run should continue: relaunch with `--replace-existing`.
5. If repeated memory errors appear:
   - keep `--keep-last-runs` enabled
   - inspect `session.log`
   - reduce suite load or temporarily use smaller suite
6. For distributed mode queue stalls:
   - `POST /admin/requeue_stale`
   - `POST /admin/requeue_processing`

## Optional Distributed Notes

- Lease and heartbeat env knobs:
  - `TEMPORALCI_TASK_LEASE_SEC`
  - `TEMPORALCI_HEARTBEAT_INTERVAL_SEC`
- Requeue endpoints should be used before manual DB edits.
