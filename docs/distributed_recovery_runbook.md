# Distributed Recovery Runbook

This runbook validates that distributed execution recovers from worker failure:

1. submit a run
2. kill worker during processing
3. requeue processing/stale tasks
4. complete on a replacement worker

Prerequisite:

```bash
python -m pip install -e .[distributed]
```

## One-command Scenario

```bash
python scripts/distributed_recovery_e2e.py \
  --artifacts-dir artifacts/distributed-recovery-e2e \
  --coordinator-port 18080 \
  --task-sleep-sec 20
```

By default the script:

- starts Postgres/Redis via `infra/docker-compose.yml` only when ports are not already reachable
- starts a local coordinator (`temporalci.coordinator.cli serve`)
- starts `worker1`, kills it during task execution
- calls:
  - `POST /admin/requeue_processing`
  - `POST /admin/requeue_stale`
- starts `worker2` and waits for completion
- verifies DB/Redis state:
  - task is `completed`
  - task `attempts >= 2`
  - queue and processing queue depths are both `0`

## Useful Flags

- `--skip-compose`: assume Postgres/Redis are already running
- `--use-existing-coordinator`: target an already running coordinator
- `--coordinator-url`: explicit coordinator endpoint
- `--postgres-dsn`: DB connection string for verification
- `--redis-url`: Redis connection string for verification
- `--compose-down`: run `docker compose down` when the script started services

When `--use-existing-coordinator` is used without `--queue-name`, the script defaults
to `temporalci:tasks` and relaxes queue-empty checks (shared queue may have other jobs).

## Outputs

- summary JSON:
  - `artifacts/distributed-recovery-e2e/distributed_recovery_summary.json`
- process logs:
  - `artifacts/distributed-recovery-e2e/logs/coordinator.log`
  - `artifacts/distributed-recovery-e2e/logs/worker1.log`
  - `artifacts/distributed-recovery-e2e/logs/worker2.log`

## Integration Test Gate

Optional heavy test:

```bash
set RUN_DISTRIBUTED_E2E=1&&python -m pytest -q tests\test_distributed_recovery_e2e.py -vv
```

Optional env toggles:

- `RUN_DISTRIBUTED_E2E_USE_COMPOSE=1`
- `RUN_DISTRIBUTED_E2E_POSTGRES_DSN=...`
- `RUN_DISTRIBUTED_E2E_REDIS_URL=...`
- `RUN_DISTRIBUTED_E2E_COORDINATOR_URL=...`
- `RUN_DISTRIBUTED_E2E_QUEUE_NAME=...`
