# Distributed Recovery E2E Proof (2026-02-13)

## Scope

- Scenario: `worker kill -> requeue_processing -> worker replacement -> completion`
- Script: `scripts/distributed_recovery_e2e.py`
- Artifacts:
  - `artifacts/distributed-recovery-e2e-proof/distributed_recovery_summary.json`
  - `artifacts/distributed-recovery-e2e-proof/logs/coordinator.log`
  - `artifacts/distributed-recovery-e2e-proof/logs/worker1.log`
  - `artifacts/distributed-recovery-e2e-proof/logs/worker2.log`

## Command

```bash
python scripts/distributed_recovery_e2e.py \
  --skip-compose \
  --postgres-dsn postgresql://temporalci:temporalci@localhost:55432/temporalci \
  --redis-url redis://localhost:56379/0 \
  --task-sleep-sec 10 \
  --completion-timeout-sec 240 \
  --wait-running-timeout-sec 90 \
  --artifacts-dir artifacts/distributed-recovery-e2e-proof
```

## Result Summary

- `status`: `ok`
- `run.final_snapshot.status`: `completed`
- `run.final_snapshot.payload.status`: `PASS`
- `requeue_processing.moved`: `1`
- `requeue_stale.moved`: `0`
- `verification.task_attempts`: `2`
- `verification.queue_depth`: `0`
- `verification.processing_depth`: `0`

## Commits

- `5e2d31f` (`Add_distributed_recovery_e2e_and_fix_coordinator_entrypoints`)
- `e80033a` (`Add_autopilot_cycle_telemetry_and_mock_delay_control`)
