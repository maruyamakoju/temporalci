from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from temporalci.coordinator.store import CoordinatorStore


@dataclass
class _FakeRedis:
    processing_items: list[str]
    queue_items: list[str]
    lrem_calls: list[tuple[str, int, str]]

    def __init__(self, processing_items: list[str] | None = None) -> None:
        self.processing_items = list(processing_items or [])
        self.queue_items = []
        self.lrem_calls = []

    def rpop(self, _queue: str) -> str | None:
        if not self.processing_items:
            return None
        return self.processing_items.pop(0)

    def lpush(self, _queue: str, task_id: str) -> int:
        self.queue_items.append(task_id)
        return len(self.queue_items)

    def lrem(self, queue: str, count: int, task_id: str) -> int:
        self.lrem_calls.append((queue, count, task_id))
        return 1


class _FakeCursor:
    def __init__(self, rows: list[tuple[str]]) -> None:
        self._rows = rows

    def __enter__(self) -> "_FakeCursor":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        return None

    def execute(self, _sql: str, _params: tuple[int]) -> None:
        return None

    def fetchall(self) -> list[tuple[str]]:
        return list(self._rows)


class _FakeConn:
    def __init__(self, rows: list[tuple[str]]) -> None:
        self._rows = rows

    def cursor(self) -> _FakeCursor:
        return _FakeCursor(self._rows)


def _store_with_fakes(*, processing_items: list[str], stale_rows: list[tuple[str]]) -> CoordinatorStore:
    store = CoordinatorStore.__new__(CoordinatorStore)
    store.settings = SimpleNamespace(
        queue_name="queue",
        processing_queue_name="processing",
    )
    store._redis = _FakeRedis(processing_items=processing_items)
    store._conn = _FakeConn(rows=stale_rows)
    return store


def test_requeue_processing_skips_db_miss() -> None:
    store = _store_with_fakes(processing_items=["task-1", "task-2"], stale_rows=[])

    def fake_mark_task_queued(*, task_id: str) -> bool:
        return task_id == "task-1"

    store._mark_task_queued = fake_mark_task_queued  # type: ignore[attr-defined]

    moved = store.requeue_processing_tasks(limit=10)
    assert moved == 1
    assert store._redis.queue_items == ["task-1"]  # type: ignore[attr-defined]


def test_requeue_stale_running_skips_db_miss() -> None:
    store = _store_with_fakes(
        processing_items=[],
        stale_rows=[("task-a",), ("task-b",), ("task-c",)],
    )

    def fake_mark_task_queued(*, task_id: str) -> bool:
        return task_id in {"task-a", "task-c"}

    store._mark_task_queued = fake_mark_task_queued  # type: ignore[attr-defined]

    moved = store.requeue_stale_running_tasks(limit=10)
    assert moved == 2
    assert store._redis.queue_items == ["task-a", "task-c"]  # type: ignore[attr-defined]
    assert len(store._redis.lrem_calls) == 3  # type: ignore[attr-defined]
