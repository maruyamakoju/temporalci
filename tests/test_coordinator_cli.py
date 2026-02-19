"""Tests for temporalci.coordinator.cli.

Covers _build_parser() and main() with 'serve' and 'worker' subcommands.
Uses monkeypatching so no real uvicorn or CoordinatorWorker connections
are made during testing.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from temporalci.coordinator.cli import _build_parser, main


# ---------------------------------------------------------------------------
# _build_parser
# ---------------------------------------------------------------------------


def test_build_parser_serve_defaults() -> None:
    parser = _build_parser()
    args = parser.parse_args(["serve"])
    assert args.command == "serve"
    assert args.host == "0.0.0.0"
    assert args.port == 8080
    assert args.reload is False


def test_build_parser_serve_custom_args() -> None:
    parser = _build_parser()
    args = parser.parse_args(["serve", "--host", "127.0.0.1", "--port", "9090", "--reload"])
    assert args.host == "127.0.0.1"
    assert args.port == 9090
    assert args.reload is True


def test_build_parser_worker_defaults() -> None:
    parser = _build_parser()
    args = parser.parse_args(["worker"])
    assert args.command == "worker"
    assert args.coordinator_url == "http://localhost:8080"
    assert args.worker_id is None
    assert args.poll_interval_sec == pytest.approx(2.0)


def test_build_parser_worker_custom_args() -> None:
    parser = _build_parser()
    args = parser.parse_args(
        [
            "worker",
            "--coordinator-url",
            "http://myhost:9000",
            "--worker-id",
            "w42",
            "--poll-interval-sec",
            "5.0",
        ]
    )
    assert args.coordinator_url == "http://myhost:9000"
    assert args.worker_id == "w42"
    assert args.poll_interval_sec == pytest.approx(5.0)


def test_build_parser_requires_subcommand() -> None:
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


# ---------------------------------------------------------------------------
# main() — serve subcommand
# ---------------------------------------------------------------------------


def test_main_serve_uvicorn_missing_raises_systemexit(monkeypatch: Any) -> None:
    """When uvicorn is not installed, main(['serve']) should raise SystemExit(1)."""
    # Temporarily hide uvicorn by replacing it with None in sys.modules
    # (importing a None-valued module raises ImportError)
    original = sys.modules.pop("uvicorn", None)
    monkeypatch.setitem(sys.modules, "uvicorn", None)  # type: ignore[arg-type]
    try:
        with pytest.raises(SystemExit) as exc_info:
            main(["serve"])
        assert exc_info.value.code == 1
    finally:
        if original is not None:
            sys.modules["uvicorn"] = original
        elif "uvicorn" in sys.modules:
            del sys.modules["uvicorn"]


def test_main_serve_calls_uvicorn_run(monkeypatch: Any) -> None:
    """When uvicorn is available, main(['serve']) should call uvicorn.run and return 0."""
    run_calls: list[dict] = []

    mock_uvicorn = types.ModuleType("uvicorn")
    mock_uvicorn.run = lambda *args, **kwargs: run_calls.append({"args": args, "kwargs": kwargs})  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "uvicorn", mock_uvicorn)

    code = main(["serve", "--host", "127.0.0.1", "--port", "9999"])
    assert code == 0
    assert len(run_calls) == 1
    call = run_calls[0]
    assert call["kwargs"]["host"] == "127.0.0.1"
    assert call["kwargs"]["port"] == 9999
    assert call["kwargs"]["factory"] is True


def test_main_serve_reload_flag_passed(monkeypatch: Any) -> None:
    run_calls: list[dict] = []

    mock_uvicorn = types.ModuleType("uvicorn")
    mock_uvicorn.run = lambda *args, **kwargs: run_calls.append(kwargs)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "uvicorn", mock_uvicorn)

    main(["serve", "--reload"])
    assert run_calls[0]["reload"] is True


def test_main_serve_default_no_reload(monkeypatch: Any) -> None:
    run_calls: list[dict] = []

    mock_uvicorn = types.ModuleType("uvicorn")
    mock_uvicorn.run = lambda *args, **kwargs: run_calls.append(kwargs)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "uvicorn", mock_uvicorn)

    main(["serve"])
    assert run_calls[0]["reload"] is False


# ---------------------------------------------------------------------------
# main() — worker subcommand
# ---------------------------------------------------------------------------


def test_main_worker_creates_worker_and_runs_forever(monkeypatch: Any) -> None:
    """main(['worker']) should instantiate CoordinatorWorker and call run_forever()."""
    init_kwargs: dict[str, Any] = {}
    run_forever_calls: list[bool] = []

    class MockWorker:
        def __init__(self, **kwargs: Any) -> None:
            init_kwargs.update(kwargs)

        def run_forever(self) -> None:
            run_forever_calls.append(True)

    monkeypatch.setattr("temporalci.coordinator.cli.CoordinatorWorker", MockWorker)

    code = main(["worker"])
    assert code == 0
    assert run_forever_calls == [True]


def test_main_worker_passes_coordinator_url(monkeypatch: Any) -> None:
    init_kwargs: dict[str, Any] = {}

    class MockWorker:
        def __init__(self, **kwargs: Any) -> None:
            init_kwargs.update(kwargs)

        def run_forever(self) -> None:
            pass

    monkeypatch.setattr("temporalci.coordinator.cli.CoordinatorWorker", MockWorker)

    main(["worker", "--coordinator-url", "http://ci-server:8080"])
    assert init_kwargs["coordinator_url"] == "http://ci-server:8080"


def test_main_worker_passes_worker_id(monkeypatch: Any) -> None:
    init_kwargs: dict[str, Any] = {}

    class MockWorker:
        def __init__(self, **kwargs: Any) -> None:
            init_kwargs.update(kwargs)

        def run_forever(self) -> None:
            pass

    monkeypatch.setattr("temporalci.coordinator.cli.CoordinatorWorker", MockWorker)

    main(["worker", "--worker-id", "node-007"])
    assert init_kwargs["worker_id"] == "node-007"


def test_main_worker_passes_poll_interval(monkeypatch: Any) -> None:
    init_kwargs: dict[str, Any] = {}

    class MockWorker:
        def __init__(self, **kwargs: Any) -> None:
            init_kwargs.update(kwargs)

        def run_forever(self) -> None:
            pass

    monkeypatch.setattr("temporalci.coordinator.cli.CoordinatorWorker", MockWorker)

    main(["worker", "--poll-interval-sec", "10.0"])
    assert init_kwargs["poll_interval_sec"] == pytest.approx(10.0)


def test_main_worker_default_worker_id_is_none(monkeypatch: Any) -> None:
    init_kwargs: dict[str, Any] = {}

    class MockWorker:
        def __init__(self, **kwargs: Any) -> None:
            init_kwargs.update(kwargs)

        def run_forever(self) -> None:
            pass

    monkeypatch.setattr("temporalci.coordinator.cli.CoordinatorWorker", MockWorker)

    main(["worker"])
    assert init_kwargs["worker_id"] is None
