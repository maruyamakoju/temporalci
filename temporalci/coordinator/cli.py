from __future__ import annotations

import argparse

from temporalci.coordinator.worker import CoordinatorWorker


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="temporalci-distributed",
        description="TemporalCI distributed coordinator/worker helper",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="Start FastAPI coordinator server")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", default=8080, type=int)
    serve.add_argument("--reload", action="store_true")

    worker = sub.add_parser("worker", help="Start polling worker")
    worker.add_argument("--coordinator-url", default="http://localhost:8080")
    worker.add_argument("--worker-id", default=None)
    worker.add_argument("--poll-interval-sec", type=float, default=2.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "serve":
        try:
            import uvicorn
        except Exception as exc:  # noqa: BLE001
            print(
                "uvicorn is required to run coordinator server. "
                "Install optional distributed dependencies."
            )
            raise SystemExit(1) from exc

        uvicorn.run(
            "temporalci.coordinator.app:create_app",
            host=args.host,
            port=args.port,
            reload=bool(args.reload),
            factory=True,
        )
        return 0

    if args.command == "worker":
        worker = CoordinatorWorker(
            coordinator_url=args.coordinator_url,
            worker_id=args.worker_id,
            poll_interval_sec=args.poll_interval_sec,
        )
        worker.run_forever()
        return 0

    parser.error(f"unknown command: {args.command}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
