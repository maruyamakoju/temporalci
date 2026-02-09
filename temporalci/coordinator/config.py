from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(slots=True)
class CoordinatorSettings:
    postgres_dsn: str
    redis_url: str
    queue_name: str
    processing_queue_name: str
    default_artifacts_dir: str
    minio_endpoint: str
    minio_access_key: str
    minio_secret_key: str
    minio_bucket: str
    minio_secure: bool
    task_lease_sec: int
    heartbeat_interval_sec: int

    @classmethod
    def from_env(cls) -> "CoordinatorSettings":
        queue_name = os.getenv("TEMPORALCI_QUEUE_NAME", "temporalci:tasks")
        return cls(
            postgres_dsn=os.getenv(
                "TEMPORALCI_POSTGRES_DSN",
                "postgresql://temporalci:temporalci@localhost:5432/temporalci",
            ),
            redis_url=os.getenv("TEMPORALCI_REDIS_URL", "redis://localhost:6379/0"),
            queue_name=queue_name,
            processing_queue_name=os.getenv(
                "TEMPORALCI_PROCESSING_QUEUE_NAME",
                f"{queue_name}:processing",
            ),
            default_artifacts_dir=os.getenv("TEMPORALCI_ARTIFACTS_DIR", "artifacts"),
            minio_endpoint=os.getenv("TEMPORALCI_MINIO_ENDPOINT", "localhost:9000"),
            minio_access_key=os.getenv("TEMPORALCI_MINIO_ACCESS_KEY", "minioadmin"),
            minio_secret_key=os.getenv("TEMPORALCI_MINIO_SECRET_KEY", "minioadmin"),
            minio_bucket=os.getenv("TEMPORALCI_MINIO_BUCKET", "temporalci-artifacts"),
            minio_secure=os.getenv("TEMPORALCI_MINIO_SECURE", "false").lower() == "true",
            task_lease_sec=max(30, int(os.getenv("TEMPORALCI_TASK_LEASE_SEC", "300"))),
            heartbeat_interval_sec=max(
                5, int(os.getenv("TEMPORALCI_HEARTBEAT_INTERVAL_SEC", "30"))
            ),
        )
