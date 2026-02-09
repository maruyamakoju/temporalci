from __future__ import annotations

from pathlib import Path
from typing import Any

from temporalci.coordinator.config import CoordinatorSettings


def _load_boto3() -> Any:
    try:
        import boto3
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "boto3 is required for MinIO artifact uploads. "
            "Install optional dependencies for distributed mode."
        ) from exc
    return boto3


class MinioArtifactUploader:
    def __init__(self, settings: CoordinatorSettings) -> None:
        boto3 = _load_boto3()
        self.bucket = settings.minio_bucket
        self.client = boto3.client(
            "s3",
            endpoint_url=(
                f"https://{settings.minio_endpoint}"
                if settings.minio_secure
                else f"http://{settings.minio_endpoint}"
            ),
            aws_access_key_id=settings.minio_access_key,
            aws_secret_access_key=settings.minio_secret_key,
        )
        self._ensure_bucket_exists()

    def _ensure_bucket_exists(self) -> None:
        buckets = self.client.list_buckets()
        names = {bucket["Name"] for bucket in buckets.get("Buckets", [])}
        if self.bucket not in names:
            self.client.create_bucket(Bucket=self.bucket)

    def upload_run_directory(self, run_dir: str | Path, prefix: str) -> list[str]:
        root = Path(run_dir)
        uploaded: list[str] = []
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            key = f"{prefix}/{path.relative_to(root).as_posix()}"
            self.client.upload_file(str(path), self.bucket, key)
            uploaded.append(key)
        return uploaded
