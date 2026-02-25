from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _as_dict(value: Any, *, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be an object")
    return dict(value)


def _as_required_text(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    text = value.strip()
    if text == "":
        raise ValueError(f"{field_name} is required")
    return text


def _as_optional_dict(value: Any, *, field_name: str) -> dict[str, Any] | None:
    if value is None:
        return None
    return _as_dict(value, field_name=field_name)


def _as_list_of_dicts(value: Any, *, field_name: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be an array")
    out: list[dict[str, Any]] = []
    for idx, item in enumerate(value):
        out.append(_as_dict(item, field_name=f"{field_name}[{idx}]"))
    return out


@dataclass(slots=True)
class CampaignManifest:
    campaign_id: str
    generated_at_utc: str
    window: dict[str, Any]
    config: dict[str, Any]
    summary: dict[str, Any]
    submitted_runs: list[dict[str, Any]]
    terminal_runs: list[dict[str, Any]]
    submit_errors: list[dict[str, Any]]
    poll_errors: list[dict[str, Any]]
    sales_packs: list[dict[str, Any]]
    validation: dict[str, Any]
    provenance: dict[str, Any]
    artifacts: dict[str, Any]
    schema_version: str = "campaign_manifest.v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "campaign_id": self.campaign_id,
            "generated_at_utc": self.generated_at_utc,
            "window": dict(self.window),
            "config": dict(self.config),
            "summary": dict(self.summary),
            "submitted_runs": list(self.submitted_runs),
            "terminal_runs": list(self.terminal_runs),
            "submit_errors": list(self.submit_errors),
            "poll_errors": list(self.poll_errors),
            "sales_packs": list(self.sales_packs),
            "validation": dict(self.validation),
            "provenance": dict(self.provenance),
            "artifacts": dict(self.artifacts),
        }

    def to_json_bytes(self) -> bytes:
        return json.dumps(self.to_dict(), ensure_ascii=True, indent=2, sort_keys=True).encode(
            "utf-8"
        )

    def write_json(self, path: Path | str) -> Path:
        target = Path(path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(self.to_json_bytes())
        return target

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CampaignManifest:
        raw = _as_dict(payload, field_name="manifest")
        schema_version = _as_required_text(raw.get("schema_version"), field_name="schema_version")
        if schema_version != "campaign_manifest.v1":
            raise ValueError("schema_version must be campaign_manifest.v1")
        return cls(
            schema_version=schema_version,
            campaign_id=_as_required_text(raw.get("campaign_id"), field_name="campaign_id"),
            generated_at_utc=_as_required_text(
                raw.get("generated_at_utc"), field_name="generated_at_utc"
            ),
            window=_as_dict(raw.get("window"), field_name="window"),
            config=_as_dict(raw.get("config"), field_name="config"),
            summary=_as_dict(raw.get("summary"), field_name="summary"),
            submitted_runs=_as_list_of_dicts(
                raw.get("submitted_runs"), field_name="submitted_runs"
            ),
            terminal_runs=_as_list_of_dicts(raw.get("terminal_runs"), field_name="terminal_runs"),
            submit_errors=_as_list_of_dicts(raw.get("submit_errors"), field_name="submit_errors"),
            poll_errors=_as_list_of_dicts(raw.get("poll_errors"), field_name="poll_errors"),
            sales_packs=_as_list_of_dicts(raw.get("sales_packs"), field_name="sales_packs"),
            validation=_as_dict(raw.get("validation"), field_name="validation"),
            provenance=_as_dict(raw.get("provenance"), field_name="provenance"),
            artifacts=_as_dict(raw.get("artifacts"), field_name="artifacts"),
        )

    @classmethod
    def from_file(cls, path: Path | str) -> CampaignManifest:
        source = Path(path).expanduser().resolve()
        payload = json.loads(source.read_text(encoding="utf-8"))
        return cls.from_dict(payload)


@dataclass(slots=True)
class CampaignProvenance:
    generated_at_utc: str
    campaign_id: str
    temporalci: dict[str, Any]
    runtime: dict[str, Any]
    git: dict[str, Any] | None
    config: dict[str, Any]
    suite_snapshots: list[dict[str, Any]]
    artifacts: dict[str, Any]
    schema_version: str = "campaign_provenance.v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "generated_at_utc": self.generated_at_utc,
            "campaign_id": self.campaign_id,
            "temporalci": dict(self.temporalci),
            "runtime": dict(self.runtime),
            "git": (None if self.git is None else dict(self.git)),
            "config": dict(self.config),
            "suite_snapshots": list(self.suite_snapshots),
            "artifacts": dict(self.artifacts),
        }

    def to_json_bytes(self) -> bytes:
        return json.dumps(self.to_dict(), ensure_ascii=True, indent=2, sort_keys=True).encode(
            "utf-8"
        )

    def write_json(self, path: Path | str) -> Path:
        target = Path(path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(self.to_json_bytes())
        return target

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CampaignProvenance:
        raw = _as_dict(payload, field_name="provenance")
        schema_version = _as_required_text(raw.get("schema_version"), field_name="schema_version")
        if schema_version != "campaign_provenance.v1":
            raise ValueError("schema_version must be campaign_provenance.v1")
        return cls(
            schema_version=schema_version,
            generated_at_utc=_as_required_text(
                raw.get("generated_at_utc"), field_name="generated_at_utc"
            ),
            campaign_id=_as_required_text(raw.get("campaign_id"), field_name="campaign_id"),
            temporalci=_as_dict(raw.get("temporalci"), field_name="temporalci"),
            runtime=_as_dict(raw.get("runtime"), field_name="runtime"),
            git=_as_optional_dict(raw.get("git"), field_name="git"),
            config=_as_dict(raw.get("config"), field_name="config"),
            suite_snapshots=_as_list_of_dicts(
                raw.get("suite_snapshots"), field_name="suite_snapshots"
            ),
            artifacts=_as_dict(raw.get("artifacts"), field_name="artifacts"),
        )

    @classmethod
    def from_file(cls, path: Path | str) -> CampaignProvenance:
        source = Path(path).expanduser().resolve()
        payload = json.loads(source.read_text(encoding="utf-8"))
        return cls.from_dict(payload)
