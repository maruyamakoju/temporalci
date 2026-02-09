from __future__ import annotations

import json
import os
import signal
import subprocess
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json_dict(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    return payload if isinstance(payload, dict) else None


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temp.replace(path)


def safe_write_json(path: Path, payload: dict[str, Any]) -> bool:
    try:
        atomic_write_json(path, payload)
        return True
    except Exception:  # noqa: BLE001
        return False


def pid_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        try:
            import ctypes

            process_query_limited_information = 0x1000
            handle = ctypes.windll.kernel32.OpenProcess(
                process_query_limited_information,
                False,
                pid,
            )
            if handle:
                ctypes.windll.kernel32.CloseHandle(handle)
                return True
            return ctypes.windll.kernel32.GetLastError() == 5
        except Exception:  # noqa: BLE001
            pass
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except (OSError, ValueError, SystemError):
        return False
    return True


def _taskkill(pid: int) -> bool:
    system_root = Path(os.environ.get("SystemRoot", r"C:\Windows"))
    taskkill = system_root / "System32" / "taskkill.exe"
    if not taskkill.exists():
        return False
    proc = subprocess.run(  # noqa: S603
        [str(taskkill), "/PID", str(pid), "/T", "/F"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return proc.returncode == 0


def terminate_pid(pid: int, *, timeout_sec: float = 10.0) -> bool:
    if pid <= 0:
        return False
    try:
        import psutil
    except Exception:
        psutil = None

    if psutil is not None:
        try:
            proc = psutil.Process(pid)
        except psutil.Error:
            return False
        processes = [proc, *proc.children(recursive=True)]
        for child in processes:
            try:
                child.terminate()
            except psutil.Error:
                pass
        _, alive = psutil.wait_procs(processes, timeout=timeout_sec)
        if alive:
            for survivor in alive:
                try:
                    survivor.kill()
                except psutil.Error:
                    pass
            _, alive = psutil.wait_procs(alive, timeout=timeout_sec)
        return not alive

    if os.name == "nt":
        return _taskkill(pid)

    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        return False
    return not pid_exists(pid)
