"""Utilities for autopilot long-run automation.

JSON I/O and time helpers are re-exported from :mod:`temporalci.utils` for
backward compatibility.  Process management helpers remain here.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from pathlib import Path

# Re-export shared helpers so existing callers continue to work.
from temporalci.utils import (  # noqa: F401
    atomic_write_json,
    read_json_dict,
    safe_write_json,
    utc_now_iso,
)


def pid_exists(pid: int) -> bool:
    """Return ``True`` if a process with *pid* appears to be alive."""
    if pid <= 0:
        return False
    if sys.platform == "win32":
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
            return bool(ctypes.windll.kernel32.GetLastError() == 5)
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
    """Attempt to terminate a process tree rooted at *pid*."""
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

    if sys.platform == "win32":
        return _taskkill(pid)

    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        return False
    return not pid_exists(pid)
