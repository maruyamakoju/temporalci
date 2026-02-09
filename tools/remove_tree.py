from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


def _onerror(func, path, exc_info):  # type: ignore[no-untyped-def]
    try:
        os.chmod(path, 0o700)
    except OSError:
        pass
    func(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Remove directory tree")
    parser.add_argument("path")
    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists():
        return 0
    shutil.rmtree(path, onerror=_onerror)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
