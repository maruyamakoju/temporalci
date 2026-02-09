from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Print file with line numbers")
    parser.add_argument("path", help="File path")
    args = parser.parse_args()

    path = Path(args.path)
    text = path.read_text(encoding="utf-8")
    for idx, line in enumerate(text.splitlines(), start=1):
        print(f"{idx}:{line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
