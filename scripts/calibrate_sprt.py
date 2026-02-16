from __future__ import annotations

from temporalci.sprt_calibration import calibrate_main


def main(argv: list[str] | None = None) -> int:
    return calibrate_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
