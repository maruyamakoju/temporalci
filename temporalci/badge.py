"""Status badge SVG generator for TemporalCI.

Auto-generated after every run at ``model_root/badge.svg``.

Public API
----------
write_badge_svg(path, status)   ->  None
"""

from __future__ import annotations

from pathlib import Path

_PASS_COLOR = "#2da44e"  # GitHub green
_FAIL_COLOR = "#cf222e"  # GitHub red
_LABEL_COLOR = "#555"

# Shields.io-compatible flat badge template.
# All measurements are in SVG user units (px).  Text is scaled by 0.1 so that
# font-size="110" renders as 11 px — the canonical shields.io trick.
_BADGE_TEMPLATE = """\
<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"
     width="{total_w}" height="20" role="img" aria-label="TemporalCI: {status}">
  <title>TemporalCI: {status}</title>
  <linearGradient id="s" x2="0" y2="100%">
    <stop offset="0"  stop-color="#bbb" stop-opacity=".1"/>
    <stop offset="1"  stop-opacity=".1"/>
  </linearGradient>
  <clipPath id="r">
    <rect width="{total_w}" height="20" rx="3" fill="#fff"/>
  </clipPath>
  <g clip-path="url(#r)">
    <rect width="{label_w}"  height="20" fill="{label_color}"/>
    <rect x="{label_w}" width="{status_w}" height="20" fill="{status_color}"/>
    <rect width="{total_w}" height="20" fill="url(#s)"/>
  </g>
  <g fill="#fff" text-anchor="middle"
     font-family="DejaVu Sans,Verdana,Geneva,sans-serif" font-size="110">
    <text x="{label_cx}" y="150" fill="#010101" fill-opacity=".3"
          transform="scale(.1)" textLength="{label_text_w}" lengthAdjust="spacing">{label}</text>
    <text x="{label_cx}" y="140" transform="scale(.1)"
          textLength="{label_text_w}" lengthAdjust="spacing">{label}</text>
    <text x="{status_cx}" y="150" fill="#010101" fill-opacity=".3"
          transform="scale(.1)" textLength="{status_text_w}" lengthAdjust="spacing">{status}</text>
    <text x="{status_cx}" y="140" transform="scale(.1)"
          textLength="{status_text_w}" lengthAdjust="spacing">{status}</text>
  </g>
</svg>
"""

# Average character width in px at 11 px font size (DejaVu Sans approximation)
_CH_W = 6.5
_PAD = 10  # horizontal padding per section


def _section_width(text: str) -> int:
    """Pixel width of one badge section (rounded up to nearest int)."""
    return max(int(len(text) * _CH_W + _PAD), 20)


def write_badge_svg(path: Path, status: str) -> None:
    """Write a shields.io-style status badge SVG to *path*.

    ``status`` should be ``"PASS"`` or ``"FAIL"`` (case-insensitive).
    Any other value produces a grey badge labeled with the raw status string.
    """
    label = "TemporalCI"
    status_upper = str(status).upper()

    if status_upper == "PASS":
        status_color = _PASS_COLOR
    elif status_upper == "FAIL":
        status_color = _FAIL_COLOR
    else:
        status_color = "#9f9f9f"

    label_w = _section_width(label)
    status_w = _section_width(status_upper)
    total_w = label_w + status_w

    # Text centres — in the "scaled by 10" coordinate system used by the template.
    label_cx = (label_w // 2) * 10
    status_cx = (label_w + status_w // 2) * 10

    # textLength keeps text from overflowing narrow sections.
    label_text_w = (label_w - _PAD) * 10
    status_text_w = (status_w - _PAD) * 10

    svg = _BADGE_TEMPLATE.format(
        total_w=total_w,
        label_w=label_w,
        status_w=status_w,
        label_color=_LABEL_COLOR,
        status_color=status_color,
        label=label,
        status=status_upper,
        label_cx=label_cx,
        status_cx=status_cx,
        label_text_w=label_text_w,
        status_text_w=status_text_w,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg, encoding="utf-8")
