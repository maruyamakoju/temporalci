"""Computer vision pipeline for catenary infrastructure inspection.

Provides three layers:
1. **Semantic segmentation** — pixel-level classification of vegetation,
   sky, infrastructure, and ground using SegFormer.
2. **Monocular depth estimation** — relative distance estimation using
   Depth Anything V2.
3. **Clearance calculation** — combines segmentation + depth + wire
   detection to estimate minimum vegetation-to-catenary distance.
"""

from __future__ import annotations
