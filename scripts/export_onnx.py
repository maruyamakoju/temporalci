#!/usr/bin/env python3
"""Export SegFormer and Depth Anything V2 models to ONNX format.

Usage::

    python scripts/export_onnx.py --output-dir models/onnx
    python scripts/export_onnx.py --output-dir models/onnx --quantize int8
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch


def export_segformer(output_dir: Path, *, input_h: int = 512, input_w: int = 512) -> Path:
    """Export SegFormer-B0 to ONNX."""
    from transformers import SegformerForSemanticSegmentation

    print("Loading SegFormer-B0...")
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b0-finetuned-ade-512-512",
    )
    model.eval()

    dummy = torch.randn(1, 3, input_h, input_w)
    onnx_path = output_dir / "segformer_b0_ade20k.onnx"

    print(f"Exporting to {onnx_path}...")
    torch.onnx.export(
        model,
        (dummy,),
        str(onnx_path),
        input_names=["pixel_values"],
        output_names=["logits"],
        dynamic_axes={
            "pixel_values": {0: "batch", 2: "height", 3: "width"},
            "logits": {0: "batch", 2: "out_h", 3: "out_w"},
        },
        opset_version=17,
    )
    size_mb = onnx_path.stat().st_size / (1024 * 1024)
    print(f"  SegFormer ONNX: {size_mb:.1f} MB")
    return onnx_path


def export_depth_anything(output_dir: Path, *, input_h: int = 518, input_w: int = 518) -> Path:
    """Export Depth Anything V2 Small to ONNX."""
    from transformers import AutoModelForDepthEstimation

    print("Loading Depth Anything V2 Small...")
    model = AutoModelForDepthEstimation.from_pretrained(
        "depth-anything/Depth-Anything-V2-Small-hf",
    )
    model.eval()

    dummy = torch.randn(1, 3, input_h, input_w)
    onnx_path = output_dir / "depth_anything_v2_small.onnx"

    print(f"Exporting to {onnx_path}...")
    torch.onnx.export(
        model,
        (dummy,),
        str(onnx_path),
        input_names=["pixel_values"],
        output_names=["predicted_depth"],
        dynamic_axes={
            "pixel_values": {0: "batch", 2: "height", 3: "width"},
            "predicted_depth": {0: "batch", 2: "out_h", 3: "out_w"},
        },
        opset_version=17,
    )
    size_mb = onnx_path.stat().st_size / (1024 * 1024)
    print(f"  Depth Anything ONNX: {size_mb:.1f} MB")
    return onnx_path


def quantize_int8(onnx_path: Path) -> Path:
    """Apply INT8 dynamic quantization to an ONNX model."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quant_path = onnx_path.with_suffix(".int8.onnx")
    print(f"Quantizing {onnx_path.name} → {quant_path.name}...")
    quantize_dynamic(
        str(onnx_path),
        str(quant_path),
        weight_type=QuantType.QInt8,
    )
    orig_mb = onnx_path.stat().st_size / (1024 * 1024)
    quant_mb = quant_path.stat().st_size / (1024 * 1024)
    ratio = quant_mb / orig_mb * 100
    print(f"  {orig_mb:.1f} MB → {quant_mb:.1f} MB ({ratio:.0f}%)")
    return quant_path


def benchmark(
    onnx_path: Path,
    *,
    input_shape: tuple[int, ...],
    n_warmup: int = 3,
    n_runs: int = 10,
    provider: str = "CPUExecutionProvider",
) -> float:
    """Benchmark ONNX model inference speed."""
    import onnxruntime as ort

    try:
        sess = ort.InferenceSession(str(onnx_path), providers=[provider])
    except Exception as exc:  # noqa: BLE001
        print(f"  {onnx_path.name} [{provider}]: SKIPPED ({exc})")
        return -1.0

    input_name = sess.get_inputs()[0].name
    dummy = np.random.randn(*input_shape).astype(np.float32)

    for _ in range(n_warmup):
        sess.run(None, {input_name: dummy})

    t0 = time.perf_counter()
    for _ in range(n_runs):
        sess.run(None, {input_name: dummy})
    elapsed = (time.perf_counter() - t0) / n_runs

    print(f"  {onnx_path.name} [{provider}]: {elapsed * 1000:.1f} ms/frame")
    return elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Export models to ONNX")
    parser.add_argument("--output-dir", default="models/onnx", help="Output directory")
    parser.add_argument(
        "--quantize",
        choices=["none", "int8"],
        default="int8",
        help="Quantization mode (default: int8)",
    )
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark after export")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seg_path = export_segformer(output_dir)
    depth_path = export_depth_anything(output_dir)

    seg_int8 = None
    depth_int8 = None
    if args.quantize == "int8":
        seg_int8 = quantize_int8(seg_path)
        depth_int8 = quantize_int8(depth_path)

    if args.benchmark:
        print("\nBenchmark (CPU):")
        benchmark(seg_path, input_shape=(1, 3, 512, 512))
        benchmark(depth_path, input_shape=(1, 3, 518, 518))
        if seg_int8:
            benchmark(seg_int8, input_shape=(1, 3, 512, 512))
        if depth_int8:
            benchmark(depth_int8, input_shape=(1, 3, 518, 518))

    print("\nDone. Models saved to:", output_dir)


if __name__ == "__main__":
    main()
