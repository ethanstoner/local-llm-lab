"""Measured hardware ceilings: what this card can actually do, not what the box says.

    python -m src.benchmarks.ceilings

A roofline model needs two numbers - sustainable memory bandwidth and sustainable
matmul throughput - and the spec-sheet values are the wrong ones to use. They are peak
figures that no real kernel reaches, so a model built on them would make every measured
result look inefficient. This measures both directly, with the same kinds of operation
that dominate inference:

* **Streaming read** - a reduction over a large bf16 tensor. Reads every byte once and
  writes almost nothing, which is the access pattern of decoding.
* **Device copy** - read and write, for reference.
* **GEMV** - one activation vector times a weight matrix the size of Qwen2.5-7B's MLP
  projections. This *is* the decode operation, so its bandwidth is the realistic ceiling
  for a single-stream decoder.
* **GEMM** - a large square bf16 matmul, the compute ceiling that prefill approaches.

Each is timed with CUDA events after a warm-up, repeated, and reported as a median.
"""

from __future__ import annotations

import argparse
import logging
import statistics
import sys
from typing import Any, Callable, Sequence

import torch

from src.utils.env import collect_metadata
from src.utils.io import create_run_dir, write_json
from src.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)

#: Published RTX 4090 figures, recorded next to the measurements for comparison.
#: Bandwidth: 21 Gbps GDDR6X on a 384-bit bus. Tensor: dense FP16/BF16 with FP32
#: accumulate (the GeForce rate; the FP16-accumulate rate is double).
SPEC_SHEETS: dict[str, dict[str, float]] = {
    "NVIDIA GeForce RTX 4090": {"bandwidth_gb_s": 1008.0, "bf16_tflops": 165.2},
}


def time_cuda(fn: Callable[[], Any], repeats: int = 20, warmup: int = 3) -> list[float]:
    """Time ``fn`` on the current CUDA stream with events, returning seconds per call."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) / 1000.0)
    return times


def _summary(times: Sequence[float], work: float, unit_scale: float) -> dict[str, float]:
    median = statistics.median(times)
    return {
        "median_s": round(median, 7),
        "best_s": round(min(times), 7),
        "median_rate": round(work / median / unit_scale, 2),
        "best_rate": round(work / min(times) / unit_scale, 2),
    }


@torch.inference_mode()
def measure(device: int = 0, read_gib: float = 2.0, repeats: int = 20) -> dict[str, Any]:
    """Run every ceiling benchmark on one device."""
    dev = torch.device(f"cuda:{device}")
    results: dict[str, Any] = {}

    n = int(read_gib * 2**30 / 2)
    big = torch.ones(n, dtype=torch.bfloat16, device=dev)
    nbytes = big.numel() * big.element_size()
    results["stream_read"] = {
        "bytes": nbytes,
        "unit": "GB/s",
        **_summary(time_cuda(lambda: big.sum(dtype=torch.float32), repeats), nbytes, 1e9),
    }

    dst = torch.empty_like(big)
    results["device_copy"] = {
        "bytes_moved": 2 * nbytes,
        "unit": "GB/s",
        **_summary(time_cuda(lambda: dst.copy_(big), repeats), 2 * nbytes, 1e9),
    }
    del dst, big
    torch.cuda.empty_cache()

    gemv: dict[str, Any] = {}
    for label, (k, m) in {
        "mlp_up_3584x18944": (3584, 18944),
        "mlp_down_18944x3584": (18944, 3584),
        "attn_q_3584x3584": (3584, 3584),
    }.items():
        weight = torch.randn(m, k, dtype=torch.bfloat16, device=dev)
        x = torch.randn(1, k, dtype=torch.bfloat16, device=dev)
        wbytes = weight.numel() * weight.element_size()
        gemv[label] = {
            "weight_bytes": wbytes,
            "unit": "GB/s",
            **_summary(time_cuda(lambda: torch.nn.functional.linear(x, weight), repeats * 5), wbytes, 1e9),
        }
        del weight, x
    results["gemv"] = gemv

    size = 8192
    a = torch.randn(size, size, dtype=torch.bfloat16, device=dev)
    b = torch.randn(size, size, dtype=torch.bfloat16, device=dev)
    flops = 2.0 * size**3
    results["gemm_bf16"] = {
        "shape": [size, size, size],
        "flops": flops,
        "unit": "TFLOP/s",
        **_summary(time_cuda(lambda: a @ b, repeats), flops, 1e12),
    }
    del a, b
    torch.cuda.empty_cache()

    name = torch.cuda.get_device_name(dev)
    spec = SPEC_SHEETS.get(name)
    results["spec_sheet"] = spec
    if spec:
        results["fraction_of_spec"] = {
            "stream_read": round(results["stream_read"]["median_rate"] / spec["bandwidth_gb_s"], 4),
            "gemv_mlp_up": round(
                results["gemv"]["mlp_up_3584x18944"]["median_rate"] / spec["bandwidth_gb_s"], 4
            ),
            "gemm_bf16": round(results["gemm_bf16"]["median_rate"] / spec["bf16_tflops"], 4),
        }
    results["device_name"] = name
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.benchmarks.ceilings",
        description="Measure sustainable memory bandwidth and matmul throughput.",
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output-root", default="results")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not torch.cuda.is_available():
        print("error: no CUDA device", file=sys.stderr)
        return 1
    run_dir = create_run_dir(args.output_root, "hardware_ceilings")
    setup_logging(level=logging.DEBUG if args.verbose else logging.INFO, log_file=run_dir / "run.log")

    meta = collect_metadata("hardware_ceilings", device_index=args.device)
    write_json(run_dir / "meta.json", meta)
    results = measure(args.device, repeats=args.repeats)
    write_json(run_dir / "ceilings.json", {"meta": meta, **results})

    print("\nHardware ceilings")
    print("-" * 52)
    print(f"  device:            {results['device_name']}")
    print(f"  streaming read:    {results['stream_read']['median_rate']:.0f} GB/s")
    print(f"  device copy:       {results['device_copy']['median_rate']:.0f} GB/s (read + write)")
    for label, row in results["gemv"].items():
        print(f"  GEMV {label:<20} {row['median_rate']:.0f} GB/s")
    print(f"  bf16 GEMM 8192^3:  {results['gemm_bf16']['median_rate']:.1f} TFLOP/s")
    if results.get("fraction_of_spec"):
        print(f"  fraction of spec:  {results['fraction_of_spec']}")
    print(f"  results:           {run_dir}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
