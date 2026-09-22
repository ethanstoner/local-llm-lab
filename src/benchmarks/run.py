"""CLI entry point for the GPU benchmark sweep (Phase 1 and Phase 2).

    python -m src.benchmarks.run --config configs/qwen2.5-7b.yaml

Writes one self-contained run directory containing ``meta.json`` (hardware, software,
git state, full config), ``metrics.json`` (every cell, including failures),
``results.csv`` (flat rows for plotting), ``telemetry/`` (GPU time series) and
``run.log``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Sequence

from src.benchmarks.sweep import run_sweep
from src.utils.config import ConfigError, load_config
from src.utils.env import collect_metadata
from src.utils.io import create_run_dir, write_csv, write_json
from src.utils.logging import get_logger, setup_logging
from src.utils.seed import set_seed

logger = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser."""
    parser = argparse.ArgumentParser(
        prog="python -m src.benchmarks.run",
        description="Benchmark a causal LM across precisions and context lengths.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a YAML experiment config, e.g. configs/qwen2.5-7b.yaml",
    )
    parser.add_argument("--device", type=int, default=0, help="CUDA device index")
    parser.add_argument(
        "--output-root",
        default=None,
        help="Override experiment.output_root from the config",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip the per-token synchronization control measurement",
    )
    parser.add_argument("--verbose", action="store_true", help="Log at DEBUG level")
    return parser


def _print_summary(result_rows: Sequence[dict]) -> None:
    """Print a compact table of the sweep to stdout."""
    header = f"{'precision':<10}{'ctx':>7}{'status':>13}{'decode tok/s':>14}{'ttft ms':>10}{'peak MiB':>11}"
    print("\n" + header)
    print("-" * len(header))
    for row in result_rows:
        decode = row.get("decode_tokens_per_s_median")
        ttft = row.get("ttft_s_median")
        peak = row.get("gpu_peak_memory_used_mib")
        print(
            f"{row['precision']:<10}"
            f"{row['context_length']:>7}"
            f"{row['status']:>13}"
            f"{(f'{decode:.2f}' if decode is not None else '-'):>14}"
            f"{(f'{ttft * 1000:.1f}' if ttft is not None else '-'):>10}"
            f"{(f'{peak:.0f}' if peak is not None else '-'):>11}"
        )
    print()


def main(argv: Sequence[str] | None = None) -> int:
    """Run the benchmark sweep described by a config file.

    Returns:
        Process exit code: 0 if at least one cell produced measurements, 1 otherwise.
    """
    args = build_parser().parse_args(argv)

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    output_root = args.output_root or config.experiment.output_root
    run_dir = create_run_dir(output_root, config.experiment.name)
    setup_logging(
        level=logging.DEBUG if args.verbose else logging.INFO,
        log_file=run_dir / "run.log",
    )

    logger.info("Config: %s", config.source_path)
    determinism = set_seed(config.experiment.seed, config.experiment.deterministic)

    meta = collect_metadata(
        config.experiment.name,
        extra={
            "config": config.to_dict(),
            "determinism": determinism.to_dict(),
            "cli": {"device": args.device, "no_validate": args.no_validate},
            "run_dir": str(run_dir),
        },
        device_index=args.device,
    )
    write_json(run_dir / "meta.json", meta)

    if not meta["gpu"].get("available"):
        logger.error("No CUDA device available; this experiment requires a GPU")
        write_json(run_dir / "metrics.json", {"error": "no CUDA device available"})
        return 1

    logger.info(
        "GPU: %s | %.0f MiB total | %.0f MiB already used by other processes",
        meta["gpu"]["name"],
        meta["gpu"]["total_memory_mib"],
        meta["gpu"]["used_by_other_processes_mib"],
    )

    result = run_sweep(
        config,
        run_dir,
        device_index=args.device,
        validate_sync_overhead=not args.no_validate,
    )

    payload = {"meta": meta, **result.to_dict()}
    write_json(run_dir / "metrics.json", payload)

    rows = result.rows()
    write_csv(run_dir / "results.csv", rows)
    _print_summary(rows)

    logger.info(
        "Sweep complete: %d/%d cells measured. Results in %s",
        result.n_ok,
        len(result.cells),
        run_dir,
    )
    return 0 if result.n_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
