"""The benchmark sweep: precisions x context lengths, with telemetry and OOM tolerance.

The sweep loads each precision once and reuses it across context lengths, because model
construction dominates wall clock and re-loading per cell would turn a ten minute
experiment into an hour for no gain in fidelity.

Every cell ends up in the results with a status. ``ok`` means it ran, ``oom`` means the
card could not fit it, ``unsupported`` means the configuration cannot exist on this
machine and why, ``skipped`` means an earlier failure made it moot. Nothing is dropped
silently and nothing is estimated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch

from src.benchmarks.latency import measure_generation, measure_sync_overhead, warmup
from src.benchmarks.metrics import BenchmarkCell
from src.benchmarks.prompts import build_batch, max_supported_context
from src.models.loader import LoadedModel, load_model
from src.models.oom import oom_guard, release_memory
from src.models.registry import supported_precisions
from src.monitoring.gpu import GpuSampler
from src.monitoring.memory import allocator_snapshot, reset_peak_stats
from src.utils.config import LabConfig
from src.utils.io import write_csv
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SweepResult:
    """Everything one sweep produced."""

    cells: list[BenchmarkCell] = field(default_factory=list)
    capability_reports: list[dict[str, Any]] = field(default_factory=list)
    models: dict[str, dict[str, Any]] = field(default_factory=dict)
    validation: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        return {
            "capability_reports": self.capability_reports,
            "models": self.models,
            "validation": self.validation,
            "cells": [c.to_dict() for c in self.cells],
        }

    def rows(self) -> list[dict[str, Any]]:
        """Return one flat CSV row per cell."""
        return [c.to_row() for c in self.cells]

    @property
    def n_ok(self) -> int:
        """How many cells produced measurements."""
        return sum(1 for c in self.cells if c.status == "ok")


def _telemetry_path(run_dir: Path, precision: str, context_length: int) -> Path:
    """Where one cell's GPU time series is written."""
    return run_dir / "telemetry" / f"{precision}_ctx{context_length}.csv"


def _run_cell(
    loaded: LoadedModel,
    config: LabConfig,
    context_length: int,
    run_dir: Path,
    device_index: int,
) -> BenchmarkCell:
    """Measure one (precision, context length) cell across its repeats."""
    bench = config.benchmark
    cell = BenchmarkCell(
        precision=loaded.precision,
        context_length=context_length,
        max_new_tokens=bench.max_new_tokens,
        batch_size=bench.batch_size,
        load=loaded.metrics.to_dict(),
    )

    usable, clamp_reason = max_supported_context(loaded.model.config, context_length)
    if clamp_reason is not None:
        cell.status = "unsupported"
        cell.reason = clamp_reason
        logger.warning("ctx=%d skipped: %s", context_length, clamp_reason)
        return cell

    label = f"{loaded.precision}@ctx{context_length}"
    reset_peak_stats(device_index)

    with oom_guard(label, device=device_index, reraise_non_oom=True) as outcome:
        inputs = build_batch(loaded.tokenizer, usable, bench.batch_size, loaded.device)

        warmup(
            loaded.model,
            loaded.tokenizer,
            inputs,
            bench.max_new_tokens,
            device=loaded.device,
            iterations=bench.warmup,
        )
        # The warm-up's allocations are part of steady state, not of the measurement.
        reset_peak_stats(device_index)

        sampler = GpuSampler(
            device_index=device_index,
            interval_s=config.monitoring.sample_interval_s,
            enabled=config.monitoring.enabled,
        )
        with sampler:
            for repeat in range(bench.repeats):
                sampler.mark(f"repeat{repeat}:start")
                timing, _ = measure_generation(
                    loaded.model,
                    loaded.tokenizer,
                    inputs,
                    bench.max_new_tokens,
                    device=loaded.device,
                )
                sampler.mark(f"repeat{repeat}:end")
                cell.repeats.append(timing.to_dict())
                logger.info(
                    "%s repeat %d/%d | ttft %.1f ms | decode %.1f tok/s | prefill %.1f tok/s",
                    label,
                    repeat + 1,
                    bench.repeats,
                    timing.ttft_s * 1000,
                    timing.decode_tokens_per_s or float("nan"),
                    timing.prefill_tokens_per_s or float("nan"),
                )

        cell.telemetry = sampler.summary()
        cell.telemetry["allocator_peak"] = allocator_snapshot(device_index)
        rows = sampler.to_rows()
        if rows:
            path = _telemetry_path(run_dir, loaded.precision, context_length)
            write_csv(path, rows)
            cell.telemetry["samples_csv"] = str(path.relative_to(run_dir))
            marker_path = path.with_name(path.stem + "_markers.csv")
            write_csv(marker_path, sampler.marker_rows())
            cell.telemetry["markers_csv"] = str(marker_path.relative_to(run_dir))

    if not outcome.ok:
        cell.status = outcome.status
        cell.reason = outcome.error_message
        cell.failure = outcome.to_dict()

    return cell


def run_sweep(
    config: LabConfig,
    run_dir: Path,
    device_index: int = 0,
    validate_sync_overhead: bool = True,
    checkpoint: Callable[[SweepResult], None] | None = None,
) -> SweepResult:
    """Execute the full precision x context-length grid.

    Args:
        config: Validated experiment configuration.
        run_dir: Directory this run writes into.
        device_index: CUDA device to use.
        validate_sync_overhead: Run the per-token-synchronization control measurement
            once, on the smallest context of the first working precision.
        checkpoint: Called with the partial result after *every cell*. A long sweep
            must not be able to lose completed measurements to a crash - or to an
            operator's Ctrl-C - in a later cell. Per-cell rather than per-precision
            matters more than it sounds: an fp32 cell that exceeds VRAM can take twenty
            minutes on a platform that pages instead of failing, and the cells already
            measured should be on disk before it starts.

    Returns:
        A :class:`SweepResult` containing every cell, including failures.
    """
    bench = config.benchmark
    runnable, reports = supported_precisions(bench.precisions, device_index)
    result = SweepResult(capability_reports=[r.to_dict() for r in reports])

    for report in reports:
        if report.supported:
            continue
        for context_length in bench.context_lengths:
            result.cells.append(
                BenchmarkCell(
                    precision=report.precision,
                    context_length=context_length,
                    max_new_tokens=bench.max_new_tokens,
                    batch_size=bench.batch_size,
                    status="unsupported",
                    reason=report.reason,
                )
            )

    for precision in runnable:
        loaded: LoadedModel | None = None
        # Tolerant on purpose, unlike the per-cell guard below. Anything that stops one
        # precision from loading - OOM, an unsupported dtype, a bitsandbytes problem -
        # is a fact about that precision on this machine, and it must not discard the
        # measurements the earlier precisions already produced.
        with oom_guard(
            f"load:{precision}", device=device_index, reraise_non_oom=False
        ) as load_outcome:
            loaded = load_model(
                config.model,
                precision,
                device_index=device_index,
                sample_interval_s=config.monitoring.sample_interval_s,
                monitor=config.monitoring.enabled,
            )

        if not load_outcome.ok or loaded is None:
            logger.error("Could not load %s at %s; marking its cells", config.model.id, precision)
            for context_length in bench.context_lengths:
                result.cells.append(
                    BenchmarkCell(
                        precision=precision,
                        context_length=context_length,
                        max_new_tokens=bench.max_new_tokens,
                        batch_size=bench.batch_size,
                        status=load_outcome.status,
                        reason=load_outcome.error_message,
                        failure=load_outcome.to_dict(),
                    )
                )
            if checkpoint is not None:
                checkpoint(result)
            continue

        result.models[precision] = loaded.describe()

        for context_length in bench.context_lengths:
            result.cells.append(
                _run_cell(loaded, config, context_length, run_dir, device_index)
            )
            if checkpoint is not None:
                checkpoint(result)

        if validate_sync_overhead and "sync_overhead" not in result.validation:
            smallest = min(bench.context_lengths)
            usable, reason = max_supported_context(loaded.model.config, smallest)
            if reason is None:
                logger.info("Measuring per-token synchronization overhead at ctx=%d", usable)
                with oom_guard("sync-overhead-control", device=device_index) as ctrl:
                    inputs = build_batch(
                        loaded.tokenizer, usable, bench.batch_size, loaded.device
                    )
                    result.validation["sync_overhead"] = measure_sync_overhead(
                        loaded.model,
                        loaded.tokenizer,
                        inputs,
                        bench.max_new_tokens,
                        device=loaded.device,
                    )
                    result.validation["sync_overhead"]["precision"] = precision
                if not ctrl.ok:
                    result.validation["sync_overhead"] = {"status": ctrl.status}

        model_ref = loaded.model
        tokenizer_ref = loaded.tokenizer
        loaded.model = None
        loaded.tokenizer = None
        del model_ref, tokenizer_ref, loaded
        freed = release_memory(device_index)
        logger.info("Unloaded %s; device now holds %.0f MiB", precision, freed["used_mib"] or -1)

    return result


def torch_device(device_index: int = 0) -> str:
    """Return the device string the sweep will run on."""
    return f"cuda:{device_index}" if torch.cuda.is_available() else "cpu"
