"""Interleaved, paired benchmarking across attention backends, batch sizes and contexts.

    python -m src.benchmarks.interleaved --config configs/decode_ab.yaml

The sweep harness measures one configuration to completion before starting the next.
That is fine for a single configuration, but a *comparison* made that way is confounded
by anything that changes between the two runs. On this machine that is not
hypothetical: with other desktop applications active, the same bf16 cell has measured
anywhere from 35 to 45 tok/s at different times of day.

Here the model is loaded once and every round visits every (batch, context) cell,
switching backend in place between measurements and alternating the order each round
(A B, then B A). Each round therefore yields a *paired* comparison taken seconds apart,
and the reported effect is the median of the per-round ratios - not a ratio of two
medians measured minutes apart.

It also records whether the backends produce the same tokens. Two attention kernels
that differ only in floating-point summation order are expected to agree for a while and
then diverge; how soon is worth knowing before calling one a drop-in replacement.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import statistics
import sys
from collections.abc import Sequence
from typing import Any

import torch

from src.benchmarks.latency import measure_generation, warmup
from src.benchmarks.prompts import build_batch, build_fixed_length_ids
from src.evaluation.quality import distribution_divergence
from src.models.attention import set_attn_implementation
from src.models.loader import load_model
from src.models.oom import oom_guard, release_memory
from src.monitoring.gpu import GpuSampler
from src.monitoring.memory import paging_suspected
from src.utils.config import ConfigError, load_config
from src.utils.env import collect_metadata
from src.utils.io import create_run_dir, write_csv, write_json
from src.utils.logging import get_logger, setup_logging
from src.utils.seed import set_seed

logger = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.benchmarks.interleaved",
        description="Paired, order-balanced benchmark of attention backends.",
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--verbose", action="store_true")
    return parser


def round_order(backends: Sequence[str], round_index: int) -> list[str]:
    """Alternate the backend order every round, so neither arm always goes first."""
    order = list(backends)
    return order if round_index % 2 == 0 else order[::-1]


def first_divergence(a: torch.Tensor, b: torch.Tensor) -> int | None:
    """Index of the first differing generated token in row 0, or None if identical."""
    diff = (a[0] != b[0]).nonzero()
    return int(diff[0]) if diff.numel() else None


@torch.inference_mode()
def decode_fidelity(
    model: Any,
    tokenizer: Any,
    backends: Sequence[str],
    context: int,
    steps: int,
    device: Any,
) -> dict[str, torch.Tensor]:
    """Teacher-forced next-token log-probabilities over ``steps`` decode steps.

    Every backend sees the same prompt and is then fed the same continuation one token
    at a time through the KV cache, so any difference in its output distributions is
    the decode kernel's alone - not a consequence of the two having generated different
    text, which is all an exact-match comparison can measure.
    """
    ids = build_fixed_length_ids(tokenizer, context + steps).reshape(1, -1).to(device)
    prompt, forced = ids[:, :context], ids[:, context:]
    out: dict[str, torch.Tensor] = {}
    for backend in backends:
        set_attn_implementation(model, backend)
        step = model(input_ids=prompt, use_cache=True, logits_to_keep=1)
        past = step.past_key_values
        rows = [torch.log_softmax(step.logits[:, -1].float(), dim=-1)]
        for t in range(steps - 1):
            step = model(input_ids=forced[:, t : t + 1], past_key_values=past, use_cache=True)
            past = step.past_key_values
            rows.append(torch.log_softmax(step.logits[:, -1].float(), dim=-1))
        out[backend] = torch.cat(rows).cpu()
        del past, step
    return out


def summarise(rows: Sequence[dict[str, Any]], backends: Sequence[str]) -> list[dict[str, Any]]:
    """Per (batch, context): each arm's median, and the paired ratio against arm 0."""
    cells: dict[tuple[int, int], dict[str, dict[int, dict[str, Any]]]] = {}
    for row in rows:
        if row["status"] != "ok":
            continue
        key = (row["batch_size"], row["context_length"])
        cells.setdefault(key, {}).setdefault(row["backend"], {})[row["round"]] = row

    out = []
    for (batch, ctx), arms in sorted(cells.items()):
        summary: dict[str, Any] = {"batch_size": batch, "context_length": ctx}
        for backend in backends:
            measured = arms.get(backend, {})
            decode = [r["decode_tok_s"] for r in measured.values()]
            if not decode:
                continue
            summary[f"{backend}_decode_tok_s_median"] = round(statistics.median(decode), 3)
            summary[f"{backend}_decode_tok_s_min"] = round(min(decode), 3)
            summary[f"{backend}_decode_tok_s_max"] = round(max(decode), 3)
            summary[f"{backend}_throughput_tok_s_median"] = round(statistics.median(decode) * batch, 2)
            summary[f"{backend}_ttft_s_median"] = round(
                statistics.median(r["ttft_s"] for r in measured.values()), 5
            )
            summary[f"{backend}_peak_allocated_mib_max"] = max(
                r["peak_allocated_mib"] for r in measured.values()
            )
            summary[f"{backend}_paging_suspected"] = any(r["paging_suspected"] for r in measured.values())
        base = backends[0]
        for other in backends[1:]:
            ratios = [
                arms[other][k]["decode_tok_s"] / arms[base][k]["decode_tok_s"]
                for k in arms.get(base, {})
                if k in arms.get(other, {})
            ]
            if ratios:
                summary[f"{other}_vs_{base}_ratio_median"] = round(statistics.median(ratios), 4)
                summary[f"{other}_vs_{base}_ratio_min"] = round(min(ratios), 4)
                summary[f"{other}_vs_{base}_ratio_max"] = round(max(ratios), 4)
                summary[f"{other}_vs_{base}_n_pairs"] = len(ratios)
        out.append(summary)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    cmp = config.comparison
    bench = config.benchmark
    if len(bench.precisions) != 1:
        print("config error: an interleaved comparison uses exactly one precision", file=sys.stderr)
        return 2
    precision = bench.precisions[0]

    run_dir = create_run_dir(args.output_root or config.experiment.output_root, config.experiment.name)
    setup_logging(level=logging.DEBUG if args.verbose else logging.INFO, log_file=run_dir / "run.log")
    determinism = set_seed(config.experiment.seed, config.experiment.deterministic)

    meta = collect_metadata(
        config.experiment.name,
        extra={"config": config.to_dict(), "determinism": determinism.to_dict(), "run_dir": str(run_dir)},
        device_index=args.device,
    )
    write_json(run_dir / "meta.json", meta)
    if not meta["gpu"].get("available"):
        logger.error("No CUDA device available")
        return 1
    total_mib = torch.cuda.get_device_properties(args.device).total_memory / 2**20

    loaded = load_model(config.model, precision, device_index=args.device, monitor=False)
    model, tokenizer, device = loaded.model, loaded.tokenizer, loaded.device

    rows: list[dict[str, Any]] = []
    agreement: list[dict[str, Any]] = []
    grid = [(b, c) for b in cmp.batch_sizes for c in bench.context_lengths]

    # Warm every (backend, batch, context) once so kernel selection is not measured.
    for backend in cmp.backends:
        set_attn_implementation(model, backend)
        for batch, ctx in grid:
            with oom_guard(f"warmup:{backend}:b{batch}:c{ctx}", device=args.device) as outcome:
                inputs = build_batch(tokenizer, ctx, batch, device)
                warmup(model, tokenizer, inputs, bench.max_new_tokens, device=device, iterations=bench.warmup)
            if not outcome.ok:
                logger.warning("Warm-up failed for %s b%d c%d: %s", backend, batch, ctx, outcome.error_message)
            release_memory(args.device)

    for round_index in range(cmp.rounds):
        for batch, ctx in grid:
            sequences: dict[str, torch.Tensor] = {}
            for position, backend in enumerate(round_order(cmp.backends, round_index)):
                set_attn_implementation(model, backend)
                label = f"r{round_index} {backend} b{batch} c{ctx}"
                row: dict[str, Any] = {
                    "round": round_index,
                    "position": position,
                    "backend": backend,
                    "batch_size": batch,
                    "context_length": ctx,
                }
                torch.cuda.reset_peak_memory_stats(args.device)
                sampler = GpuSampler(device_index=args.device, interval_s=config.monitoring.sample_interval_s,
                                     enabled=config.monitoring.enabled)
                with oom_guard(label, device=args.device) as outcome:
                    inputs = build_batch(tokenizer, ctx, batch, device)
                    with sampler:
                        timing, seqs = measure_generation(
                            model, tokenizer, inputs, bench.max_new_tokens, device=device
                        )
                    sequences[backend] = seqs[:, ctx:].cpu()
                if outcome.ok:
                    telemetry = sampler.summary()
                    peak_device = telemetry.get("peak_memory_used_mib") or 0.0
                    row.update(
                        {
                            "status": "ok",
                            "decode_tok_s": round(timing.decode_tokens_per_s, 4),
                            "throughput_tok_s": round(timing.decode_tokens_per_s * batch, 3),
                            "ttft_s": round(timing.ttft_s, 5),
                            "prefill_s": round(timing.prefill_latency_s or float("nan"), 5),
                            "peak_allocated_mib": round(torch.cuda.max_memory_allocated(args.device) / 2**20, 1),
                            "peak_device_used_mib": peak_device,
                            "paging_suspected": paging_suspected(peak_device, total_mib, cmp.paging_threshold),
                            "mean_gpu_util_pct": telemetry.get("mean_gpu_util_pct"),
                            "tokens_sha1": hashlib.sha1(sequences[backend].numpy().tobytes()).hexdigest()[:12],
                        }
                    )
                    logger.info(
                        "%-40s decode %6.2f tok/s  x%-3d = %8.1f tok/s  peak %7.0f MiB%s",
                        label, timing.decode_tokens_per_s, batch, timing.decode_tokens_per_s * batch,
                        row["peak_allocated_mib"], "  PAGING?" if row["paging_suspected"] else "",
                    )
                else:
                    row.update({"status": outcome.status, "reason": outcome.error_message})
                    logger.warning("%s failed: %s", label, outcome.error_message)
                rows.append(row)
                release_memory(args.device)

            if round_index == 0 and len(sequences) >= 2:
                base = cmp.backends[0]
                for other in cmp.backends[1:]:
                    if base in sequences and other in sequences:
                        a, b = sequences[base], sequences[other]
                        agreement.append(
                            {
                                "batch_size": batch,
                                "context_length": ctx,
                                "pair": f"{other}_vs_{base}",
                                "identical": bool(torch.equal(a, b)),
                                "token_agreement": round(float((a == b).float().mean()), 4),
                                "first_divergence_row0": first_divergence(a, b),
                                "tokens_compared": int(a.numel()),
                            }
                        )
        write_csv(run_dir / "measurements.csv", rows)

    fidelity: list[dict[str, Any]] = []
    reference = cmp.fidelity_reference or cmp.backends[0]
    compared = [b for b in cmp.backends if b != reference]
    if cmp.fidelity_steps and compared:
        for ctx in bench.context_lengths:
            with oom_guard(f"fidelity:c{ctx}", device=args.device) as outcome:
                logps = decode_fidelity(
                    model, tokenizer, [reference, *compared], ctx, cmp.fidelity_steps, device
                )
            if not outcome.ok:
                logger.warning("Fidelity at c%d failed: %s", ctx, outcome.error_message)
                continue
            base = reference
            for other in compared:
                fidelity.append(
                    {"context_length": ctx, "pair": f"{other}_vs_{base}", "steps": cmp.fidelity_steps,
                     **distribution_divergence(logps[base], logps[other])}
                )
                logger.info("Fidelity c%-6d %s: mean KL %.2e nats, top-1 %.3f", ctx, other,
                            fidelity[-1]["mean_kl_nats"], fidelity[-1]["top1_agreement"])
            del logps
            release_memory(args.device)

    summary = summarise(rows, cmp.backends)
    payload = {
        "meta": meta,
        "model": loaded.describe(),
        "design": {
            "rounds": cmp.rounds,
            "order": "alternating per round (ABBA) so drift affects both arms equally",
            "effect": "median of per-round paired ratios against the first backend",
        },
        "summary": summary,
        "token_agreement": agreement,
        "decode_fidelity": fidelity,
        "measurements": rows,
    }
    write_json(run_dir / "interleaved.json", payload)
    write_csv(run_dir / "summary.csv", summary)
    if agreement:
        write_csv(run_dir / "agreement.csv", agreement)
    if fidelity:
        write_csv(run_dir / "fidelity.csv", fidelity)

    print("\nInterleaved comparison")
    print("-" * 72)
    for s in summary:
        parts = [f"b{s['batch_size']:<3} c{s['context_length']:<6}"]
        for backend in cmp.backends:
            key = f"{backend}_decode_tok_s_median"
            if key in s:
                parts.append(f"{backend} {s[key]:7.2f}")
        for other in cmp.backends[1:]:
            key = f"{other}_vs_{cmp.backends[0]}_ratio_median"
            if key in s:
                parts.append(f"ratio {s[key]:.3f} [{s[key.replace('median', 'min')]:.3f}, {s[key.replace('median', 'max')]:.3f}]")
        print("  " + " | ".join(parts))
    print(f"  results: {run_dir}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
