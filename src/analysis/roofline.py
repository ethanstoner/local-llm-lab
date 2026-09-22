"""A roofline model of single-stream inference, checked against every measured run.

    python -m src.analysis.roofline

Decoding one token at batch size 1 must stream every weight and every cached key and
value through the memory system once, and does almost no arithmetic per byte, so its
speed limit is ``bytes moved / sustainable bandwidth``. Prefill does ``2 x parameters``
floating-point operations per token plus attention, and at useful lengths is limited by
matmul throughput. Both limits can be computed from the model's config and the
*measured* hardware ceilings (``src.benchmarks.ceilings``), with no free parameters.

The point is not to predict the measurements - it is to say how far each one is from
what the hardware allows, and to attribute the gap. Three decode byte counts are
modelled, from ideal to what transformers actually does:

``ideal``
    Weights plus the KV cache, each read once.
``grouped``
    Adds the cost of transformers' ``DynamicCache``, which appends to the cache with
    ``torch.cat`` and so reads and rewrites the whole cache every step.
``expanded``
    Adds ``repeat_kv``: the ``sdpa_no_gqa`` backend copies the cache ``groups`` times
    per layer and the attention kernel then reads the copy.

Everything here is arithmetic on result files; nothing touches a GPU.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.utils.io import create_run_dir, iter_run_dirs, read_json, repo_root, write_csv, write_json
from src.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)

MIB = 2**20

#: Which attention backends expand the KV heads on every decode step.
EXPANDING_BACKENDS = {"sdpa_no_gqa", "eager"}


@dataclass(frozen=True)
class Architecture:
    """The handful of config fields the byte and FLOP counts need."""

    n_layers: int
    hidden_size: int
    intermediate_size: int
    n_heads: int
    n_kv_heads: int
    vocab_size: int
    tie_word_embeddings: bool

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> Architecture:
        return cls(
            n_layers=int(config["num_hidden_layers"]),
            hidden_size=int(config["hidden_size"]),
            intermediate_size=int(config["intermediate_size"]),
            n_heads=int(config["num_attention_heads"]),
            n_kv_heads=int(config["num_key_value_heads"]),
            vocab_size=int(config["vocab_size"]),
            tie_word_embeddings=bool(config.get("tie_word_embeddings", False)),
        )

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.n_heads

    @property
    def groups(self) -> int:
        return self.n_heads // self.n_kv_heads

    def kv_bytes_per_token(self, bytes_per_element: float = 2.0) -> float:
        """Keys and values for one token across every layer."""
        return 2 * self.n_layers * self.n_kv_heads * self.head_dim * bytes_per_element

    def matmul_params(self) -> int:
        """Parameters that take part in a matmul for every token.

        The embedding table is a lookup, not a matmul, and the output head only runs on
        the final position (prefill logits are trimmed to it), so neither scales with
        the prompt. Norm weights are negligible.
        """
        h, i, kv = self.hidden_size, self.intermediate_size, self.n_kv_heads * self.head_dim
        attention = h * h + 2 * h * kv + h * h  # q, k, v, o
        mlp = 3 * h * i  # gate, up, down
        return self.n_layers * (attention + mlp)


def decode_bytes(
    arch: Architecture,
    weight_bytes: float,
    kv_tokens: float,
    model: str,
    kv_element_bytes: float = 2.0,
    batch_size: int = 1,
) -> float:
    """Bytes one decode step must move under a given traffic model.

    Args:
        arch: Model architecture.
        weight_bytes: Measured size of the weights actually resident on the device.
        kv_tokens: Tokens in each sequence's KV cache during this step.
        model: ``"ideal"``, ``"grouped"`` or ``"expanded"`` (see the module docstring).
        batch_size: Sequences decoded together. The weights are read once per step
            whatever the batch; every sequence's cache is read separately.
    """
    kv = arch.kv_bytes_per_token(kv_element_bytes) * kv_tokens * batch_size
    if model == "ideal":
        return weight_bytes + kv
    if model == "grouped":
        return weight_bytes + kv + 2 * kv
    if model == "expanded":
        return weight_bytes + kv + 2 * kv + 2 * arch.groups * kv
    raise ValueError(f"unknown traffic model {model!r}")


def mean_kv_tokens(context: int, new_tokens: int) -> float:
    """Average cache length over a generation: it grows by one per decoded token."""
    return context + (new_tokens - 1) / 2.0


def prefill_flops(arch: Architecture, tokens: int, causal: bool = True) -> float:
    """Floating-point operations to process a prompt of ``tokens`` tokens.

    ``2 x matmul params`` per token for the projections, plus attention's QK^T and PV:
    ``4 x T^2 x hidden`` per layer, halved when the kernel skips the causally masked
    half.
    """
    linear = 2.0 * arch.matmul_params() * tokens
    attention = 4.0 * arch.n_layers * tokens * tokens * arch.hidden_size
    if causal:
        attention /= 2.0
    return linear + attention


# -- reading result files ----------------------------------------------------------


def latest(results_root: Path, experiment: str) -> Path | None:
    runs = list(iter_run_dirs(results_root, experiment))
    return runs[-1] if runs else None


def load_ceilings(run_dir: Path) -> dict[str, float]:
    """Reduce a ceilings run to the two numbers the model uses."""
    payload = read_json(run_dir / "ceilings.json")
    gemv = payload["gemv"]
    decode_bw = min(gemv["mlp_up_3584x18944"]["median_rate"], gemv["mlp_down_18944x3584"]["median_rate"])
    return {
        "decode_bandwidth_gb_s": decode_bw,
        "stream_bandwidth_gb_s": payload["stream_read"]["median_rate"],
        "gemm_tflops": payload["gemm_bf16"]["median_rate"],
        "spec_bandwidth_gb_s": (payload.get("spec_sheet") or {}).get("bandwidth_gb_s"),
        "spec_tflops": (payload.get("spec_sheet") or {}).get("bf16_tflops"),
    }


def benchmark_cells(run_dir: Path) -> Iterable[dict[str, Any]]:
    """Yield every successful cell of a benchmark run with what the model needs."""
    payload = read_json(run_dir / "metrics.json")
    models = payload.get("models", {})
    for cell in payload.get("cells", []):
        if cell.get("status") != "ok":
            continue
        described = models.get(cell["precision"], {})
        yield {
            "run": run_dir.parent.name + "/" + run_dir.name,
            "precision": cell["precision"],
            "attn_implementation": described.get("attn_implementation"),
            "config": described.get("config"),
            "weights_mib": cell["load"]["weights_mib"],
            "context_length": cell["context_length"],
            "new_tokens": cell["max_new_tokens"],
            "decode_tok_s": cell["summary"]["decode_tokens_per_s_median"],
            "decode_tok_s_stdev": cell["summary"]["decode_tokens_per_s_stdev"],
            "prefill_tok_s": cell["summary"]["prefill_tokens_per_s_median"],
            "prefill_s": cell["summary"]["prefill_latency_s_median"],
        }


def analyse_cell(cell: Mapping[str, Any], ceilings: Mapping[str, float]) -> dict[str, Any]:
    """Compare one measured cell with its roofline limits.

    Decode rates are per sequence; with ``batch_size`` sequences a step produces that
    many tokens, so aggregate throughput and its ceilings are ``batch_size`` times the
    per-sequence figures. The compute ceiling - 2 FLOPs per matmul parameter per token
    at the measured GEMM rate - is reported alongside, because a large enough batch
    stops being bandwidth-bound.
    """
    arch = Architecture.from_config(cell["config"])
    batch = int(cell.get("batch_size", 1))
    weight_bytes = cell["weights_mib"] * MIB
    kv_tokens = mean_kv_tokens(cell["context_length"], cell["new_tokens"])
    bw = ceilings["decode_bandwidth_gb_s"] * 1e9

    backend = cell.get("attn_implementation")
    applicable = "expanded" if backend in EXPANDING_BACKENDS else "grouped"

    row: dict[str, Any] = {
        "run": cell["run"],
        "precision": cell["precision"],
        "attn_implementation": backend,
        "batch_size": batch,
        "context_length": cell["context_length"],
        "weights_gb": round(weight_bytes / 1e9, 3),
        "kv_cache_gb": round(arch.kv_bytes_per_token() * kv_tokens * batch / 1e9, 4),
        "measured_decode_tok_s": cell["decode_tok_s"],
        "measured_decode_tok_s_stdev": cell.get("decode_tok_s_stdev"),
        "measured_throughput_tok_s": round(cell["decode_tok_s"] * batch, 3),
        "measured_ms_per_step": round(1000.0 / cell["decode_tok_s"], 3),
        "traffic_model": applicable,
    }
    for model in ("ideal", "grouped", "expanded"):
        limit = bw / decode_bytes(arch, weight_bytes, kv_tokens, model, batch_size=batch)
        row[f"ceiling_{model}_tok_s"] = round(limit, 2)
    compute_ceiling = ceilings["gemm_tflops"] * 1e12 / (2.0 * arch.matmul_params())
    row["compute_ceiling_throughput_tok_s"] = round(compute_ceiling, 1)
    row["fraction_of_ideal"] = round(cell["decode_tok_s"] / row["ceiling_ideal_tok_s"], 4)
    row["fraction_of_applicable"] = round(cell["decode_tok_s"] / row[f"ceiling_{applicable}_tok_s"], 4)
    row["achieved_bandwidth_gb_s"] = round(
        decode_bytes(arch, weight_bytes, kv_tokens, applicable, batch_size=batch)
        * cell["decode_tok_s"] / 1e9,
        1,
    )

    prefill_s = cell.get("prefill_s")
    flops = prefill_flops(arch, cell["context_length"]) * batch
    row["prefill_tflop"] = round(flops / 1e12, 3)
    row["prefill_ceiling_s"] = round(flops / (ceilings["gemm_tflops"] * 1e12), 5)
    row["measured_prefill_s"] = prefill_s
    row["prefill_mfu"] = (
        round(flops / prefill_s / (ceilings["gemm_tflops"] * 1e12), 4) if prefill_s else None
    )
    row["attention_share_of_prefill_flops"] = round(
        (flops - 2.0 * arch.matmul_params() * cell["context_length"] * batch) / flops, 4
    )
    return row


def interleaved_cells(run_dir: Path) -> Iterable[dict[str, Any]]:
    """Yield one median cell per (backend, batch, context) of an interleaved run."""
    import statistics

    payload = read_json(run_dir / "interleaved.json")
    model = payload.get("model", {})
    groups: dict[tuple[str, int, int], list[dict[str, Any]]] = {}
    for row in payload.get("measurements", []):
        if row.get("status") != "ok" or row.get("paging_suspected"):
            continue
        groups.setdefault((row["backend"], row["batch_size"], row["context_length"]), []).append(row)
    new_tokens = payload["meta"]["config"]["benchmark"]["max_new_tokens"]
    for (backend, batch, ctx), rows in sorted(groups.items()):
        decode = [r["decode_tok_s"] for r in rows]
        prefill = [r["prefill_s"] for r in rows if r.get("prefill_s") == r.get("prefill_s")]
        yield {
            "run": run_dir.parent.name + "/" + run_dir.name,
            "precision": model.get("precision"),
            "attn_implementation": backend,
            "config": model.get("config"),
            "weights_mib": model.get("load", {}).get("weights_mib"),
            "batch_size": batch,
            "context_length": ctx,
            "new_tokens": new_tokens,
            "decode_tok_s": statistics.median(decode),
            "decode_tok_s_stdev": round(statistics.pstdev(decode), 4),
            "prefill_s": statistics.median(prefill) if prefill else None,
        }


DEFAULT_RUNS = (
    "phase1_context_sweep",
    "phase2_precision_sweep",
    "phase7_decode_ab",
    "phase8_batch_sweep",
)


def collect(results_root: Path, experiments: Sequence[str] = DEFAULT_RUNS) -> list[Path]:
    """Every run of the listed benchmark experiments, oldest first."""
    runs: list[Path] = []
    for name in experiments:
        runs.extend(iter_run_dirs(results_root, name))
    return runs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.analysis.roofline",
        description="Compare every benchmark run with its measured-ceiling roofline.",
    )
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--ceilings-run", default=None, help="Default: latest hardware_ceilings run")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = repo_root() / args.results_root
    ceilings_dir = Path(args.ceilings_run) if args.ceilings_run else latest(root, "hardware_ceilings")
    if ceilings_dir is None:
        print("error: no hardware_ceilings run; run python -m src.benchmarks.ceilings", file=sys.stderr)
        return 2

    run_dir = create_run_dir(args.results_root, "roofline")
    setup_logging(level=logging.DEBUG if args.verbose else logging.INFO, log_file=run_dir / "run.log")
    ceilings = load_ceilings(ceilings_dir)

    rows: list[dict[str, Any]] = []
    sources = collect(root)
    for source in sources:
        reader = interleaved_cells if (source / "interleaved.json").is_file() else benchmark_cells
        for cell in reader(source):
            if not cell.get("config"):
                logger.warning("No model config in %s; skipping", source)
                continue
            rows.append(analyse_cell(cell, ceilings))

    payload = {
        "ceilings_run": str(ceilings_dir.relative_to(repo_root())).replace("\\", "/"),
        "ceilings": ceilings,
        "sources": [str(s.relative_to(repo_root())).replace("\\", "/") for s in sources],
        "rows": rows,
    }
    write_json(run_dir / "roofline.json", payload)
    write_csv(run_dir / "roofline.csv", rows)

    print("\nRoofline vs measurement (decode)")
    print("-" * 104)
    print(f"  decode bandwidth ceiling {ceilings['decode_bandwidth_gb_s']:.0f} GB/s | GEMM ceiling {ceilings['gemm_tflops']:.1f} TFLOP/s")
    print(f"  {'run':<40}{'prec':<6}{'backend':<21}{'B':>4}{'ctx':>6}{'meas':>7}{'ideal':>7}{'model':>7}{'%model':>8}{'MFU':>7}")
    for r in rows:
        mfu = r["prefill_mfu"]
        print(
            f"  {r['run'][:39]:<40}{r['precision']:<6}{str(r['attn_implementation'])[:20]:<21}{r['batch_size']:>4}"
            f"{r['context_length']:>6}{r['measured_decode_tok_s']:>7.1f}{r['ceiling_ideal_tok_s']:>7.1f}"
            f"{r['ceiling_' + r['traffic_model'] + '_tok_s']:>7.1f}{r['fraction_of_applicable']:>8.2f}"
            f"{'' if mfu is None else f'{mfu:.2f}':>7}"
        )
    print(f"\n  results: {run_dir}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
