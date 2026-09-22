"""CLI entry point for the Phase 2 quality comparison.

    python -m src.evaluation.run --config configs/precision_sweep.yaml

Loads the reference precision, reduces it to a small set of comparison artifacts,
unloads it, then loads each candidate precision in turn and scores it against those
artifacts. The two models are never resident on the card at the same time, which is what
makes this runnable on a single 24 GB device.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Sequence

from src.evaluation.quality import (
    QualityComparison,
    ReferenceArtifacts,
    continuation_agreement,
    distribution_divergence,
    final_position_logprobs,
    greedy_continuations,
    teacher_forced_pass,
    trace_comparison,
)
from src.models.loader import load_model
from src.models.oom import oom_guard, release_memory
from src.models.registry import check_precision_support
from src.utils.config import ConfigError, LabConfig, load_config
from src.utils.datasets import load_perplexity_text, load_prompt_sets
from src.utils.env import collect_metadata
from src.utils.io import create_run_dir, write_csv, write_json
from src.utils.logging import get_logger, setup_logging
from src.utils.seed import set_seed

logger = get_logger(__name__)

#: Context window for the teacher-forced pass. Comfortably within every model this
#: project targets and large enough that scored positions have real left context.
TF_WINDOW = 1024


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser."""
    parser = argparse.ArgumentParser(
        prog="python -m src.evaluation.run",
        description="Compare quantized models against a full-precision reference.",
    )
    parser.add_argument("--config", required=True, help="Path to a YAML experiment config")
    parser.add_argument("--device", type=int, default=0, help="CUDA device index")
    parser.add_argument("--output-root", default=None, help="Override experiment.output_root")
    parser.add_argument("--verbose", action="store_true", help="Log at DEBUG level")
    return parser


def _evaluation_prompts(config: LabConfig) -> list[str]:
    """Return the fixed prompt set used for continuation and distribution comparisons.

    Harmless instructions are used: the comparison is about numerical fidelity between
    precisions, so the prompts should elicit ordinary, substantive answers.
    """
    sets = load_prompt_sets(
        harmful_source="bundled",
        harmless_source="bundled",
        n_per_class=config.evaluation.n_prompts,
        seed=config.experiment.seed,
    )
    return sets.harmless


def _build_reference(
    config: LabConfig,
    prompts: Sequence[str],
    device_index: int,
) -> tuple[ReferenceArtifacts | None, dict[str, Any]]:
    """Load the reference precision and reduce it to comparison artifacts."""
    precision = config.evaluation.reference_precision
    report = check_precision_support(precision, device_index)
    if not report.supported:
        return None, {"status": "unsupported", "reason": report.reason}

    loaded = None
    with oom_guard(f"reference-load:{precision}", device=device_index) as outcome:
        loaded = load_model(config.model, precision, device_index=device_index, monitor=False)

    if not outcome.ok or loaded is None:
        return None, {"status": outcome.status, "reason": outcome.error_message}

    logger.info("Reference %s loaded; computing artifacts", precision)
    trace = teacher_forced_pass(
        loaded.model,
        loaded.tokenizer,
        load_perplexity_text(),
        max_tokens=config.evaluation.perplexity_tokens,
        window=TF_WINDOW,
        stride=config.evaluation.perplexity_stride,
        device=loaded.device,
    )
    final = final_position_logprobs(
        loaded.model, loaded.tokenizer, prompts, device=loaded.device
    )
    continuations = greedy_continuations(
        loaded.model,
        loaded.tokenizer,
        prompts,
        max_new_tokens=config.evaluation.max_new_tokens,
        device=loaded.device,
    )

    artifacts = ReferenceArtifacts(
        precision=precision,
        trace=trace,
        final_logprobs=final,
        continuations=continuations,
        prompts=list(prompts),
    )
    info = {"status": "ok", **artifacts.summary(), "load": loaded.metrics.to_dict()}

    loaded.model = None
    loaded.tokenizer = None
    del loaded
    release_memory(device_index)
    return artifacts, info


def _compare_precision(
    config: LabConfig,
    precision: str,
    reference: ReferenceArtifacts,
    device_index: int,
) -> QualityComparison:
    """Load one candidate precision and score it against the reference artifacts."""
    report = check_precision_support(precision, device_index)
    if not report.supported:
        return QualityComparison(precision=precision, status="unsupported", reason=report.reason)

    comparison = QualityComparison(precision=precision)
    loaded = None
    with oom_guard(f"candidate:{precision}", device=device_index) as outcome:
        loaded = load_model(config.model, precision, device_index=device_index, monitor=False)
        comparison.load = loaded.metrics.to_dict()

        trace = teacher_forced_pass(
            loaded.model,
            loaded.tokenizer,
            load_perplexity_text(),
            max_tokens=config.evaluation.perplexity_tokens,
            window=TF_WINDOW,
            stride=config.evaluation.perplexity_stride,
            device=loaded.device,
        )
        comparison.teacher_forced = trace_comparison(reference.trace, trace)

        final = final_position_logprobs(
            loaded.model, loaded.tokenizer, reference.prompts, device=loaded.device
        )
        comparison.distribution = distribution_divergence(reference.final_logprobs, final)

        continuations = greedy_continuations(
            loaded.model,
            loaded.tokenizer,
            reference.prompts,
            max_new_tokens=config.evaluation.max_new_tokens,
            device=loaded.device,
        )
        comparison.continuation = continuation_agreement(reference.continuations, continuations)

    if not outcome.ok:
        comparison.status = outcome.status
        comparison.reason = outcome.error_message

    if loaded is not None:
        loaded.model = None
        loaded.tokenizer = None
        del loaded
    release_memory(device_index)
    return comparison


def _print_summary(rows: Sequence[dict[str, Any]]) -> None:
    """Print a compact quality table."""
    header = (
        f"{'precision':<10}{'status':>12}{'top1 agree':>12}{'ppl ratio':>11}"
        f"{'mean KL':>11}{'exact cont':>12}"
    )
    print("\n" + header)
    print("-" * len(header))
    for row in rows:
        def fmt(key: str, spec: str) -> str:
            value = row.get(key)
            return format(value, spec) if isinstance(value, (int, float)) else "-"

        print(
            f"{row['precision']:<10}"
            f"{row['status']:>12}"
            f"{fmt('tf_top1_agreement', '.4f'):>12}"
            f"{fmt('tf_perplexity_ratio', '.4f'):>11}"
            f"{fmt('dist_mean_kl_nats', '.5f'):>11}"
            f"{fmt('cont_exact_match_rate', '.3f'):>12}"
        )
    print()


def main(argv: Sequence[str] | None = None) -> int:
    """Run the quality comparison described by a config file."""
    args = build_parser().parse_args(argv)

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    output_root = args.output_root or config.experiment.output_root
    experiment = f"{config.experiment.name}_quality"
    run_dir = create_run_dir(output_root, experiment)
    setup_logging(
        level=logging.DEBUG if args.verbose else logging.INFO,
        log_file=run_dir / "run.log",
    )

    determinism = set_seed(config.experiment.seed, config.experiment.deterministic)
    meta = collect_metadata(
        experiment,
        extra={
            "config": config.to_dict(),
            "determinism": determinism.to_dict(),
            "run_dir": str(run_dir),
        },
        device_index=args.device,
    )
    write_json(run_dir / "meta.json", meta)

    if not meta["gpu"].get("available"):
        logger.error("No CUDA device available")
        return 1

    prompts = _evaluation_prompts(config)
    logger.info("Evaluating with %d fixed prompts", len(prompts))

    reference, reference_info = _build_reference(config, prompts, args.device)
    if reference is None:
        logger.error("Reference precision unavailable: %s", reference_info.get("reason"))
        write_json(run_dir / "quality.json", {"meta": meta, "reference": reference_info})
        return 1

    comparisons = [
        _compare_precision(config, precision, reference, args.device)
        for precision in config.evaluation.compare_precisions
    ]

    payload = {
        "meta": meta,
        "reference": reference_info,
        "comparisons": [c.to_dict() for c in comparisons],
    }
    write_json(run_dir / "quality.json", payload)

    rows = [c.to_row() for c in comparisons]
    write_csv(run_dir / "quality.csv", rows)
    _print_summary(rows)

    logger.info("Quality comparison complete. Results in %s", run_dir)
    return 0 if any(c.status == "ok" for c in comparisons) else 1


if __name__ == "__main__":
    raise SystemExit(main())
