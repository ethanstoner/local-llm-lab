"""CLI entry point for activation capture and the refusal-direction analysis.

    python -m src.interpretability.run --config configs/refusal.yaml

Order of operations matters here. The behavioural check runs first, because a direction
that separates two prompt sets only says something about *refusal* if the model's
refusal behaviour on those sets actually differs. Activations are then captured for a
train and a test split of each class, the direction is fitted on train only, and every
reported statistic comes from the held-out test split.

Nothing in this pipeline modifies the model. Hooks are read-only and are removed when
the recorder's context exits.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Sequence

import torch

from src.interpretability.hooks import ActivationRecorder, save_activations
from src.interpretability.refusal import (
    direction_agreement,
    fit_all_layers,
    refusal_behaviour_check,
    summarise,
)
from src.interpretability.stats import layer_norm_profile, pca
from src.models.loader import load_model
from src.models.oom import oom_guard, release_memory
from src.models.registry import check_precision_support
from src.monitoring.gpu import GpuSampler
from src.utils.config import ConfigError, LabConfig, load_config
from src.utils.datasets import load_prompt_sets, split_prompt_sets
from src.utils.env import collect_metadata
from src.utils.io import create_run_dir, write_csv, write_json
from src.utils.logging import get_logger, setup_logging
from src.utils.seed import set_seed

logger = get_logger(__name__)

CAPTURE_BATCH_SIZE = 8


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser."""
    parser = argparse.ArgumentParser(
        prog="python -m src.interpretability.run",
        description=(
            "Capture residual-stream activations and measure the layer-wise refusal "
            "direction. Analysis only: no weights are modified."
        ),
    )
    parser.add_argument("--config", required=True, help="Path to a YAML experiment config")
    parser.add_argument("--device", type=int, default=0, help="CUDA device index")
    parser.add_argument("--output-root", default=None, help="Override experiment.output_root")
    parser.add_argument(
        "--skip-behaviour-check",
        action="store_true",
        help="Skip generation and measure the direction only",
    )
    parser.add_argument("--verbose", action="store_true", help="Log at DEBUG level")
    return parser


def _capture_split(
    recorder: ActivationRecorder,
    tokenizer: Any,
    prompts: Sequence[str],
    label: str,
    device: str,
) -> dict[int, torch.Tensor]:
    """Capture activations for one labelled prompt split."""
    logger.info("Capturing activations for %s (%d prompts)", label, len(prompts))
    result = recorder.capture_prompts(
        tokenizer,
        prompts,
        batch_size=CAPTURE_BATCH_SIZE,
        apply_chat_template=True,
        device=device,
    )
    return result.activations


def _pca_at_layer(
    harmful: torch.Tensor,
    harmless: torch.Tensor,
    layer: int,
) -> dict[str, Any]:
    """Run a 2-component PCA over both classes at one layer, for the scatter figure."""
    combined = torch.cat([harmful, harmless], dim=0)
    result = pca(combined, n_components=2)
    projected = result["projected"]
    n_harmful = harmful.shape[0]
    return {
        "layer": layer,
        "explained_variance_ratio": [round(v, 6) for v in result["explained_variance_ratio"]],
        "harmful": projected[:n_harmful].tolist(),
        "harmless": projected[n_harmful:].tolist(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run the interpretability pipeline described by a config file."""
    args = build_parser().parse_args(argv)

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    interp = config.interpretability
    output_root = args.output_root or config.experiment.output_root
    run_dir = create_run_dir(output_root, config.experiment.name)
    setup_logging(
        level=logging.DEBUG if args.verbose else logging.INFO,
        log_file=run_dir / "run.log",
    )

    determinism = set_seed(config.experiment.seed, config.experiment.deterministic)

    report = check_precision_support(interp.precision, args.device)
    if not report.supported:
        logger.error("Precision %s unsupported: %s", interp.precision, report.reason)
        return 1

    sets = load_prompt_sets(
        harmful_source=interp.harmful_source,
        harmless_source=interp.harmless_source,
        n_per_class=interp.n_prompts_per_class,
        seed=config.experiment.seed,
    )
    splits = split_prompt_sets(sets, interp.test_fraction, seed=config.experiment.seed)
    logger.info("Prompt splits: %s", splits.counts())

    meta = collect_metadata(
        config.experiment.name,
        extra={
            "config": config.to_dict(),
            "determinism": determinism.to_dict(),
            "prompt_sets": sets.summary(),
            "splits": splits.counts(),
            "run_dir": str(run_dir),
            "scope": (
                "Measurement only. No weight modification, directional ablation or "
                "activation steering is performed."
            ),
            "reference": (
                "Arditi et al. 2024, 'Refusal in Language Models Is Mediated by a "
                "Single Direction', arXiv:2406.11717"
            ),
        },
        device_index=args.device,
    )
    write_json(run_dir / "meta.json", meta)

    if not meta["gpu"].get("available"):
        logger.error("No CUDA device available")
        return 1

    payload: dict[str, Any] = {"meta": meta}
    loaded = None
    sampler = GpuSampler(
        device_index=args.device,
        interval_s=config.monitoring.sample_interval_s,
        enabled=config.monitoring.enabled,
    )
    sampler.start()

    with oom_guard("interpretability", device=args.device) as outcome:
        loaded = load_model(
            config.model, interp.precision, device_index=args.device, monitor=False
        )
        payload["model"] = loaded.describe()
        sampler.mark("model:loaded")

        if interp.run_refusal_behaviour_check and not args.skip_behaviour_check:
            n = interp.refusal_check_n_prompts
            payload["behaviour_check"] = refusal_behaviour_check(
                loaded.model,
                loaded.tokenizer,
                sets.harmful[:n],
                sets.harmless[:n],
                max_new_tokens=interp.refusal_check_max_new_tokens,
                device=loaded.device,
            )
            sampler.mark("behaviour_check:done")
        else:
            payload["behaviour_check"] = {"status": "skipped"}

        with ActivationRecorder(
            loaded.model, layers=interp.layers, aggregation=interp.aggregation
        ) as recorder:
            captured = {
                "harmful_train": _capture_split(
                    recorder, loaded.tokenizer, splits.harmful_train, "harmful_train", loaded.device
                ),
                "harmless_train": _capture_split(
                    recorder, loaded.tokenizer, splits.harmless_train, "harmless_train", loaded.device
                ),
                "harmful_test": _capture_split(
                    recorder, loaded.tokenizer, splits.harmful_test, "harmful_test", loaded.device
                ),
                "harmless_test": _capture_split(
                    recorder, loaded.tokenizer, splits.harmless_test, "harmless_test", loaded.device
                ),
            }
        sampler.mark("capture:done")

    sampler.stop()
    payload["telemetry"] = sampler.summary()

    if not outcome.ok:
        payload["status"] = outcome.status
        payload["failure"] = outcome.to_dict()
        write_json(run_dir / "refusal_analysis.json", payload)
        logger.error("Run failed: %s", outcome.error_message)
        return 1

    if interp.save_activations:
        activations_dir = run_dir / "activations"
        for name, tensors in captured.items():
            save_activations(
                activations_dir / f"{name}.safetensors",
                tensors,
                metadata={
                    "split": name,
                    "aggregation": interp.aggregation,
                    "model": config.model.id,
                    "precision": interp.precision,
                },
            )
        payload["activations_dir"] = "activations"
        logger.info("Saved activations to %s", activations_dir)

    results = fit_all_layers(
        captured["harmful_train"],
        captured["harmless_train"],
        captured["harmful_test"],
        captured["harmless_test"],
    )

    payload["summary"] = summarise(results)
    payload["direction_agreement"] = direction_agreement(results)
    payload["layers"] = [r.to_dict() for r in results]
    payload["activation_norm_profile"] = {
        "harmful_test": layer_norm_profile(captured["harmful_test"]),
        "harmless_test": layer_norm_profile(captured["harmless_test"]),
    }

    if results:
        best_layer = payload["summary"]["best_layer_by_cohens_d"]["layer"]
        payload["pca"] = _pca_at_layer(
            captured["harmful_test"][best_layer],
            captured["harmless_test"][best_layer],
            best_layer,
        )
        save_activations(
            run_dir / "directions.safetensors",
            {r.layer: r.direction.unsqueeze(0) for r in results},
            metadata={"kind": "difference_in_means_unit_directions", "model": config.model.id},
        )

    write_json(run_dir / "refusal_analysis.json", payload)
    write_csv(run_dir / "layer_separation.csv", [r.to_row() for r in results])

    if loaded is not None:
        loaded.model = None
        loaded.tokenizer = None
        del loaded
    release_memory(args.device)

    best = payload["summary"].get("best_layer_by_cohens_d", {})
    behaviour = payload["behaviour_check"]
    print("\nRefusal direction analysis")
    print("-" * 52)
    if isinstance(behaviour, dict) and "harmful" in behaviour:
        print(
            f"  behavioural refusal rate: harmful "
            f"{behaviour['harmful']['refusal_rate']:.2f} | harmless "
            f"{behaviour['harmless']['refusal_rate']:.2f}"
        )
    print(f"  layers analysed:          {payload['summary'].get('n_layers')}")
    print(f"  best layer (held-out d):  {best.get('layer')}")
    print(f"  Cohen's d:                {best.get('cohens_d')}")
    print(f"  AUROC:                    {best.get('auroc')}")
    print(f"  results:                  {run_dir}\n")

    logger.info("Analysis complete. Results in %s", run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
