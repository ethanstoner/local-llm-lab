"""CLI for Phase 6: is the Phase 5 refusal direction causal?

    python -m src.interpretability.intervene --config configs/intervention.yaml

Phase 5 found a direction that separates harmful from harmless prompts. This asks
whether the model *uses* it, with the interventions from Arditi et al. (2024):

1. **Necessity.** Project the direction out of every residual-stream write and measure
   the refusal rate on held-out harmful prompts. Controls: random unit directions, and
   directions fitted at layers where Phase 5 found no signal.
2. **Which layer's direction.** Repeat the ablation with the direction fitted at each
   layer in ``source_layers`` - the causal counterpart of Phase 5's observational sweep.
3. **Sufficiency.** Add the direction at the target layer and measure the refusal rate
   on held-out harmless prompts. Control: a random vector of the same norm.
4. **Cost.** Perplexity, teacher-forced agreement and next-token KL with the direction
   ablated, against the intact model. An ablation that removes refusal by breaking the
   network is not evidence of anything.

Every intervention is an inference-time hook, removed as soon as its condition finishes.
No weights are modified or saved. Harmful-prompt completions are classified, scored for
fluency and discarded; only counts and summary statistics are written.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import torch

from src.evaluation.quality import (
    distribution_divergence,
    final_position_logprobs,
    teacher_forced_pass,
    trace_comparison,
)
from src.interpretability.causal import (
    ConditionResult,
    evaluate_condition,
    select_causal_layer,
)
from src.interpretability.hooks import load_activations
from src.interpretability.intervention import (
    ActivationAddition,
    DirectionalAblation,
    NoIntervention,
    random_unit_direction,
    residual_component,
    unit,
)
from src.interpretability.stats import cosine_similarity, wilson_interval
from src.models.loader import load_model
from src.models.oom import oom_guard, release_memory
from src.models.registry import check_precision_support
from src.monitoring.gpu import GpuSampler
from src.utils.config import ConfigError, LabConfig, ModelConfig, load_config
from src.utils.datasets import (
    load_bundled_prompts,
    load_perplexity_text,
    load_prompt_sets,
    split_prompt_sets,
)
from src.utils.env import collect_metadata
from src.utils.io import (
    create_run_dir,
    latest_run_dir,
    read_json,
    repo_root,
    resolve_under_repo,
    write_csv,
    write_json,
)
from src.utils.logging import get_logger, setup_logging
from src.utils.seed import set_seed

logger = get_logger(__name__)

#: Interpretability settings that determine the train/test split. If any differ from
#: the direction run, the "held-out" prompts here might be ones the direction was
#: fitted on, so the run refuses to start.
_SPLIT_KEYS = ("n_prompts_per_class", "test_fraction", "harmful_source", "harmless_source", "precision")

#: The prompt set the causal layer is chosen on. Every other set is a clean test of it.
SELECTION_SET = "jbb_heldout"

TF_WINDOW = 1024
TF_STRIDE = 512
TF_MAX_TOKENS = 4096


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.interpretability.intervene",
        description=(
            "Test whether the Phase 5 refusal direction is causal, with inference-time "
            "ablation and activation addition. No weights are modified."
        ),
    )
    parser.add_argument("--config", required=True, help="Path to a YAML experiment config")
    parser.add_argument("--device", type=int, default=0, help="CUDA device index")
    parser.add_argument("--output-root", default=None, help="Override experiment.output_root")
    parser.add_argument("--verbose", action="store_true", help="Log at DEBUG level")
    return parser


class DirectionSource:
    """The fitted directions, per-layer statistics and provenance of a Phase 5 run."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        path = run_dir / "directions.safetensors"
        if not path.is_file():
            raise FileNotFoundError(f"{path} not found - run Phase 5 first")
        self.path = path
        self.sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        self.directions = {layer: unit(t) for layer, t in load_activations(path).items()}
        self.analysis = read_json(run_dir / "refusal_analysis.json")
        self.meta = read_json(run_dir / "meta.json")
        self.layer_stats = {entry["layer"]: entry for entry in self.analysis.get("layers", [])}

    def raw_norm(self, layer: int) -> float:
        return float(self.layer_stats[layer]["raw_difference_norm"])

    def test_d(self, layer: int) -> float:
        return float(self.layer_stats[layer]["test"]["cohens_d"])

    def best_layer(self) -> int:
        return int(self.analysis["summary"]["best_layer_by_cohens_d"]["layer"])

    def describe(self) -> dict[str, Any]:
        try:
            rel = str(self.run_dir.relative_to(repo_root()))
        except ValueError:
            rel = str(self.run_dir)
        return {"run_dir": rel.replace("\\", "/"), "directions_sha256": self.sha256}


def resolve_direction_run(config: LabConfig, output_root: str) -> Path:
    spec = config.intervention.direction_run
    if spec == "latest":
        run = latest_run_dir(output_root, config.intervention.direction_experiment)
        if run is None:
            raise FileNotFoundError(
                f"no {config.intervention.direction_experiment} run under {output_root}"
            )
        return run
    return resolve_under_repo(spec)


def check_split_matches(config: LabConfig, source: DirectionSource) -> None:
    """Refuse to run if this config would rebuild a different split from Phase 5's."""
    recorded = source.meta.get("config", {})
    mismatches = []
    if recorded.get("experiment", {}).get("seed") != config.experiment.seed:
        mismatches.append(
            f"experiment.seed {config.experiment.seed} != {recorded.get('experiment', {}).get('seed')}"
        )
    theirs = recorded.get("interpretability", {})
    ours = config.interpretability
    for key in _SPLIT_KEYS:
        if theirs.get(key) != getattr(ours, key):
            mismatches.append(f"interpretability.{key} {getattr(ours, key)!r} != {theirs.get(key)!r}")
    if recorded.get("model", {}).get("id") != config.model.id:
        mismatches.append(f"model.id {config.model.id!r} != {recorded.get('model', {}).get('id')!r}")
    if mismatches:
        raise ConfigError(
            "config does not reproduce the direction run's split: " + "; ".join(mismatches)
        )


def pooled(results: Sequence[ConditionResult]) -> dict[str, Any]:
    """Pool a condition's results across prompt sets."""
    n = sum(r.n for r in results)
    k = sum(r.refusals for r in results)
    low, high = wilson_interval(k, n)
    return {
        "n": n,
        "refusals": k,
        "refusal_rate": round(k / n, 4) if n else None,
        "refusal_ci_low": round(low, 4),
        "refusal_ci_high": round(high, 4),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    iv = config.intervention
    output_root = args.output_root or config.experiment.output_root

    try:
        source = DirectionSource(resolve_direction_run(config, output_root))
        check_split_matches(config, source)
    except (FileNotFoundError, ConfigError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    target = iv.target_layer if iv.target_layer is not None else source.best_layer()
    n_layers = len(source.directions)
    for layer in (target, *iv.source_layers):
        if layer not in source.directions:
            print(f"error: layer {layer} not in direction run (0..{n_layers - 1})", file=sys.stderr)
            return 2

    run_dir = create_run_dir(output_root, config.experiment.name)
    setup_logging(level=logging.DEBUG if args.verbose else logging.INFO, log_file=run_dir / "run.log")
    determinism = set_seed(config.experiment.seed, config.experiment.deterministic)

    interp = config.interpretability
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
    recorded_counts = source.meta.get("splits")
    if recorded_counts and recorded_counts != splits.counts():
        logger.error("Rebuilt split %s differs from the direction run's %s", splits.counts(), recorded_counts)
        return 1

    harmful_sets: dict[str, list[str]] = {"jbb_heldout": splits.harmful_test}
    harmless_sets: dict[str, list[str]] = {"jbb_heldout": splits.harmless_test}
    if iv.include_bundled_prompts:
        bundled_harmful, bundled_harmless = load_bundled_prompts()
        harmful_sets["bundled"] = bundled_harmful
        harmless_sets["bundled"] = bundled_harmless

    meta = collect_metadata(
        config.experiment.name,
        extra={
            "config": config.to_dict(),
            "determinism": determinism.to_dict(),
            "direction_source": source.describe(),
            "target_layer": target,
            "prompt_sets": {
                "harmful": {k: len(v) for k, v in harmful_sets.items()},
                "harmless": {k: len(v) for k, v in harmless_sets.items()},
            },
            "run_dir": str(run_dir),
            "scope": (
                "Inference-time hooks only, removed after each condition. No weights are "
                "modified or saved. Harmful-prompt completions are classified, scored and "
                "discarded; only aggregate statistics are written."
            ),
            "reference": (
                "Arditi et al. 2024, 'Refusal in Language Models Is Mediated by a Single "
                "Direction', arXiv:2406.11717"
            ),
        },
        device_index=args.device,
    )
    write_json(run_dir / "meta.json", meta)
    if not meta["gpu"].get("available"):
        logger.error("No CUDA device available")
        return 1

    sampler = GpuSampler(
        device_index=args.device,
        interval_s=config.monitoring.sample_interval_s,
        enabled=config.monitoring.enabled,
    )
    sampler.start()

    payload: dict[str, Any] = {"meta": meta, "target_layer": target}
    rows: list[dict[str, Any]] = []
    by_condition: dict[str, list[ConditionResult]] = {}
    loaded = None
    judge = None

    with oom_guard("intervention", device=args.device) as outcome:
        loaded = load_model(config.model, interp.precision, device_index=args.device, monitor=False)
        payload["model"] = loaded.describe()
        model, tokenizer, device = loaded.model, loaded.tokenizer, loaded.device
        hidden = loaded.hidden_size
        sampler.mark("model:loaded")

        r_target = source.directions[target]
        randoms = [random_unit_direction(hidden, config.experiment.seed + k) for k in range(iv.n_random_controls)]

        # Self-check: ablation must actually remove the direction from the stream.
        probe = splits.harmful_test[:4]
        before = residual_component(model, tokenizer, probe, r_target, device)
        with DirectionalAblation(model, r_target):
            during = residual_component(model, tokenizer, probe, r_target, device)
        payload["ablation_self_check"] = {
            "intact": {k: round(v, 6) for k, v in before.items()},
            "ablated": {k: round(v, 6) for k, v in during.items()},
        }
        logger.info(
            "Ablation self-check: mean |cos| %.4f -> %.6f, max |cos| %.4f -> %.6f",
            before["mean_abs_cosine"], during["mean_abs_cosine"],
            before["max_abs_cosine"], during["max_abs_cosine"],
        )

        scorers: dict[str, Any] = {"self": model}
        if iv.judge_model_id:
            judge_cfg = ModelConfig(
                id=iv.judge_model_id, local_path=iv.judge_local_path, attn_implementation="auto"
            )
            judge = load_model(judge_cfg, "bf16", device_index=args.device, monitor=False)
            if judge.tokenizer.get_vocab() != tokenizer.get_vocab():
                raise ValueError(f"judge {iv.judge_model_id} does not share the subject's vocabulary")
            scorers["judge"] = judge.model
            payload["judge"] = judge.describe()
            sampler.mark("judge:loaded")

        def run(
            name: str,
            factory: Callable[[], Any],
            prompt_sets: dict[str, list[str]],
            extra: dict[str, Any],
            keep_examples: int = 0,
        ) -> None:
            for set_name, prompts in prompt_sets.items():
                result = evaluate_condition(
                    model,
                    tokenizer,
                    prompts,
                    factory,
                    condition=name,
                    prompt_set=set_name,
                    max_new_tokens=iv.max_new_tokens,
                    batch_size=iv.batch_size,
                    keep_examples=keep_examples,
                    extra=extra,
                    device=device,
                    scorers=scorers,
                )
                rows.append(result.to_row())
                by_condition.setdefault(f"{name}|{extra.get('target_class')}", []).append(result)
                payload.setdefault("conditions", []).append(result.to_dict())

        # --- necessity: ablation on harmful prompts --------------------------------
        harmful_extra = {"target_class": "harmful"}
        run("baseline", NoIntervention, harmful_sets, {**harmful_extra, "kind": "baseline"})
        sweep_layers = sorted(set(iv.source_layers) | {target})
        for layer in sweep_layers:
            r = source.directions[layer]
            run(
                f"ablate:L{layer}",
                lambda r=r: DirectionalAblation(model, r),
                harmful_sets,
                {
                    **harmful_extra,
                    "kind": "ablate",
                    "direction": "refusal",
                    "source_layer": layer,
                    "cosine_to_target": round(cosine_similarity(r, r_target), 4),
                    "phase5_test_cohens_d": round(source.test_d(layer), 4),
                },
            )
        for k, r in enumerate(randoms):
            run(
                f"ablate:random{k}",
                lambda r=r: DirectionalAblation(model, r),
                harmful_sets,
                {**harmful_extra, "kind": "ablate", "direction": "random", "random_seed": config.experiment.seed + k},
            )
        sampler.mark("necessity:done")

        # --- cost: what else does ablation change? ---------------------------------
        if iv.measure_capability:
            text = load_perplexity_text()
            eval_prompts = [p for prompts in harmless_sets.values() for p in prompts]

            def capability(factory: Callable[[], Any]) -> tuple[Any, torch.Tensor]:
                with factory():
                    trace = teacher_forced_pass(
                        model, tokenizer, text, max_tokens=TF_MAX_TOKENS,
                        window=TF_WINDOW, stride=TF_STRIDE, device=device,
                    )
                    final = final_position_logprobs(model, tokenizer, eval_prompts, device=device)
                return trace, final

            ref_trace, ref_final = capability(NoIntervention)
            cost: dict[str, Any] = {
                "reference_perplexity": round(ref_trace.perplexity(), 4),
                "n_positions": ref_trace.n_positions,
                "n_prompts": len(eval_prompts),
                "conditions": {},
            }
            candidates: list[tuple[str, Callable[[], Any]]] = [
                (f"ablate:L{layer}", lambda r=source.directions[layer]: DirectionalAblation(model, r))
                for layer in sweep_layers
            ]
            candidates += [
                (f"ablate:random{k}", lambda r=r: DirectionalAblation(model, r))
                for k, r in enumerate(randoms)
            ]
            for name, factory in candidates:
                trace, final = capability(factory)
                cost["conditions"][name] = {
                    "perplexity": round(trace.perplexity(), 4),
                    "teacher_forced": trace_comparison(ref_trace, trace),
                    "next_token": distribution_divergence(ref_final, final),
                }
                logger.info("Capability %-18s ppl %.3f", name, trace.perplexity())
            payload["capability"] = cost
            sampler.mark("capability:done")

        # --- selection: the causally best layer, chosen without the bundled set ----
        # Screen every layer that passes the cheap criteria (depth, perplexity) for the
        # expensive one - does adding its direction induce refusal? - on the selection
        # split only, so the bundled prompts stay out of the choice.
        depth_limit = iv.max_relative_depth * n_layers
        screen_prompts = harmless_sets[SELECTION_SET]
        screen_baseline = evaluate_condition(
            model, tokenizer, screen_prompts, NoIntervention, condition="screen:baseline",
            prompt_set=SELECTION_SET, max_new_tokens=iv.max_new_tokens, batch_size=iv.batch_size,
            device=device, scorers={"self": model},
        )
        baseline_rate = screen_baseline.refusal_rate
        selection_rows = []
        for layer in sweep_layers:
            held = next(
                r for r in by_condition[f"ablate:L{layer}|harmful"] if r.prompt_set == SELECTION_SET
            )
            ratio = (
                payload.get("capability", {})
                .get("conditions", {})
                .get(f"ablate:L{layer}", {})
                .get("teacher_forced", {})
                .get("perplexity_ratio")
            )
            row: dict[str, Any] = {
                "layer": layer, "refusals": held.refusals, "n": held.n, "perplexity_ratio": ratio,
                "harmless_baseline_rate": round(baseline_rate, 4),
                "induced_refusals": None, "induced_n": None,
            }
            if layer < depth_limit and ratio is not None and ratio <= iv.max_perplexity_ratio:
                vector = source.raw_norm(layer) * source.directions[layer]
                induced = evaluate_condition(
                    model, tokenizer, screen_prompts,
                    lambda v=vector, layer=layer: ActivationAddition(model, layer, v),
                    condition=f"screen:add:L{layer}x1", prompt_set=SELECTION_SET,
                    max_new_tokens=iv.max_new_tokens, batch_size=iv.batch_size,
                    device=device, scorers={"self": model},
                )
                row["induced_refusals"], row["induced_n"] = induced.refusals, induced.n
            selection_rows.append(row)
        selected = select_causal_layer(selection_rows, iv.max_perplexity_ratio, depth_limit)
        payload["selection"] = {
            "rule": (
                f"Arditi et al. (2024) criteria on the {SELECTION_SET} split: layers before "
                f"{iv.max_relative_depth:g} of the network's depth, whose ablation keeps corpus "
                f"perplexity within {iv.max_perplexity_ratio:g}x, and whose direction added at "
                f"1x raises harmless-prompt refusal above the unmodified rate (95% Wilson lower "
                f"bound); among those, the lowest harmful-prompt refusal with the direction "
                f"ablated, ties to lower perplexity. The bundled prompt set plays no part."
            ),
            "observational_layer": target,
            "causal_layer": selected,
            "depth_limit_exclusive": depth_limit,
            "candidates": selection_rows,
        }
        logger.info("Observational best layer %d; causal selection %s", target, selected)
        addition_layers = [target] + ([selected] if selected is not None and selected != target else [])

        # --- sufficiency: addition on harmless prompts -----------------------------
        harmless_extra = {"target_class": "harmless"}
        run("baseline", NoIntervention, harmless_sets, {**harmless_extra, "kind": "baseline"}, keep_examples=3)
        for layer in addition_layers:
            r_layer = source.directions[layer]
            raw = source.raw_norm(layer)
            run(
                f"ablate:L{layer}",
                lambda r=r_layer: DirectionalAblation(model, r),
                harmless_sets,
                {**harmless_extra, "kind": "ablate", "direction": "refusal", "source_layer": layer},
                keep_examples=3,
            )
            for c in iv.addition_coefficients:
                run(
                    f"add:L{layer}x{c:g}",
                    lambda v=c * raw * r_layer, layer=layer: ActivationAddition(model, layer, v),
                    harmless_sets,
                    {
                        **harmless_extra,
                        "kind": "add",
                        "direction": "refusal",
                        "source_layer": layer,
                        "coefficient": c,
                        "added_norm": round(c * raw, 3),
                    },
                    keep_examples=3,
                )
            for k, r in enumerate(randoms):
                run(
                    f"add:L{layer}random{k}x1",
                    lambda v=raw * r, layer=layer: ActivationAddition(model, layer, v),
                    harmless_sets,
                    {
                        **harmless_extra,
                        "kind": "add",
                        "direction": "random",
                        "source_layer": layer,
                        "coefficient": 1.0,
                        "added_norm": round(raw, 3),
                        "random_seed": config.experiment.seed + k,
                    },
                    keep_examples=3,
                )
        sampler.mark("sufficiency:done")

    sampler.stop()
    payload["telemetry"] = sampler.summary()

    if not outcome.ok:
        payload["status"] = outcome.status
        payload["failure"] = outcome.to_dict()
        write_json(run_dir / "intervention.json", payload)
        logger.error("Run failed: %s", outcome.error_message)
        return 1

    summary: dict[str, Any] = {}
    for key, results in by_condition.items():
        name, target_class = key.split("|")
        summary.setdefault(target_class, {})[name] = pooled(results)
    payload["pooled"] = summary

    layer_rows = []
    capability_by_name = payload.get("capability", {}).get("conditions", {})
    for layer in sweep_layers:
        pooled_row = summary["harmful"][f"ablate:L{layer}"]
        cap = capability_by_name.get(f"ablate:L{layer}", {})
        layer_rows.append(
            {
                "source_layer": layer,
                "phase5_test_cohens_d": round(source.test_d(layer), 4),
                "cosine_to_target": round(cosine_similarity(source.directions[layer], r_target), 4),
                **{f"harmful_{k}": v for k, v in pooled_row.items()},
                "perplexity": cap.get("perplexity"),
                "perplexity_ratio": cap.get("teacher_forced", {}).get("perplexity_ratio"),
                "harmless_next_token_kl": cap.get("next_token", {}).get("mean_kl_nats"),
            }
        )
    payload["layer_sweep"] = layer_rows
    payload["status"] = "ok"

    write_json(run_dir / "intervention.json", payload)
    write_csv(run_dir / "conditions.csv", rows)
    write_csv(run_dir / "layer_sweep.csv", layer_rows)

    loaded.model = None
    loaded.tokenizer = None
    del loaded
    if judge is not None:
        judge.model = None
        del judge
    scorers.clear()
    release_memory(args.device)

    harmful = summary["harmful"]
    harmless = summary["harmless"]
    selected = payload["selection"]["causal_layer"]
    shown = [layer for layer in dict.fromkeys([target, selected]) if layer is not None]
    print("\nCausal test of the refusal direction")
    print("-" * 64)
    print(f"  observational best layer (Phase 5):   {target}")
    print(f"  causally selected layer:              {selected}")
    print(f"  harmful refusal, intact:              {harmful['baseline']['refusal_rate']:.2f}")
    for layer in shown:
        print(f"  harmful refusal, L{layer} ablated:        {harmful[f'ablate:L{layer}']['refusal_rate']:.2f}")
    for k in range(iv.n_random_controls):
        print(f"  harmful refusal, random{k} ablated:     {harmful[f'ablate:random{k}']['refusal_rate']:.2f}")
    print(f"  harmless refusal, intact:             {harmless['baseline']['refusal_rate']:.2f}")
    for layer in shown:
        for c in iv.addition_coefficients:
            print(f"  harmless refusal, L{layer} +{c:g}x:          {harmless[f'add:L{layer}x{c:g}']['refusal_rate']:.2f}")
    print(f"  results:                              {run_dir}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
