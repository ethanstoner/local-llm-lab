"""CLI that turns finished runs into figures.

    python -m src.visualization.render

Discovers the most recent run of each kind under ``results/``, reads the result files,
and writes every figure those results can support. Figures whose inputs are missing are
skipped with a log line rather than drawn from partial or substituted data.

Each figure carries a provenance caption naming the model, the GPU and the run directory
it came from, so a PNG pulled out of ``figures/`` is still traceable to its measurements.
"""

from __future__ import annotations

import argparse
import csv
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from src.utils.io import iter_run_dirs, read_json, repo_root, resolve_under_repo
from src.utils.logging import get_logger, setup_logging
from src.visualization import plots

logger = get_logger(__name__)

#: Which result file identifies each kind of run.
RUN_KINDS = {
    "benchmark": "metrics.json",
    "quality": "quality.json",
    "refusal": "refusal_analysis.json",
    "intervention": "intervention.json",
    "interleaved": "interleaved.json",
    "roofline": "roofline.json",
}

#: The model whose interpretability runs supply the headline figures. Runs on any other
#: model feed only the cross-scale comparison, so a later run on a smaller model cannot
#: silently replace the headline figures - the same failure the recency rule below was
#: written to prevent for benchmarks.
PRIMARY_MODEL = "Qwen/Qwen2.5-7B-Instruct"

#: Kinds whose runs are filtered to the primary model.
MODEL_SCOPED_KINDS = ("refusal", "intervention")


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser."""
    parser = argparse.ArgumentParser(
        prog="python -m src.visualization.render",
        description="Render figures from finished experiment runs.",
    )
    parser.add_argument("--results-root", default="results", help="Where runs are stored")
    parser.add_argument("--figures-dir", default="figures", help="Where figures are written")
    parser.add_argument("--benchmark-run", default=None, help="Use a specific context-sweep run dir")
    parser.add_argument(
        "--precision-run", default=None, help="Use a specific precision-sweep run dir"
    )
    parser.add_argument("--quality-run", default=None, help="Use a specific quality run dir")
    parser.add_argument("--refusal-run", default=None, help="Use a specific refusal run dir")
    parser.add_argument("--primary-model", default=PRIMARY_MODEL,
                        help="Model whose interpretability runs supply the headline figures")
    parser.add_argument("--verbose", action="store_true", help="Log at DEBUG level")
    return parser


def _coerce(value: str) -> Any:
    """Turn a CSV field into a number, a bool, None or the original string."""
    if value == "":
        return None
    lowered = value.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def read_rows(path: Path) -> list[dict[str, Any]]:
    """Read a results CSV into typed dictionaries."""
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [{k: _coerce(v) for k, v in row.items()} for row in csv.DictReader(handle)]


def _n_distinct(rows: Sequence[Mapping[str, Any]], column: str) -> int:
    """How many distinct values of ``column`` a results table measured successfully."""
    return len({r.get(column) for r in rows if r.get("status") == "ok"})


def _run_model(run_dir: Path) -> str | None:
    meta_path = run_dir / "meta.json"
    if not meta_path.is_file():
        return None
    return ((read_json(meta_path).get("config") or {}).get("model") or {}).get("id")


def discover_runs(results_root: str | Path, primary_model: str = PRIMARY_MODEL) -> dict[str, Path]:
    """Return the most recent run directory of each kind.

    Runs are classified by which result file they contain rather than by experiment
    name, so a renamed experiment still renders.

    Two subtleties, both of which produced wrong figures before they were handled:

    * **Most recent means by timestamp, not by iteration order.** Run directories are
      named with a UTC timestamp, so comparing those names orders runs globally.
      Selecting whichever run happened to be visited last instead picks the
      alphabetically-last *experiment* - which is how a smoke run on a 1.5B model came
      to supply the headline throughput figures for a 7B experiment.
    * **A context sweep and a precision sweep want different figures.** They are tracked
      separately, so a precision sweep does not displace the context sweep's curves, and
      each is chosen by how richly it sweeps its own axis before recency is considered -
      a two-point context sweep run later should not replace a five-point one.
    """
    ranked: dict[str, list[tuple[int, str, Path]]] = {kind: [] for kind in RUN_KINDS}
    ranked["benchmark_precision"] = []
    ranked["decode_ab"] = []
    ranked["batch_sweep"] = []

    for run_dir in iter_run_dirs(results_root):
        for kind, marker in RUN_KINDS.items():
            if not (run_dir / marker).is_file():
                continue
            if kind == "benchmark":
                rows = read_rows(run_dir / "results.csv")
                n_precisions = _n_distinct(rows, "precision")
                n_contexts = _n_distinct(rows, "context_length")
                if n_precisions > 1:
                    ranked["benchmark_precision"].append((n_precisions, run_dir.name, run_dir))
                if n_contexts > 1:
                    ranked["benchmark"].append((n_contexts, run_dir.name, run_dir))
            elif kind == "interleaved":
                config = (read_json(run_dir / "interleaved.json").get("meta") or {}).get("config") or {}
                n_batches = len((config.get("comparison") or {}).get("batch_sizes") or [])
                n_contexts = len((config.get("benchmark") or {}).get("context_lengths") or [])
                if n_batches > 1:
                    ranked["batch_sweep"].append((n_batches, run_dir.name, run_dir))
                elif n_contexts > 1:
                    ranked["decode_ab"].append((n_contexts, run_dir.name, run_dir))
            elif kind in MODEL_SCOPED_KINDS:
                if _run_model(run_dir) == primary_model:
                    ranked[kind].append((0, run_dir.name, run_dir))
            else:
                ranked[kind].append((0, run_dir.name, run_dir))

    # Richest sweep first; directory names are sortable UTC timestamps, so they break ties
    # in favour of the most recent run.
    ranked.pop("interleaved", None)
    return {
        kind: max(entries)[2]
        for kind, entries in ranked.items()
        if entries
    }


def _provenance(meta: Mapping[str, Any], run_dir: Path) -> str:
    """Build the caption line placed under each figure."""
    gpu = (meta.get("gpu") or {}).get("name", "unknown GPU")
    model = ((meta.get("config") or {}).get("model") or {}).get("id", "unknown model")
    stamp = meta.get("timestamp_utc", "")
    try:
        relative = run_dir.relative_to(repo_root())
    except ValueError:
        relative = run_dir
    return f"{model} on {gpu} · {stamp} · {relative.as_posix()}"


def _memory_series(
    run_dir: Path, metrics: Mapping[str, Any]
) -> list[tuple[str, list[float], list[float]]]:
    """Collect per-cell GPU memory time series referenced by a benchmark run."""
    series: list[tuple[str, list[float], list[float]]] = []
    for cell in metrics.get("cells", []):
        telemetry = cell.get("telemetry") or {}
        relative = telemetry.get("samples_csv")
        if cell.get("status") != "ok" or not relative:
            continue
        rows = read_rows(run_dir / relative)
        if len(rows) < 2:
            continue
        start = rows[0]["t_rel_s"]
        times = [float(r["t_rel_s"]) - start for r in rows]
        memory = [float(r["memory_used_mib"]) for r in rows]
        context = cell["context_length"]
        label = f"{cell['precision']} · {context if context < 1000 else f'{context // 1000}k'} ctx"
        series.append((label, times, memory))
    return series


def render_benchmark(run_dir: Path, figures_dir: Path) -> list[Path]:
    """Render every figure a benchmark run supports."""
    metrics = read_json(run_dir / "metrics.json")
    meta = metrics.get("meta", {})
    caption = _provenance(meta, run_dir)
    rows = read_rows(run_dir / "results.csv")
    if not rows:
        logger.warning("No results.csv rows in %s", run_dir)
        return []

    contexts = {r["context_length"] for r in rows if r.get("status") == "ok"}
    written: list[Path] = []

    def keep(path: Path | None) -> None:
        if path is not None:
            written.append(path)

    # A sweep over contexts and a sweep over precisions want different figures; a run
    # that varies both gets both sets.
    if len(contexts) > 1:
        keep(plots.throughput_vs_context(rows, figures_dir / "throughput_vs_context.png", caption))
        keep(plots.latency_vs_context(rows, figures_dir / "latency_vs_context.png", caption))
        keep(plots.vram_vs_context(rows, figures_dir / "vram_vs_context.png", caption))

    keep(plots.inter_token_latency(rows, figures_dir / "inter_token_latency.png", caption))
    keep(
        plots.memory_over_time(
            _memory_series(run_dir, metrics), figures_dir / "memory_over_time.png", caption
        )
    )

    return written


def render_benchmark_precision(run_dir: Path, figures_dir: Path) -> list[Path]:
    """Render the precision-comparison figures from a run that swept precisions."""
    metrics = read_json(run_dir / "metrics.json")
    caption = _provenance(metrics.get("meta", {}), run_dir)
    rows = read_rows(run_dir / "results.csv")
    if not rows:
        return []

    contexts = {r["context_length"] for r in rows if r.get("status") == "ok"}
    # Hold context fixed so the comparison is between precisions and nothing else.
    target = sorted(contexts)[0] if contexts else None
    fixed = [r for r in rows if r.get("context_length") == target]
    note = f"{caption} · context fixed at {target} tokens"

    written: list[Path] = []
    for path in (
        plots.throughput_by_precision(fixed, figures_dir / "throughput_by_precision.png", note),
        plots.vram_by_precision(fixed, figures_dir / "vram_by_precision.png", note),
        plots.memory_throughput_tradeoff(
            fixed, figures_dir / "memory_throughput_tradeoff.png", note
        ),
    ):
        if path is not None:
            written.append(path)
    return written


def render_quality(run_dir: Path, figures_dir: Path) -> list[Path]:
    """Render the quantization-fidelity figure."""
    payload = read_json(run_dir / "quality.json")
    caption = _provenance(payload.get("meta", {}), run_dir)
    rows = read_rows(run_dir / "quality.csv")
    if not rows:
        logger.warning("No quality.csv rows in %s", run_dir)
        return []
    path = plots.quality_by_precision(rows, figures_dir / "quality_by_precision.png", caption)
    return [path] if path else []


def render_refusal(run_dir: Path, figures_dir: Path) -> list[Path]:
    """Render every figure a refusal-direction run supports."""
    payload = read_json(run_dir / "refusal_analysis.json")
    caption = _provenance(payload.get("meta", {}), run_dir)
    layers = payload.get("layers") or []
    written: list[Path] = []

    def keep(path: Path | None) -> None:
        if path is not None:
            written.append(path)

    keep(plots.layer_separation(layers, figures_dir / "refusal_layer_separation.png", caption))
    keep(
        plots.direction_consistency(
            payload.get("direction_agreement") or {},
            figures_dir / "refusal_direction_consistency.png",
            caption,
        )
    )
    keep(
        plots.activation_norms(
            payload.get("activation_norm_profile") or {},
            figures_dir / "activation_norms_by_layer.png",
            caption,
        )
    )
    if payload.get("pca"):
        keep(plots.pca_scatter(payload["pca"], figures_dir / "refusal_pca.png", caption))

    projections = _best_layer_projections(run_dir, payload)
    if projections:
        layer, harmful, harmless = projections
        keep(
            plots.projection_histogram(
                harmful, harmless, layer, figures_dir / "refusal_projections.png", caption
            )
        )
    return written


def render_intervention(run_dir: Path, figures_dir: Path) -> list[Path]:
    """Render the Phase 6 causal-test figures."""
    payload = read_json(run_dir / "intervention.json")
    caption = _provenance(payload.get("meta", {}), run_dir)
    written = [
        plots.intervention_overview(payload, figures_dir / "intervention_overview.png", caption),
        plots.intervention_layer_sweep(
            payload.get("layer_sweep") or [],
            payload.get("selection") or {},
            ((payload.get("pooled") or {}).get("harmful") or {}).get("baseline", {}).get("refusal_rate"),
            (((payload.get("meta") or {}).get("config") or {}).get("intervention") or {}).get("max_perplexity_ratio"),
            figures_dir / "intervention_layer_sweep.png",
            caption,
        ),
        plots.intervention_dose_response(
            payload.get("conditions") or [], figures_dir / "intervention_dose_response.png", caption
        ),
    ]
    return [p for p in written if p is not None]


def _latest_roofline_rows(results_root: Path, experiment_dir: Path) -> list[dict[str, Any]]:
    """Roofline rows for one source run, from the most recent roofline analysis."""
    runs = list(iter_run_dirs(results_root, "roofline"))
    if not runs:
        return []
    payload = read_json(runs[-1] / "roofline.json")
    wanted = experiment_dir.parent.name + "/" + experiment_dir.name
    return [r for r in payload.get("rows", []) if r.get("run") == wanted]


def render_decode_ab(run_dir: Path, figures_dir: Path) -> list[Path]:
    payload = read_json(run_dir / "interleaved.json")
    caption = _provenance(payload.get("meta", {}), run_dir)
    backends = payload["meta"]["config"]["comparison"]["backends"]
    roofline = _latest_roofline_rows(run_dir.parent.parent, run_dir)
    path = plots.decode_backend_ab(payload.get("summary") or [], roofline, backends,
                                   figures_dir / "decode_backend_ab.png", caption)
    return [path] if path else []


def render_batch_sweep(run_dir: Path, figures_dir: Path) -> list[Path]:
    payload = read_json(run_dir / "interleaved.json")
    caption = _provenance(payload.get("meta", {}), run_dir)
    backends = payload["meta"]["config"]["comparison"]["backends"]
    roofline = _latest_roofline_rows(run_dir.parent.parent, run_dir)
    path = plots.batch_throughput(payload.get("summary") or [], roofline, backends,
                                  figures_dir / "batch_throughput.png", caption)
    return [path] if path else []


def render_roofline(run_dir: Path, figures_dir: Path) -> list[Path]:
    payload = read_json(run_dir / "roofline.json")
    rows = [r for r in payload.get("rows", []) if r.get("run", "").startswith("phase1_context_sweep/")]
    caption = f"roofline analysis · {run_dir.relative_to(repo_root()).as_posix()} · ceilings from {payload.get('ceilings_run')}"
    path = plots.prefill_roofline(rows, figures_dir / "prefill_roofline.png", caption)
    return [path] if path else []


def render_cross_scale(results_root: Path, figures_dir: Path) -> list[Path]:
    """Compare the latest refusal-direction run of every model."""
    latest: dict[str, Path] = {}
    for run_dir in iter_run_dirs(results_root):
        if (run_dir / "refusal_analysis.json").is_file():
            model = _run_model(run_dir)
            if model and (model not in latest or run_dir.name > latest[model].name):
                latest[model] = run_dir
    if len(latest) < 2:
        return []
    runs = {model.split("/")[-1]: read_json(path / "refusal_analysis.json").get("layers") or []
            for model, path in latest.items()}
    caption = " · ".join(f"{m}: {p.relative_to(repo_root()).as_posix()}" for m, p in sorted(latest.items()))
    path = plots.cross_scale_layers(runs, figures_dir / "refusal_cross_scale.png", caption)
    return [path] if path else []


def _best_layer_projections(
    run_dir: Path, payload: Mapping[str, Any]
) -> tuple[int, list[float], list[float]] | None:
    """Recompute held-out projections at the best layer from the saved tensors.

    The per-prompt projections are not stored in the JSON - only their summary statistics
    are - so the histogram is rebuilt from the saved activations and directions. If
    activation saving was turned off for that run, the figure is skipped.
    """
    summary = payload.get("summary") or {}
    best = (summary.get("best_layer_by_cohens_d") or {}).get("layer")
    if best is None:
        return None

    activations_dir = run_dir / (payload.get("activations_dir") or "activations")
    directions_path = run_dir / "directions.safetensors"
    harmful_path = activations_dir / "harmful_test.safetensors"
    harmless_path = activations_dir / "harmless_test.safetensors"
    if not (directions_path.is_file() and harmful_path.is_file() and harmless_path.is_file()):
        logger.info("Saved activations unavailable; skipping the projection histogram")
        return None

    from src.interpretability.hooks import load_activations

    directions = load_activations(directions_path)
    harmful = load_activations(harmful_path)
    harmless = load_activations(harmless_path)
    if best not in directions or best not in harmful or best not in harmless:
        return None

    direction = directions[best].squeeze(0)
    return (
        int(best),
        (harmful[best] @ direction).tolist(),
        (harmless[best] @ direction).tolist(),
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Render figures for the most recent runs."""
    args = build_parser().parse_args(argv)
    setup_logging(level=logging.DEBUG if args.verbose else logging.INFO)
    plots.apply_style()

    figures_dir = resolve_under_repo(args.figures_dir)
    discovered = discover_runs(args.results_root, args.primary_model)

    overrides = {
        "benchmark": args.benchmark_run,
        "benchmark_precision": args.precision_run,
        "quality": args.quality_run,
        "refusal": args.refusal_run,
    }
    for kind, override in overrides.items():
        if override:
            discovered[kind] = resolve_under_repo(override)

    if not discovered:
        logger.error("No runs found under %s; run an experiment first", args.results_root)
        return 1

    renderers = {
        "benchmark": render_benchmark,
        "benchmark_precision": render_benchmark_precision,
        "quality": render_quality,
        "refusal": render_refusal,
        "intervention": render_intervention,
        "decode_ab": render_decode_ab,
        "batch_sweep": render_batch_sweep,
        "roofline": render_roofline,
    }

    written: list[Path] = []
    for kind, run_dir in discovered.items():
        logger.info("Rendering %s figures from %s", kind, run_dir)
        try:
            written.extend(renderers[kind](run_dir, figures_dir))
        except Exception as exc:
            logger.exception("Failed to render %s figures: %s", kind, exc)

    try:
        written.extend(render_cross_scale(resolve_under_repo(args.results_root), figures_dir))
    except Exception as exc:
        logger.exception("Failed to render the cross-scale figure: %s", exc)

    if not written:
        logger.warning("No figures were produced; the runs contained no usable measurements")
        return 1

    print(f"\nWrote {len(written)} figures to {figures_dir}:")
    for path in written:
        print(f"  {path.name}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
