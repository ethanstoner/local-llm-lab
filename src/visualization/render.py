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
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.utils.io import iter_run_dirs, read_json, repo_root, resolve_under_repo
from src.utils.logging import get_logger, setup_logging
from src.visualization import plots

logger = get_logger(__name__)

#: Which result file identifies each kind of run.
RUN_KINDS = {
    "benchmark": "metrics.json",
    "quality": "quality.json",
    "refusal": "refusal_analysis.json",
}


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser."""
    parser = argparse.ArgumentParser(
        prog="python -m src.visualization.render",
        description="Render figures from finished experiment runs.",
    )
    parser.add_argument("--results-root", default="results", help="Where runs are stored")
    parser.add_argument("--figures-dir", default="figures", help="Where figures are written")
    parser.add_argument("--benchmark-run", default=None, help="Use a specific benchmark run dir")
    parser.add_argument("--quality-run", default=None, help="Use a specific quality run dir")
    parser.add_argument("--refusal-run", default=None, help="Use a specific refusal run dir")
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


def discover_runs(results_root: str | Path) -> dict[str, Path]:
    """Return the most recent run directory of each kind.

    Runs are classified by which result file they contain rather than by experiment
    name, so a renamed experiment still renders.
    """
    found: dict[str, Path] = {}
    for run_dir in iter_run_dirs(results_root):
        for kind, marker in RUN_KINDS.items():
            if (run_dir / marker).is_file():
                # iter_run_dirs yields oldest first, so the last write wins.
                found[kind] = run_dir
    return found


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
    precisions = {r["precision"] for r in rows if r.get("status") == "ok"}
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

    if len(precisions) > 1:
        # Hold context fixed so the comparison is between precisions and nothing else.
        target = sorted(contexts)[0] if contexts else None
        fixed = [r for r in rows if r.get("context_length") == target]
        note = f"{caption} · context fixed at {target} tokens"
        keep(plots.throughput_by_precision(fixed, figures_dir / "throughput_by_precision.png", note))
        keep(plots.vram_by_precision(fixed, figures_dir / "vram_by_precision.png", note))
        keep(
            plots.memory_throughput_tradeoff(
                fixed, figures_dir / "memory_throughput_tradeoff.png", note
            )
        )
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
    discovered = discover_runs(args.results_root)

    overrides = {
        "benchmark": args.benchmark_run,
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
        "quality": render_quality,
        "refusal": render_refusal,
    }

    written: list[Path] = []
    for kind, run_dir in discovered.items():
        logger.info("Rendering %s figures from %s", kind, run_dir)
        try:
            written.extend(renderers[kind](run_dir, figures_dir))
        except Exception as exc:  # noqa: BLE001 - one bad run must not block the others
            logger.exception("Failed to render %s figures: %s", kind, exc)

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
