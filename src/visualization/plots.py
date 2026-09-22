"""Publication-style matplotlib figures built from result files.

Every function here takes data that was read off disk and returns the path it wrote, or
``None`` when the inputs for that figure are absent. Nothing is interpolated, defaulted
or invented: a figure that cannot be drawn from real measurements is not drawn.

Design rules applied throughout, in order of how often they are got wrong:

* **Never two y-axes.** Measures on different scales go in separate panels, never
  overlaid on a twin axis where the crossing point is an artifact of the scaling.
* **Colour follows the entity, not its rank.** ``bf16`` is the same blue in every
  figure whether or not ``fp32`` succeeded, so figures can be compared to each other.
* Categorical hues are assigned from a fixed, contrast-validated order and never
  cycled; scatter plots that would need more than three hues use one colour plus
  direct labels instead.
* Bars carry direct value labels, because several palette slots sit below a 3:1
  contrast ratio against the surface and must not rely on colour alone.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # No display on this machine; render straight to file.

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from src.utils.logging import get_logger

logger = get_logger(__name__)

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e3e2de"
NEUTRAL_MARK = "#52514e"

#: Fixed categorical order, validated for colour-vision deficiency separation on this
#: surface (worst adjacent pair dE 9.1 protan, 19.6 normal vision).
CATEGORICAL = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")

#: Stable slot per precision so a precision keeps its colour across every figure.
PRECISION_COLORS: dict[str, str] = {
    "bf16": CATEGORICAL[0],
    "fp16": CATEGORICAL[1],
    "int8": CATEGORICAL[2],
    "nf4": CATEGORICAL[3],
    "fp4": CATEGORICAL[4],
    "fp32": CATEGORICAL[6],
}

CLASS_COLORS = {"harmful": CATEGORICAL[7], "harmless": CATEGORICAL[0]}


def apply_style() -> None:
    """Set the shared matplotlib style for every figure in this project."""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": TEXT_SECONDARY,
            "axes.titlecolor": TEXT_PRIMARY,
            "axes.titlesize": 12,
            "axes.titleweight": "semibold",
            "axes.titlelocation": "left",
            "axes.titlepad": 10,
            "axes.labelsize": 10,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": TEXT_SECONDARY,
            "ytick.color": TEXT_SECONDARY,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "legend.labelcolor": TEXT_SECONDARY,
            "lines.linewidth": 2.0,
            "lines.markersize": 6,
            "figure.dpi": 130,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "font.size": 10,
        }
    )


def _finish(fig: plt.Figure, path: Path, subtitle: str | None = None) -> Path:
    """Add provenance, write the figure and close it."""
    if subtitle:
        fig.text(
            0.0,
            -0.02,
            subtitle,
            fontsize=8,
            color=TEXT_SECONDARY,
            ha="left",
            va="top",
            transform=fig.transFigure,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    logger.info("Wrote %s", path)
    return path


def _despine(ax: plt.Axes) -> None:
    """Remove the top and right spines so the data reads before the frame does."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.grid(axis="x", visible=False)


def _label_bars(ax: plt.Axes, bars: Any, values: Sequence[float], fmt: str = "{:.0f}") -> None:
    """Write each bar's value above it, in text ink rather than the series colour."""
    for bar, value in zip(bars, values):
        if value is None:
            continue
        ax.annotate(
            fmt.format(value),
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color=TEXT_PRIMARY,
        )


def _context_formatter() -> FuncFormatter:
    """Format context lengths as 128 / 2k / 16k."""
    return FuncFormatter(lambda v, _: f"{int(v/1000)}k" if v >= 1000 else f"{int(v)}")


# --------------------------------------------------------------------------------------
# Phase 1: context-length behaviour
# --------------------------------------------------------------------------------------


def throughput_vs_context(rows: Sequence[Mapping[str, Any]], path: Path, subtitle: str = "") -> Path | None:
    """Decode and end-to-end throughput against prompt length.

    Two series in the same unit, so one axis is correct. They diverge as context grows
    because prefill starts to dominate the end-to-end figure - which is the point.
    """
    usable = [r for r in rows if r.get("status") == "ok" and r.get("decode_tokens_per_s_median")]
    if not usable:
        return None

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    by_precision: dict[str, list[Mapping[str, Any]]] = {}
    for row in usable:
        by_precision.setdefault(str(row["precision"]), []).append(row)

    for precision, group in by_precision.items():
        group = sorted(group, key=lambda r: r["context_length"])
        x = [r["context_length"] for r in group]
        color = PRECISION_COLORS.get(precision, NEUTRAL_MARK)
        ax.plot(
            x,
            [r["decode_tokens_per_s_median"] for r in group],
            marker="o",
            color=color,
            label=f"{precision} decode",
        )
        e2e = [r.get("end_to_end_tokens_per_s_median") for r in group]
        if all(v is not None for v in e2e):
            ax.plot(
                x,
                e2e,
                marker="s",
                linestyle="--",
                color=color,
                alpha=0.55,
                label=f"{precision} end-to-end",
            )

    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(_context_formatter())
    ax.set_xticks([r["context_length"] for r in usable])
    ax.set_xlabel("prompt length (tokens)")
    ax.set_ylabel("tokens / second")
    ax.set_title("Generation throughput against prompt length")
    ax.set_ylim(bottom=0)
    ax.legend(loc="lower left")
    _despine(ax)
    return _finish(fig, path, subtitle)


def latency_vs_context(rows: Sequence[Mapping[str, Any]], path: Path, subtitle: str = "") -> Path | None:
    """Time to first token and dedicated prefill latency against prompt length.

    Both are milliseconds, so they share an axis. Plotting them together is the
    cross-check: TTFT measured inside ``generate`` should track the separately timed
    prefill pass, and a gap between them is a measurement problem worth seeing.
    """
    usable = [r for r in rows if r.get("status") == "ok" and r.get("ttft_s_median")]
    if not usable:
        return None

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    by_precision: dict[str, list[Mapping[str, Any]]] = {}
    for row in usable:
        by_precision.setdefault(str(row["precision"]), []).append(row)

    for precision, group in by_precision.items():
        group = sorted(group, key=lambda r: r["context_length"])
        x = [r["context_length"] for r in group]
        color = PRECISION_COLORS.get(precision, NEUTRAL_MARK)
        ax.plot(
            x,
            [r["ttft_s_median"] * 1000 for r in group],
            marker="o",
            color=color,
            label=f"{precision} time to first token",
        )
        prefill = [r.get("prefill_latency_s_median") for r in group]
        if all(v is not None for v in prefill):
            ax.plot(
                x,
                [v * 1000 for v in prefill],
                marker="s",
                linestyle="--",
                color=color,
                alpha=0.55,
                label=f"{precision} prefill pass",
            )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(_context_formatter())
    ax.set_xticks([r["context_length"] for r in usable])
    ax.set_xlabel("prompt length (tokens)")
    ax.set_ylabel("latency (ms, log scale)")
    ax.set_title("Prompt-processing latency against prompt length")
    ax.legend(loc="upper left")
    _despine(ax)
    return _finish(fig, path, subtitle)


def vram_vs_context(rows: Sequence[Mapping[str, Any]], path: Path, subtitle: str = "") -> Path | None:
    """Peak device memory against prompt length, as the driver reports it."""
    usable = [
        r for r in rows if r.get("status") == "ok" and r.get("gpu_peak_memory_used_mib")
    ]
    if not usable:
        return None

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    groups = sorted({str(r["precision"]) for r in usable})
    width = 0.8 / len(groups)

    positions = sorted({r["context_length"] for r in usable})
    index = {c: i for i, c in enumerate(positions)}

    for slot, precision in enumerate(groups):
        group = sorted(
            (r for r in usable if str(r["precision"]) == precision),
            key=lambda r: r["context_length"],
        )
        xs = [index[r["context_length"]] + slot * width - 0.4 + width / 2 for r in group]
        values = [r["gpu_peak_memory_used_mib"] for r in group]
        bars = ax.bar(
            xs,
            values,
            width=width * 0.92,
            color=PRECISION_COLORS.get(precision, NEUTRAL_MARK),
            label=precision,
        )
        _label_bars(ax, bars, values)

    ax.set_xticks(range(len(positions)))
    ax.set_xticklabels([f"{c/1000:.0f}k" if c >= 1000 else str(c) for c in positions])
    ax.set_xlabel("prompt length (tokens)")
    ax.set_ylabel("peak device memory (MiB)")
    ax.set_title("Peak VRAM against prompt length")
    if len(groups) > 1:
        ax.legend(loc="upper left")
    _despine(ax)
    return _finish(fig, path, subtitle)


def memory_over_time(
    series: Sequence[tuple[str, Sequence[float], Sequence[float]]],
    path: Path,
    subtitle: str = "",
) -> Path | None:
    """Device memory during generation, one line per measured configuration.

    Args:
        series: ``(label, times_s, memory_mib)`` triples.
    """
    usable = [(label, t, m) for label, t, m in series if len(t) > 1]
    if not usable:
        return None

    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    for slot, (label, times, memory) in enumerate(usable):
        ax.plot(times, memory, color=CATEGORICAL[slot % len(CATEGORICAL)], label=label)

    ax.set_xlabel("time within measured region (s)")
    ax.set_ylabel("device memory in use (MiB)")
    ax.set_title("Device memory over the generation window")
    ax.legend(loc="lower right", ncols=2)
    _despine(ax)
    return _finish(fig, path, subtitle)


def inter_token_latency(rows: Sequence[Mapping[str, Any]], path: Path, subtitle: str = "") -> Path | None:
    """Median, p90 and p99 inter-token latency per context length."""
    usable = [r for r in rows if r.get("status") == "ok" and r.get("itl_p50_ms_median")]
    if not usable:
        return None

    usable = sorted(usable, key=lambda r: (str(r["precision"]), r["context_length"]))
    labels = [
        f"{r['precision']}\n{int(r['context_length']/1000)}k"
        if r["context_length"] >= 1000
        else f"{r['precision']}\n{r['context_length']}"
        for r in usable
    ]
    quantiles = (("itl_p50_ms_median", "p50"), ("itl_p90_ms_median", "p90"), ("itl_p99_ms_median", "p99"))

    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    width = 0.26
    for slot, (key, name) in enumerate(quantiles):
        values = [r.get(key) or 0.0 for r in usable]
        xs = [i + (slot - 1) * width for i in range(len(usable))]
        ax.bar(xs, values, width=width * 0.92, color=CATEGORICAL[slot], label=name)

    ax.set_xticks(range(len(usable)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("inter-token latency (ms)")
    ax.set_title("Inter-token latency distribution")
    ax.legend(loc="upper left")
    _despine(ax)
    return _finish(fig, path, subtitle)


# --------------------------------------------------------------------------------------
# Phase 2: precision comparison
# --------------------------------------------------------------------------------------


def _precision_bar(
    rows: Sequence[Mapping[str, Any]],
    value_key: str,
    title: str,
    ylabel: str,
    path: Path,
    fmt: str = "{:.1f}",
    subtitle: str = "",
) -> Path | None:
    """Draw one bar per precision for a single measure."""
    usable = [r for r in rows if r.get(value_key) is not None]
    if not usable:
        return None

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    labels = [str(r["precision"]) for r in usable]
    values = [r[value_key] for r in usable]
    colors = [PRECISION_COLORS.get(p, NEUTRAL_MARK) for p in labels]
    bars = ax.bar(labels, values, color=colors, width=0.62)
    _label_bars(ax, bars, values, fmt=fmt)

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(top=max(values) * 1.15)
    _despine(ax)
    return _finish(fig, path, subtitle)


def throughput_by_precision(rows: Sequence[Mapping[str, Any]], path: Path, subtitle: str = "") -> Path | None:
    """Decode throughput for each precision at a fixed context length."""
    return _precision_bar(
        [r for r in rows if r.get("status") == "ok"],
        "decode_tokens_per_s_median",
        "Decode throughput by precision",
        "tokens / second",
        path,
        fmt="{:.1f}",
        subtitle=subtitle,
    )


def vram_by_precision(rows: Sequence[Mapping[str, Any]], path: Path, subtitle: str = "") -> Path | None:
    """Peak device memory for each precision at a fixed context length."""
    return _precision_bar(
        [r for r in rows if r.get("status") == "ok"],
        "gpu_peak_memory_used_mib",
        "Peak VRAM by precision",
        "peak device memory (MiB)",
        path,
        fmt="{:.0f}",
        subtitle=subtitle,
    )


def memory_throughput_tradeoff(
    rows: Sequence[Mapping[str, Any]], path: Path, subtitle: str = ""
) -> Path | None:
    """Throughput against memory, one labelled point per precision.

    A scatter would need a distinct hue per point, and beyond three hues the all-pairs
    colour-separation floor cannot be met. One neutral mark with direct labels carries
    the same information and stays readable in greyscale.
    """
    usable = [
        r
        for r in rows
        if r.get("status") == "ok"
        and r.get("decode_tokens_per_s_median")
        and r.get("gpu_peak_memory_used_mib")
    ]
    if len(usable) < 2:
        return None

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    xs = [r["gpu_peak_memory_used_mib"] for r in usable]
    ys = [r["decode_tokens_per_s_median"] for r in usable]
    ax.scatter(xs, ys, s=70, color=NEUTRAL_MARK, zorder=3, edgecolors=SURFACE, linewidths=2)

    # bf16 and fp16 land almost exactly on top of each other - same weights, same speed -
    # so labels are nudged apart when points are close in axis-fraction terms.
    x_span = (max(xs) - min(xs)) or 1.0
    y_span = (max(ys) - min(ys)) or 1.0
    placed: list[tuple[float, float]] = []

    for row, x, y in zip(usable, xs, ys):
        offset_y = 4.0
        for px, py in placed:
            close_x = abs(x - px) / x_span < 0.06
            close_y = abs(y - py) / y_span < 0.06
            if close_x and close_y:
                offset_y -= 13.0
        placed.append((x, y))
        ax.annotate(
            str(row["precision"]),
            xy=(x, y),
            xytext=(8, offset_y),
            textcoords="offset points",
            fontsize=9.5,
            color=TEXT_PRIMARY,
        )

    ax.set_xlabel("peak device memory (MiB)")
    ax.set_ylabel("decode tokens / second")
    ax.set_title("What memory buys, and what it costs")
    _despine(ax)
    return _finish(fig, path, subtitle)


def quality_by_precision(
    rows: Sequence[Mapping[str, Any]], path: Path, subtitle: str = ""
) -> Path | None:
    """Fidelity to the reference, as two panels rather than one twin-axis chart.

    Perplexity ratio and top-1 agreement are on incompatible scales; overlaying them on
    two y-axes would put their crossing point wherever the scaling happened to place it.
    """
    usable = [r for r in rows if r.get("status") == "ok"]
    if not usable:
        return None

    panels = (
        ("tf_perplexity_ratio", "perplexity relative to reference", "{:.3f}", 1.0),
        ("tf_top1_agreement", "teacher-forced top-1 agreement", "{:.3f}", 1.0),
    )
    available = [p for p in panels if any(r.get(p[0]) is not None for r in usable)]
    if not available:
        return None

    fig, axes = plt.subplots(1, len(available), figsize=(5.4 * len(available), 4.0))
    if len(available) == 1:
        axes = [axes]

    for ax, (key, label, fmt, reference_line) in zip(axes, available):
        subset = [r for r in usable if r.get(key) is not None]
        names = [str(r["precision"]) for r in subset]
        values = [r[key] for r in subset]
        bars = ax.bar(
            names,
            values,
            color=[PRECISION_COLORS.get(n, NEUTRAL_MARK) for n in names],
            width=0.6,
        )
        _label_bars(ax, bars, values, fmt=fmt)
        ax.axhline(reference_line, color=TEXT_SECONDARY, linewidth=1.0, linestyle=":")
        ax.annotate(
            "reference",
            xy=(len(names) - 0.5, reference_line),
            xytext=(0, 4),
            textcoords="offset points",
            ha="right",
            fontsize=8,
            color=TEXT_SECONDARY,
        )
        ax.set_ylabel(label)
        ax.set_ylim(0, max(max(values) * 1.2, reference_line * 1.2))
        _despine(ax)

    fig.suptitle(
        "Quantization fidelity against the bf16 reference",
        x=0.0,
        ha="left",
        fontsize=12,
        fontweight="semibold",
        color=TEXT_PRIMARY,
    )
    fig.tight_layout()
    return _finish(fig, path, subtitle)


# --------------------------------------------------------------------------------------
# Phase 5: refusal direction
# --------------------------------------------------------------------------------------


def layer_separation(
    layers: Sequence[Mapping[str, Any]], path: Path, subtitle: str = ""
) -> Path | None:
    """Layer-wise separation, as two panels on their natural scales.

    Cohen's *d* is unbounded and AUROC lives in [0, 1]; they answer different questions
    and get their own axes rather than being forced onto one.
    """
    if not layers:
        return None

    ordered = sorted(layers, key=lambda r: r["layer"])
    x = [r["layer"] for r in ordered]
    train_d = [r["train"]["cohens_d"] for r in ordered]
    test_d = [r["test"]["cohens_d"] for r in ordered]
    test_auroc = [r["test"]["auroc"] for r in ordered]

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.4), sharex=True)

    axes[0].plot(x, test_d, marker="o", color=CATEGORICAL[0], label="held-out split")
    axes[0].plot(x, train_d, marker="s", linestyle="--", color=CATEGORICAL[1], alpha=0.7, label="fitting split")
    axes[0].set_ylabel("Cohen's d")
    axes[0].set_title("Separation along the difference-in-means direction, by layer")
    axes[0].legend(loc="upper left")
    _despine(axes[0])

    axes[1].plot(x, test_auroc, marker="o", color=CATEGORICAL[0])
    axes[1].axhline(0.5, color=TEXT_SECONDARY, linewidth=1.0, linestyle=":")
    axes[1].annotate(
        "chance",
        xy=(x[-1], 0.5),
        xytext=(-4, 5),
        textcoords="offset points",
        ha="right",
        fontsize=8,
        color=TEXT_SECONDARY,
    )
    axes[1].set_ylabel("AUROC (held-out)")
    axes[1].set_xlabel("transformer block index")
    axes[1].set_ylim(0.0, 1.05)
    _despine(axes[1])

    best = max(ordered, key=lambda r: abs(r["test"]["cohens_d"]))
    for ax in axes:
        ax.axvline(best["layer"], color=GRID, linewidth=8, zorder=0)

    # Flip the label inwards when the peak is near the right edge, or it overflows
    # the axes and gets clipped.
    on_the_right = best["layer"] > (x[0] + x[-1]) / 2
    axes[0].annotate(
        f"peak: layer {best['layer']}",
        xy=(best["layer"], max(test_d)),
        xytext=(-6 if on_the_right else 6, -2),
        textcoords="offset points",
        ha="right" if on_the_right else "left",
        fontsize=9,
        color=TEXT_PRIMARY,
    )

    fig.tight_layout()
    return _finish(fig, path, subtitle)


def projection_histogram(
    harmful: Sequence[float],
    harmless: Sequence[float],
    layer: int,
    path: Path,
    subtitle: str = "",
) -> Path | None:
    """Distribution of held-out projections onto the direction at one layer."""
    if not harmful or not harmless:
        return None

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    bins = 24
    ax.hist(
        harmless,
        bins=bins,
        color=CLASS_COLORS["harmless"],
        alpha=0.75,
        label="harmless (held out)",
    )
    ax.hist(
        harmful,
        bins=bins,
        color=CLASS_COLORS["harmful"],
        alpha=0.75,
        label="refusal-eliciting (held out)",
    )
    ax.set_xlabel(f"projection onto the layer-{layer} direction")
    ax.set_ylabel("prompts")
    ax.set_title(f"Held-out projections at layer {layer}")
    ax.legend(loc="upper center")
    _despine(ax)
    return _finish(fig, path, subtitle)


def pca_scatter(pca_payload: Mapping[str, Any], path: Path, subtitle: str = "") -> Path | None:
    """Two-component PCA of held-out activations at the best-separating layer."""
    harmful = pca_payload.get("harmful") or []
    harmless = pca_payload.get("harmless") or []
    if not harmful or not harmless:
        return None

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    ax.scatter(
        [p[0] for p in harmless],
        [p[1] for p in harmless],
        s=42,
        color=CLASS_COLORS["harmless"],
        label="harmless",
        edgecolors=SURFACE,
        linewidths=1.2,
    )
    ax.scatter(
        [p[0] for p in harmful],
        [p[1] for p in harmful],
        s=42,
        color=CLASS_COLORS["harmful"],
        label="refusal-eliciting",
        marker="^",
        edgecolors=SURFACE,
        linewidths=1.2,
    )

    ratios = pca_payload.get("explained_variance_ratio") or [0.0, 0.0]
    ax.set_xlabel(f"PC1 ({ratios[0]*100:.1f}% of variance)")
    ax.set_ylabel(f"PC2 ({ratios[1]*100:.1f}% of variance)")
    ax.set_title(f"Activation structure at layer {pca_payload.get('layer')}")
    ax.legend(loc="best")
    _despine(ax)
    return _finish(fig, path, subtitle)


def direction_consistency(
    agreement: Mapping[str, Any], path: Path, subtitle: str = ""
) -> Path | None:
    """Cosine similarity of each layer's direction to the best layer's direction.

    A weak, observation-only test of the paper's "single direction" claim: if the same
    feature is being read at different depths, these should stay high across the region
    where the signal is strong.
    """
    entries = agreement.get("cosine_to_best_layer") or []
    if not entries:
        return None

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    x = [e["layer"] for e in entries]
    y = [e["cosine_to_best_layer"] for e in entries]
    ax.plot(x, y, marker="o", color=CATEGORICAL[0])
    ax.axhline(0.0, color=TEXT_SECONDARY, linewidth=1.0, linestyle=":")
    ax.axvline(agreement.get("best_layer", x[0]), color=GRID, linewidth=8, zorder=0)
    ax.set_xlabel("transformer block index")
    ax.set_ylabel(f"cosine similarity to layer {agreement.get('best_layer')}")
    ax.set_title("Is it the same direction at every depth?")
    ax.set_ylim(-1.05, 1.05)
    _despine(ax)
    return _finish(fig, path, subtitle)


def activation_norms(
    profile: Mapping[str, Sequence[Mapping[str, Any]]], path: Path, subtitle: str = ""
) -> Path | None:
    """Mean residual-stream norm by layer, for both prompt classes.

    Context for the projection figures: raw projections scale with activation norm, and
    norm grows steeply with depth in most models.
    """
    harmful = profile.get("harmful_test") or []
    harmless = profile.get("harmless_test") or []
    if not harmful or not harmless:
        return None

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.plot(
        [r["layer"] for r in harmless],
        [r["mean"] for r in harmless],
        marker="o",
        color=CLASS_COLORS["harmless"],
        label="harmless",
    )
    ax.plot(
        [r["layer"] for r in harmful],
        [r["mean"] for r in harmful],
        marker="^",
        color=CLASS_COLORS["harmful"],
        label="refusal-eliciting",
    )
    ax.set_yscale("log")
    ax.set_xlabel("transformer block index")
    ax.set_ylabel("mean activation L2 norm (log scale)")
    ax.set_title("Residual-stream norm by depth")
    ax.legend(loc="upper left")
    _despine(ax)
    return _finish(fig, path, subtitle)


# --------------------------------------------------------------------------------------
# Phase 6: causal test of the refusal direction
# --------------------------------------------------------------------------------------

#: One hue for "the refusal direction", one for controls; baseline in ink.
INTERVENTION_COLOR = CATEGORICAL[7]
CONTROL_COLOR = "#a3a19b"
BASELINE_COLOR = NEUTRAL_MARK


def _rate_bars(
    ax: plt.Axes,
    entries: Sequence[tuple[str, Mapping[str, Any], str]],
    title: str,
) -> None:
    """Horizontal refusal-rate bars with Wilson intervals and direct labels."""
    labels = [e[0] for e in entries]
    rates = [e[1]["refusal_rate"] * 100 for e in entries]
    low = [(e[1]["refusal_rate"] - e[1]["refusal_ci_low"]) * 100 for e in entries]
    high = [(e[1]["refusal_ci_high"] - e[1]["refusal_rate"]) * 100 for e in entries]
    colors = [e[2] for e in entries]
    y = list(range(len(entries)))[::-1]
    ax.barh(y, rates, color=colors, height=0.62, xerr=[low, high],
            error_kw={"ecolor": TEXT_SECONDARY, "elinewidth": 1.0, "capsize": 2.5})
    for yi, rate, entry in zip(y, rates, entries):
        ax.annotate(
            f"{rate:.0f}%  ({entry[1]['refusals']}/{entry[1]['n']})",
            xy=(min(rate + (entry[1]["refusal_ci_high"] * 100 - rate) + 1.5, 100), yi),
            va="center", ha="left", fontsize=8.5, color=TEXT_PRIMARY,
        )
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 118)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("refusal rate (%), 95% Wilson interval")
    ax.set_title(title)
    ax.grid(axis="y", visible=False)
    _despine(ax)


def intervention_overview(payload: Mapping[str, Any], path: Path, subtitle: str = "") -> Path | None:
    """Necessity and sufficiency side by side, each against its controls.

    Necessity shows three layers' directions: Phase 5's observational pick, the
    selection rule's pick, and the cheapest layer that brings refusal to 5% or below
    (lowest corpus perplexity among them) - each labelled with why it is there.
    Sufficiency shows the whole addition dose at the observational layer.
    """
    pooled = payload.get("pooled") or {}
    harmful, harmless = pooled.get("harmful"), pooled.get("harmless")
    selection = payload.get("selection") or {}
    if not harmful or not harmless:
        return None
    observed = selection.get("observational_layer")
    picked = selection.get("causal_layer")

    sweep = payload.get("layer_sweep") or []
    effective = [r for r in sweep if r.get("harmful_refusal_rate") is not None
                 and r["harmful_refusal_rate"] <= 0.05 and r.get("perplexity_ratio") is not None]
    cheapest = min(effective, key=lambda r: r["perplexity_ratio"])["source_layer"] if effective else None

    reasons: dict[int, list[str]] = {}
    for layer, why in ((cheapest, "cheapest to reach <=5%"), (observed, "Phase 5 best separation"),
                       (picked, "selection rule's pick")):
        if layer is not None:
            reasons.setdefault(layer, []).append(why)

    necessity = [("no intervention", harmful["baseline"], BASELINE_COLOR)]
    for layer, why in reasons.items():
        necessity.append((f"ablate layer {layer} direction\n({'; '.join(why)})", harmful[f"ablate:L{layer}"],
                          INTERVENTION_COLOR))
    for key in sorted(k for k in harmful if k.startswith("ablate:random")):
        necessity.append((f"ablate random direction {key[-1]}", harmful[key], CONTROL_COLOR))
    low_signal = sorted((k for k in harmful if k.startswith("ablate:L") and int(k[8:]) <= 4), key=lambda k: int(k[8:]))
    if low_signal:
        key = low_signal[0]
        necessity.append((f"ablate layer {key[8:]} direction\n(weakest Phase 5 signal)", harmful[key], CONTROL_COLOR))

    sufficiency = [("no intervention", harmless["baseline"], BASELINE_COLOR)]
    doses = sorted((k for k in harmless if k.startswith(f"add:L{observed}x")), key=lambda k: float(k.split("x")[-1]))
    for key in doses:
        c = float(key.split("x")[-1])
        if c >= 1.0:
            sufficiency.append((f"add layer {observed} direction x{c:g}", harmless[key], INTERVENTION_COLOR))
    for key in sorted(k for k in harmless if k.startswith(f"add:L{observed}random")):
        sufficiency.append((f"add random vector {key.split('random')[1][0]}\n(same norm as x1)", harmless[key], CONTROL_COLOR))

    height = 0.55 * max(len(necessity), len(sufficiency)) + 1.6
    fig, axes = plt.subplots(1, 2, figsize=(12.6, height))
    _rate_bars(axes[0], necessity, "Necessity: refusal on harmful prompts")
    _rate_bars(axes[1], sufficiency, "Sufficiency: refusal on harmless prompts")
    fig.suptitle("Removing the direction stops refusal; adding it induces refusal",
                 x=0.0, ha="left", fontsize=12.5, fontweight="semibold", color=TEXT_PRIMARY)
    fig.tight_layout()
    return _finish(fig, path, subtitle)


def intervention_layer_sweep(rows: Sequence[Mapping[str, Any]], selection: Mapping[str, Any],
                             baseline_rate: float | None, budget: float | None,
                             path: Path, subtitle: str = "") -> Path | None:
    """Observation vs intervention vs cost, one panel each, sharing the layer axis."""
    usable = sorted((r for r in rows if r.get("harmful_refusal_rate") is not None), key=lambda r: r["source_layer"])
    if not usable:
        return None
    x = [r["source_layer"] for r in usable]
    fig, axes = plt.subplots(3, 1, figsize=(7.4, 8.4), sharex=True)

    axes[0].plot(x, [r["phase5_test_cohens_d"] for r in usable], marker="o", color=CATEGORICAL[0])
    axes[0].set_ylabel("held-out Cohen's d")
    axes[0].set_title("Observation (Phase 5): how well each layer's direction separates prompts")

    rate = [r["harmful_refusal_rate"] * 100 for r in usable]
    lo = [r["harmful_refusal_ci_low"] * 100 for r in usable]
    hi = [r["harmful_refusal_ci_high"] * 100 for r in usable]
    axes[1].fill_between(x, lo, hi, color=INTERVENTION_COLOR, alpha=0.15, linewidth=0)
    axes[1].plot(x, rate, marker="o", color=INTERVENTION_COLOR)
    if baseline_rate is not None:
        axes[1].axhline(baseline_rate * 100, color=TEXT_SECONDARY, linestyle=":", linewidth=1.0)
        axes[1].annotate("no intervention", xy=(x[0], baseline_rate * 100), xytext=(2, -11),
                         textcoords="offset points", fontsize=8, color=TEXT_SECONDARY)
    axes[1].set_ylim(0, 105)
    axes[1].set_ylabel("harmful refusal (%)")
    axes[1].set_title("Intervention: refusal left after ablating that layer's direction")

    ratio = [r.get("perplexity_ratio") for r in usable]
    if all(v is not None for v in ratio):
        axes[2].plot(x, [(v - 1) * 100 for v in ratio], marker="o", color=CATEGORICAL[3])
        if budget is not None:
            axes[2].axhline((budget - 1) * 100, color=TEXT_SECONDARY, linestyle=":", linewidth=1.0)
            axes[2].annotate(f"selection budget (+{(budget - 1) * 100:.0f}%)", xy=(x[0], (budget - 1) * 100),
                             xytext=(2, 4), textcoords="offset points", fontsize=8, color=TEXT_SECONDARY)
        axes[2].set_yscale("symlog", linthresh=1.0)
        axes[2].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:+.0f}%" if v else "0%"))
    axes[2].set_ylabel("perplexity change (%)")
    axes[2].set_xlabel("layer the direction was fitted at")
    axes[2].set_title("Cost: corpus perplexity with that direction ablated")

    causal, observed = selection.get("causal_layer"), selection.get("observational_layer")
    for ax in axes:
        for layer, style in ((observed, (0, (1, 2))), (causal, "-")):
            if layer is not None:
                ax.axvline(layer, color=GRID, linewidth=6, zorder=0, linestyle=style)
        _despine(ax)
    if causal is not None:
        axes[1].annotate(f"selection rule's pick: {causal}", xy=(causal, 100), xytext=(-8, -4),
                         textcoords="offset points", ha="right", fontsize=8.5, color=TEXT_PRIMARY)
    if observed is not None and observed != causal:
        axes[0].annotate(f"observational pick: {observed}", xy=(observed, max(r["phase5_test_cohens_d"] for r in usable)),
                         xytext=(-4, -2), textcoords="offset points", ha="right", fontsize=8.5, color=TEXT_PRIMARY)
    fig.tight_layout()
    return _finish(fig, path, subtitle)


def intervention_dose_response(conditions: Sequence[Mapping[str, Any]], path: Path,
                               subtitle: str = "") -> Path | None:
    """Harmless-prompt refusal against the size of the added vector, per layer."""
    adds = [c for c in conditions if c.get("kind") == "add" and c.get("target_class") == "harmless"]
    if not adds:
        return None
    pooled: dict[tuple[int, str, float], list[Mapping[str, Any]]] = {}
    for c in adds:
        pooled.setdefault((c["source_layer"], c["direction"], c["coefficient"]), []).append(c)

    def rate(group: Sequence[Mapping[str, Any]]) -> tuple[float, int, int]:
        k = sum(g["refusals"] for g in group)
        n = sum(g["n"] for g in group)
        return 100.0 * k / n, k, n

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    layers = sorted({k[0] for k in pooled})
    palette = {layer: CATEGORICAL[(7, 0, 2)[i % 3]] for i, layer in enumerate(layers)}
    for layer in layers:
        keys = sorted(k for k in pooled if k[0] == layer and k[1] == "refusal")
        xs = [k[2] for k in keys]
        axes[0].plot(xs, [rate(pooled[k])[0] for k in keys], marker="o", color=palette[layer], label=f"layer {layer} direction")
        nll = [statistics_mean([g.get("mean_nll_judge") for g in pooled[k]]) for k in keys]
        if all(v is not None for v in nll):
            axes[1].plot(xs, nll, marker="o", color=palette[layer], label=f"layer {layer} direction")
        controls = [k for k in pooled if k[0] == layer and k[1] == "random"]
        if controls:
            values = [rate(pooled[k])[0] for k in controls]
            axes[0].scatter([1.0] * len(values), values, marker="x", color=palette[layer], s=40, zorder=3,
                            label=f"random vector, layer {layer}")
    axes[0].set_xlabel("added vector, as a multiple of the class-mean difference")
    axes[0].set_ylabel("refusal on harmless prompts (%)")
    axes[0].set_ylim(0, 105)
    axes[0].set_title("Adding the direction induces refusal")
    axes[0].legend(loc="upper left")
    axes[1].set_xlabel("added vector, as a multiple of the class-mean difference")
    axes[1].set_ylabel("judge NLL per token (nats)")
    axes[1].set_title("Fluency of what the model says instead")
    for ax in axes:
        _despine(ax)
    fig.tight_layout()
    return _finish(fig, path, subtitle)


def statistics_mean(values: Sequence[float | None]) -> float | None:
    usable = [v for v in values if v is not None]
    return sum(usable) / len(usable) if usable else None


# --------------------------------------------------------------------------------------
# Phases 7 and 8: decode attention, batching and the roofline
# --------------------------------------------------------------------------------------

BACKEND_COLORS = {
    "sdpa_no_gqa": CATEGORICAL[1],
    "sdpa_grouped_decode": CATEGORICAL[2],
}
BACKEND_LABELS = {
    "sdpa_no_gqa": "sdpa_no_gqa (repeat_kv + SDPA)",
    "sdpa_grouped_decode": "sdpa_grouped_decode (this repo)",
}


def decode_backend_ab(summary: Sequence[Mapping[str, Any]], roofline: Sequence[Mapping[str, Any]],
                      backends: Sequence[str], path: Path, subtitle: str = "") -> Path | None:
    """Decode speed of each backend against its roofline, and the paired speed-up."""
    rows = sorted((s for s in summary if s.get("batch_size") == 1), key=lambda s: s["context_length"])
    if not rows or len(backends) < 2:
        return None
    x = [s["context_length"] for s in rows]
    fig, axes = plt.subplots(2, 1, figsize=(7.4, 7.2), sharex=True, gridspec_kw={"height_ratios": [3, 2]})

    by_backend = {b: {r["context_length"]: r for r in roofline if r["attn_implementation"] == b and r["batch_size"] == 1}
                  for b in backends}
    ideal = [by_backend[backends[0]].get(c, {}).get("ceiling_ideal_tok_s") for c in x]
    if all(v is not None for v in ideal):
        axes[0].plot(x, ideal, color=TEXT_SECONDARY, linestyle=":", linewidth=1.2, label="roofline: weights + KV read once")
    for backend in backends:
        color = BACKEND_COLORS.get(backend, NEUTRAL_MARK)
        med = [s.get(f"{backend}_decode_tok_s_median") for s in rows]
        lo = [m - s.get(f"{backend}_decode_tok_s_min", m) for m, s in zip(med, rows)]
        hi = [s.get(f"{backend}_decode_tok_s_max", m) - m for m, s in zip(med, rows)]
        axes[0].errorbar(x, med, yerr=[lo, hi], marker="o", color=color, capsize=2.5,
                         label=BACKEND_LABELS.get(backend, backend))
        model_ceiling = [by_backend[backend].get(c, {}).get(
            f"ceiling_{by_backend[backend].get(c, {}).get('traffic_model', 'ideal')}_tok_s") for c in x]
        if all(v is not None for v in model_ceiling):
            axes[0].plot(x, model_ceiling, color=color, linestyle="--", linewidth=1.0, alpha=0.7)
    from matplotlib.lines import Line2D

    axes[0].set_ylabel("decode tokens / second")
    axes[0].set_ylim(bottom=0)
    axes[0].set_title("Single-stream decode against the bandwidth roofline")
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(Line2D([], [], color=TEXT_SECONDARY, linestyle="--", linewidth=1.0))
    labels.append("roofline for each backend's modelled traffic (its colour)")
    axes[0].legend(handles, labels, loc="lower left", fontsize=8.5)

    other, base = backends[1], backends[0]
    key = f"{other}_vs_{base}_ratio"
    med = [s.get(f"{key}_median") for s in rows]
    if all(v is not None for v in med):
        lo = [s[f"{key}_min"] for s in rows]
        hi = [s[f"{key}_max"] for s in rows]
        axes[1].fill_between(x, lo, hi, color=BACKEND_COLORS.get(other, NEUTRAL_MARK), alpha=0.18, linewidth=0)
        axes[1].plot(x, med, marker="o", color=BACKEND_COLORS.get(other, NEUTRAL_MARK))
        axes[1].axhline(1.0, color=TEXT_SECONDARY, linestyle=":", linewidth=1.0)
        for xi, m in zip(x, med):
            axes[1].annotate(f"{m:.2f}x", xy=(xi, m), xytext=(0, 6), textcoords="offset points",
                             ha="center", fontsize=8, color=TEXT_PRIMARY)
    axes[1].set_ylabel("paired speed-up")
    axes[1].set_title("Per-round paired ratio (median, band = min to max)")
    axes[1].set_xscale("log", base=2)
    axes[1].xaxis.set_major_formatter(_context_formatter())
    axes[1].set_xticks(x)
    axes[1].set_xlabel("prompt length (tokens)")
    for ax in axes:
        _despine(ax)
    fig.tight_layout()
    return _finish(fig, path, subtitle)


def batch_throughput(summary: Sequence[Mapping[str, Any]], roofline: Sequence[Mapping[str, Any]],
                     backends: Sequence[str], path: Path, subtitle: str = "") -> Path | None:
    """Aggregate throughput and per-sequence speed against batch size."""
    rows = sorted(summary, key=lambda s: s["batch_size"])
    if len({s["batch_size"] for s in rows}) < 2:
        return None
    x = [s["batch_size"] for s in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.4))
    paged_any = False
    for backend in backends:
        color = BACKEND_COLORS.get(backend, NEUTRAL_MARK)
        for ax, key in ((axes[0], "throughput_tok_s_median"), (axes[1], "decode_tok_s_median")):
            clean = [(s["batch_size"], s[f"{backend}_{key}"]) for s in rows
                     if s.get(f"{backend}_{key}") is not None and not s.get(f"{backend}_paging_suspected")]
            paged = [(s["batch_size"], s[f"{backend}_{key}"]) for s in rows
                     if s.get(f"{backend}_{key}") is not None and s.get(f"{backend}_paging_suspected")]
            if clean:
                ax.plot(*zip(*clean), marker="o", color=color, label=BACKEND_LABELS.get(backend, backend))
                if ax is axes[0]:
                    ax.annotate(f"{clean[-1][1]:,.0f}", xy=clean[-1], xytext=(7, 0), textcoords="offset points", va="center",
                                ha="left", fontsize=8.5, color=TEXT_PRIMARY)
            if paged:
                paged_any = True
                ax.scatter(*zip(*paged), marker="o", facecolors="none", edgecolors=color, s=40, zorder=3)
    ref = {r["batch_size"]: r for r in roofline if r["attn_implementation"] == backends[-1]}
    ideal = [(b, ref[b]["ceiling_ideal_tok_s"] * b) for b in x if b in ref]
    if ideal:
        axes[0].plot(*zip(*ideal), color=TEXT_SECONDARY, linestyle=":", linewidth=1.2,
                     label="bandwidth roofline (weights + KV)")
        compute = next(iter(ref.values()))["compute_ceiling_throughput_tok_s"]
        axes[0].axhline(compute, color=TEXT_SECONDARY, linestyle="--", linewidth=1.0)
        axes[0].annotate(f"compute roofline {compute:,.0f} tok/s", xy=(x[-1], compute), xytext=(0, -12),
                         textcoords="offset points", ha="right", fontsize=8, color=TEXT_SECONDARY)
    if paged_any:
        axes[1].annotate("hollow: VRAM full, driver paging - not a valid measurement",
                         xy=(0.02, 0.04), xycoords="axes fraction", fontsize=8, color=TEXT_SECONDARY)
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks(x)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}"))
        ax.set_xlabel("batch size (sequences)")
        _despine(ax)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("aggregate tokens / second")
    axes[0].set_title("Batching: throughput")
    axes[0].legend(loc="lower right", fontsize=8.5)
    axes[1].set_ylabel("tokens / second per sequence")
    axes[1].set_ylim(bottom=0)
    axes[1].set_title("Batching: what each user sees")
    fig.tight_layout()
    return _finish(fig, path, subtitle)


def prefill_roofline(roofline: Sequence[Mapping[str, Any]], path: Path, subtitle: str = "") -> Path | None:
    """Prefill model-FLOPs utilisation, and how much of the work is attention."""
    rows = sorted((r for r in roofline if r.get("prefill_mfu") is not None and r["batch_size"] == 1
                   and r["precision"] == "bf16"), key=lambda r: r["context_length"])
    by_ctx: dict[int, Mapping[str, Any]] = {}
    for r in rows:
        by_ctx.setdefault(r["context_length"], r)
    rows = [by_ctx[c] for c in sorted(by_ctx)]
    if len(rows) < 2:
        return None
    x = [r["context_length"] for r in rows]
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.0), sharex=True)
    axes[0].plot(x, [r["prefill_mfu"] * 100 for r in rows], marker="o", color=CATEGORICAL[0])
    for xi, r in zip(x, rows):
        axes[0].annotate(f"{r['prefill_mfu'] * 100:.0f}%", xy=(xi, r["prefill_mfu"] * 100), xytext=(0, 6),
                         textcoords="offset points", ha="center", fontsize=8, color=TEXT_PRIMARY)
    axes[0].set_ylim(0, 105)
    axes[0].set_ylabel("MFU (% of measured GEMM peak)")
    axes[0].set_title("Prefill: model-FLOPs utilisation against prompt length")
    axes[1].plot(x, [r["attention_share_of_prefill_flops"] * 100 for r in rows], marker="o", color=CATEGORICAL[3])
    axes[1].set_ylabel("attention share of FLOPs (%)")
    axes[1].set_xlabel("prompt length (tokens)")
    axes[1].set_title("Why it falls again: the quadratic term's share of the work")
    axes[1].set_xscale("log", base=2)
    axes[1].xaxis.set_major_formatter(_context_formatter())
    axes[1].set_xticks(x)
    for ax in axes:
        _despine(ax)
    fig.tight_layout()
    return _finish(fig, path, subtitle)


def cross_scale_layers(runs: Mapping[str, Sequence[Mapping[str, Any]]], path: Path,
                       subtitle: str = "") -> Path | None:
    """Held-out separation by relative depth, one line per model size."""
    if len(runs) < 2:
        return None
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for i, (label, layers) in enumerate(sorted(runs.items())):
        ordered = sorted(layers, key=lambda r: r["layer"])
        n = len(ordered)
        ax.plot([r["layer"] / (n - 1) for r in ordered], [r["test"]["cohens_d"] for r in ordered],
                marker="o", markersize=4, color=CATEGORICAL[(0, 7, 2)[i % 3]], label=label)
    ax.set_xlabel("relative depth (block index / last block)")
    ax.set_ylabel("held-out Cohen's d")
    ax.set_title("The refusal direction emerges at a similar relative depth")
    ax.legend(loc="upper left")
    _despine(ax)
    fig.tight_layout()
    return _finish(fig, path, subtitle)


# --------------------------------------------------------------------------------------
# Overview
# --------------------------------------------------------------------------------------


def results_at_a_glance(
    decode_summary: Sequence[Mapping[str, Any]],
    backends: Sequence[str],
    intervention: Mapping[str, Any],
    path: Path,
    subtitle: str = "",
) -> Path | None:
    """Three panels, one per headline result, for the top of the README."""
    rows = sorted((s for s in decode_summary if s.get("batch_size") == 1), key=lambda s: s["context_length"])
    sweep = sorted(intervention.get("layer_sweep") or [], key=lambda r: r["source_layer"])
    harmful = (intervention.get("pooled") or {}).get("harmful") or {}
    harmless = (intervention.get("pooled") or {}).get("harmless") or {}
    observed = (intervention.get("selection") or {}).get("observational_layer")
    if not rows or not sweep or not harmful or len(backends) < 2:
        return None

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.6))

    ax = axes[0]
    x = [s["context_length"] for s in rows]
    for backend in backends:
        ax.plot(x, [s[f"{backend}_decode_tok_s_median"] for s in rows], marker="o",
                color=BACKEND_COLORS.get(backend, NEUTRAL_MARK), label=BACKEND_LABELS.get(backend, backend))
    last = rows[-1]
    ratio = last.get(f"{backends[1]}_vs_{backends[0]}_ratio_median")
    if ratio:
        top = last[f"{backends[1]}_decode_tok_s_median"]
        bottom = last[f"{backends[0]}_decode_tok_s_median"]
        ax.annotate("", xy=(x[-1] * 1.12, top), xytext=(x[-1] * 1.12, bottom),
                    arrowprops={"arrowstyle": "<->", "color": TEXT_PRIMARY, "linewidth": 1.2})
        ax.annotate(f"{ratio:.2f}x", xy=(x[-1] * 1.12, (top + bottom) / 2), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=11, fontweight="semibold", color=TEXT_PRIMARY)
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(_context_formatter())
    ax.set_xticks(x)
    ax.set_xlim(x[0] / 1.4, x[-1] * 2.0)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("prompt length (tokens)")
    ax.set_ylabel("decode tokens / second")
    ax.set_title("1. A faster decode path")
    ax.legend(loc="lower left", fontsize=8.5)

    ax = axes[1]
    layers = [r["source_layer"] for r in sweep]
    ax.fill_between(layers, [r["harmful_refusal_ci_low"] * 100 for r in sweep],
                    [r["harmful_refusal_ci_high"] * 100 for r in sweep], color=INTERVENTION_COLOR, alpha=0.15, linewidth=0)
    ax.plot(layers, [r["harmful_refusal_rate"] * 100 for r in sweep], marker="o", color=INTERVENTION_COLOR,
            label="refusal direction from that layer")
    randoms = [v["refusal_rate"] * 100 for k, v in harmful.items() if k.startswith("ablate:random")]
    if randoms:
        ax.axhline(sum(randoms) / len(randoms), color=CONTROL_COLOR, linestyle="--", linewidth=1.5,
                   label="random directions (controls)")
    ax.set_ylim(-3, 105)
    ax.set_xlabel("layer the ablated direction came from")
    ax.set_ylabel("refusal on harmful prompts (%)")
    ax.set_title("2. Removing one direction stops refusal")
    ax.legend(loc="lower left", fontsize=8.5)

    ax = axes[2]
    doses = sorted((float(k.split("x")[-1]), v) for k, v in harmless.items() if k.startswith(f"add:L{observed}x"))
    if doses:
        ax.plot([0.0] + [d for d, _ in doses], [harmless["baseline"]["refusal_rate"] * 100] +
                [v["refusal_rate"] * 100 for _, v in doses], marker="o", color=INTERVENTION_COLOR,
                label=f"layer {observed} direction")
    controls = [v["refusal_rate"] * 100 for k, v in harmless.items() if k.startswith(f"add:L{observed}random")]
    if controls:
        ax.scatter([1.0] * len(controls), controls, marker="x", s=55, color=CONTROL_COLOR, zorder=3,
                   label="random vectors, same norm")
    ax.set_ylim(-3, 105)
    ax.set_xlabel("added vector (x class-mean difference)")
    ax.set_ylabel("refusal on harmless prompts (%)")
    ax.set_title("3. Adding it makes the model refuse")
    ax.legend(loc="upper left", fontsize=8.5)

    for ax in axes:
        _despine(ax)
    fig.suptitle("Qwen2.5-7B-Instruct on one RTX 4090", x=0.0, ha="left", fontsize=13,
                 fontweight="semibold", color=TEXT_PRIMARY)
    fig.tight_layout()
    return _finish(fig, path, subtitle)
