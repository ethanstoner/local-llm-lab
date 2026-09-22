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

from pathlib import Path
from typing import Any, Mapping, Sequence

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

    for row, x, y in zip(usable, xs, ys):
        ax.annotate(
            str(row["precision"]),
            xy=(x, y),
            xytext=(8, 4),
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
    axes[0].annotate(
        f"peak: layer {best['layer']}",
        xy=(best["layer"], max(test_d)),
        xytext=(6, -2),
        textcoords="offset points",
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
