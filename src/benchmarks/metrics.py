"""Result records for the benchmark harness.

Separating the record types from the measurement code keeps the definition of "what a
result is" in one readable place, and makes the shape of ``metrics.json`` reviewable
without reading the timing logic.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any, Sequence


def _percentile(values: Sequence[float], q: float) -> float | None:
    """Return the ``q``-th percentile (0-100) using linear interpolation.

    ``statistics.quantiles`` needs at least two data points and returns cut points
    rather than an arbitrary percentile, so the calculation is done directly here.
    """
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * (q / 100.0)
    lower = int(pos)
    upper = min(lower + 1, len(ordered) - 1)
    frac = pos - lower
    return ordered[lower] * (1 - frac) + ordered[upper] * frac


@dataclass
class GenerationTiming:
    """Timings from a single generation call."""

    prompt_tokens: int
    new_tokens: int
    prefill_latency_s: float | None
    ttft_s: float
    total_generation_s: float
    decode_time_s: float
    inter_token_latency_ms: list[float] = field(default_factory=list)

    @property
    def prefill_tokens_per_s(self) -> float | None:
        """Prompt tokens processed per second in the dedicated prefill pass."""
        if not self.prefill_latency_s:
            return None
        return self.prompt_tokens / self.prefill_latency_s

    @property
    def decode_tokens_per_s(self) -> float | None:
        """Decode throughput, excluding prefill.

        Measured across the interval between the first and last generated token, so the
        prompt-processing cost cannot inflate it.
        """
        if self.new_tokens < 2 or self.decode_time_s <= 0:
            return None
        return (self.new_tokens - 1) / self.decode_time_s

    @property
    def end_to_end_tokens_per_s(self) -> float | None:
        """Generated tokens per second including prefill - what a user experiences."""
        if self.total_generation_s <= 0:
            return None
        return self.new_tokens / self.total_generation_s

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view, percentiles included."""
        itl = self.inter_token_latency_ms
        return {
            "prompt_tokens": self.prompt_tokens,
            "new_tokens": self.new_tokens,
            "prefill_latency_s": _round(self.prefill_latency_s, 5),
            "prefill_tokens_per_s": _round(self.prefill_tokens_per_s, 2),
            "ttft_s": _round(self.ttft_s, 5),
            "total_generation_s": _round(self.total_generation_s, 5),
            "decode_time_s": _round(self.decode_time_s, 5),
            "decode_tokens_per_s": _round(self.decode_tokens_per_s, 3),
            "end_to_end_tokens_per_s": _round(self.end_to_end_tokens_per_s, 3),
            "itl_mean_ms": _round(statistics.fmean(itl), 3) if itl else None,
            "itl_p50_ms": _round(_percentile(itl, 50), 3),
            "itl_p90_ms": _round(_percentile(itl, 90), 3),
            "itl_p99_ms": _round(_percentile(itl, 99), 3),
            "itl_max_ms": _round(max(itl), 3) if itl else None,
        }


@dataclass
class BenchmarkCell:
    """One cell of the sweep grid: a (precision, context length) pair and its repeats."""

    precision: str
    context_length: int
    max_new_tokens: int
    batch_size: int
    status: str = "ok"
    reason: str | None = None
    repeats: list[dict[str, Any]] = field(default_factory=list)
    load: dict[str, Any] = field(default_factory=dict)
    telemetry: dict[str, Any] = field(default_factory=dict)
    failure: dict[str, Any] = field(default_factory=dict)

    def aggregate(self) -> dict[str, Any]:
        """Summarise the repeats with median and spread.

        Median rather than mean: a single scheduler hiccup or a background process
        waking up should not move the headline number.
        """
        if not self.repeats:
            return {}

        numeric_keys = [
            "prefill_latency_s",
            "prefill_tokens_per_s",
            "ttft_s",
            "total_generation_s",
            "decode_time_s",
            "decode_tokens_per_s",
            "end_to_end_tokens_per_s",
            "itl_mean_ms",
            "itl_p50_ms",
            "itl_p90_ms",
            "itl_p99_ms",
        ]
        summary: dict[str, Any] = {"n_repeats": len(self.repeats)}
        for key in numeric_keys:
            values = [r[key] for r in self.repeats if r.get(key) is not None]
            if not values:
                summary[f"{key}_median"] = None
                summary[f"{key}_stdev"] = None
                continue
            summary[f"{key}_median"] = _round(statistics.median(values), 5)
            summary[f"{key}_stdev"] = (
                _round(statistics.stdev(values), 5) if len(values) > 1 else 0.0
            )
        return summary

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view of the cell."""
        return {
            "precision": self.precision,
            "context_length": self.context_length,
            "max_new_tokens": self.max_new_tokens,
            "batch_size": self.batch_size,
            "status": self.status,
            "reason": self.reason,
            "load": self.load,
            "telemetry": self.telemetry,
            "failure": self.failure,
            "summary": self.aggregate(),
            "repeats": self.repeats,
        }

    def to_row(self) -> dict[str, Any]:
        """Flatten the cell into one CSV row for downstream plotting."""
        row: dict[str, Any] = {
            "precision": self.precision,
            "context_length": self.context_length,
            "max_new_tokens": self.max_new_tokens,
            "batch_size": self.batch_size,
            "status": self.status,
            "reason": self.reason,
        }
        row.update({f"load_{k}": v for k, v in self.load.items() if not isinstance(v, dict)})
        row.update(
            {f"gpu_{k}": v for k, v in self.telemetry.items() if not isinstance(v, (dict, list))}
        )
        row.update(self.aggregate())
        return row


def _round(value: float | None, digits: int) -> float | None:
    """Round when there is something to round."""
    return None if value is None else round(value, digits)
