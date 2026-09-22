"""Timing arithmetic.

These are the formulas that decide what the headline numbers say, so they are tested
against hand-computed values rather than against themselves.
"""

from __future__ import annotations

import math

import pytest

from src.benchmarks.metrics import BenchmarkCell, GenerationTiming, _percentile


def make_timing(**overrides) -> GenerationTiming:
    """Build a timing record with sane defaults."""
    defaults = dict(
        prompt_tokens=512,
        new_tokens=11,
        prefill_latency_s=0.25,
        ttft_s=0.3,
        total_generation_s=1.3,
        decode_time_s=1.0,
        inter_token_latency_ms=[100.0] * 10,
    )
    defaults.update(overrides)
    return GenerationTiming(**defaults)


def test_decode_rate_excludes_prefill() -> None:
    # 11 tokens, 10 gaps over 1.0 s of decoding => 10 tok/s, regardless of the 0.3 s TTFT.
    assert make_timing().decode_tokens_per_s == pytest.approx(10.0)


def test_end_to_end_rate_includes_prefill() -> None:
    # 11 tokens over the full 1.3 s.
    assert make_timing().end_to_end_tokens_per_s == pytest.approx(11 / 1.3)


def test_end_to_end_is_slower_than_decode() -> None:
    """The end-to-end rate must never exceed the decode rate; prefill is pure overhead."""
    timing = make_timing()
    assert timing.end_to_end_tokens_per_s < timing.decode_tokens_per_s


def test_prefill_rate() -> None:
    assert make_timing().prefill_tokens_per_s == pytest.approx(512 / 0.25)


def test_rates_are_none_when_undefined() -> None:
    assert make_timing(new_tokens=1, decode_time_s=0.0).decode_tokens_per_s is None
    assert make_timing(prefill_latency_s=None).prefill_tokens_per_s is None
    assert make_timing(total_generation_s=0.0).end_to_end_tokens_per_s is None


def test_percentile_interpolates() -> None:
    values = [0.0, 10.0, 20.0, 30.0]
    assert _percentile(values, 0) == 0.0
    assert _percentile(values, 100) == 30.0
    assert _percentile(values, 50) == pytest.approx(15.0)


def test_percentile_edge_cases() -> None:
    assert _percentile([], 50) is None
    assert _percentile([7.0], 99) == 7.0


def test_cell_aggregate_uses_median() -> None:
    cell = BenchmarkCell(precision="bf16", context_length=128, max_new_tokens=16, batch_size=1)
    # A single outlier must not move the reported figure.
    for value in (10.0, 11.0, 12.0, 200.0):
        cell.repeats.append(make_timing(decode_time_s=10 / value).to_dict())

    summary = cell.aggregate()
    assert summary["n_repeats"] == 4
    assert summary["decode_tokens_per_s_median"] == pytest.approx(11.5, abs=0.01)
    assert summary["decode_tokens_per_s_stdev"] > 0


def test_cell_aggregate_empty() -> None:
    cell = BenchmarkCell(precision="bf16", context_length=128, max_new_tokens=16, batch_size=1)
    assert cell.aggregate() == {}


def test_cell_row_is_flat() -> None:
    cell = BenchmarkCell(
        precision="nf4",
        context_length=512,
        max_new_tokens=16,
        batch_size=1,
        status="oom",
        reason="out of memory",
        load={"weights_mib": 5500.0, "allocator_after_load": {"nested": 1}},
        telemetry={"peak_memory_used_mib": 6000.0, "samples_csv": "telemetry/x.csv"},
    )
    row = cell.to_row()
    assert row["status"] == "oom"
    assert row["load_weights_mib"] == 5500.0
    assert row["gpu_peak_memory_used_mib"] == 6000.0
    assert all(not isinstance(v, dict) for v in row.values())


def test_timing_dict_has_no_nans() -> None:
    payload = make_timing(inter_token_latency_ms=[]).to_dict()
    assert payload["itl_p50_ms"] is None
    assert all(not (isinstance(v, float) and math.isnan(v)) for v in payload.values())
