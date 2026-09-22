"""Roofline arithmetic, pinned against numbers that can be checked by hand."""

from __future__ import annotations

import pytest

from src.analysis.roofline import (
    Architecture,
    analyse_cell,
    decode_bytes,
    mean_kv_tokens,
    prefill_flops,
)

QWEN_7B = {
    "num_hidden_layers": 28,
    "hidden_size": 3584,
    "intermediate_size": 18944,
    "num_attention_heads": 28,
    "num_key_value_heads": 4,
    "vocab_size": 152064,
    "tie_word_embeddings": False,
}


@pytest.fixture
def arch() -> Architecture:
    return Architecture.from_config(QWEN_7B)


def test_parameter_count_reconciles_with_the_checkpoint(arch: Architecture) -> None:
    # Matmul params + untied input and output embeddings should account for the
    # recorded 7,615,616,512 to within the q/k/v biases and norm weights (<0.01%).
    total = arch.matmul_params() + 2 * arch.vocab_size * arch.hidden_size
    assert total == pytest.approx(7_615_616_512, rel=1e-4)
    assert total < 7_615_616_512


def test_kv_cache_size(arch: Architecture) -> None:
    # 2 (K,V) x 28 layers x 4 heads x 128 dims x 2 bytes
    assert arch.kv_bytes_per_token() == 57344
    assert arch.groups == 7


def test_traffic_models_are_ordered(arch: Architecture) -> None:
    w = 15e9
    ideal = decode_bytes(arch, w, 16384, "ideal")
    grouped = decode_bytes(arch, w, 16384, "grouped")
    expanded = decode_bytes(arch, w, 16384, "expanded")
    kv = 57344 * 16384
    assert ideal == w + kv
    assert grouped == w + 3 * kv
    assert expanded == w + 3 * kv + 14 * kv
    with pytest.raises(ValueError):
        decode_bytes(arch, w, 1, "magic")


def test_mean_kv_tokens() -> None:
    assert mean_kv_tokens(128, 1) == 128
    assert mean_kv_tokens(128, 129) == 192


def test_prefill_flops_is_linear_plus_quadratic(arch: Architecture) -> None:
    linear = 2 * arch.matmul_params()
    one = prefill_flops(arch, 1)
    assert one == pytest.approx(linear + 2 * 28 * 3584)
    # Doubling T more than doubles the work once attention matters.
    assert prefill_flops(arch, 16384) > 2 * prefill_flops(arch, 8192)
    assert prefill_flops(arch, 8, causal=False) > prefill_flops(arch, 8, causal=True)


def test_analyse_cell_uses_the_applicable_model(arch: Architecture) -> None:
    ceilings = {"decode_bandwidth_gb_s": 900.0, "gemm_tflops": 150.0}
    cell = {
        "run": "x/1",
        "precision": "bf16",
        "attn_implementation": "sdpa_no_gqa",
        "config": QWEN_7B,
        "weights_mib": 14525.6,
        "context_length": 128,
        "new_tokens": 128,
        "decode_tok_s": 44.8,
        "decode_tok_s_stdev": 0.0,
        "prefill_tok_s": 5000.0,
        "prefill_s": 0.0256,
    }
    row = analyse_cell(cell, ceilings)
    assert row["traffic_model"] == "expanded"
    # 15.23 GB of weights at 900 GB/s: the ideal ceiling is ~59 tok/s.
    assert row["ceiling_ideal_tok_s"] == pytest.approx(58.8, abs=0.3)
    assert 0 < row["fraction_of_applicable"] < 1
    assert row["ceiling_expanded_tok_s"] < row["ceiling_grouped_tok_s"] < row["ceiling_ideal_tok_s"]

    grouped = analyse_cell({**cell, "attn_implementation": "sdpa_grouped_decode"}, ceilings)
    assert grouped["traffic_model"] == "grouped"


def test_batching_shares_the_weight_read(arch: Architecture) -> None:
    w = 15e9
    one = decode_bytes(arch, w, 512, "ideal", batch_size=1)
    eight = decode_bytes(arch, w, 512, "ideal", batch_size=8)
    assert eight - w == pytest.approx(8 * (one - w))
    # Per-sequence ceilings fall only slightly, so aggregate throughput rises ~8x.
    ceilings = {"decode_bandwidth_gb_s": 900.0, "gemm_tflops": 150.0}
    base = {
        "run": "x/1", "precision": "bf16", "attn_implementation": "sdpa_grouped_decode",
        "config": QWEN_7B, "weights_mib": 14525.6, "context_length": 512, "new_tokens": 128,
        "decode_tok_s": 40.0, "prefill_s": 0.06,
    }
    r1 = analyse_cell({**base, "batch_size": 1}, ceilings)
    r8 = analyse_cell({**base, "batch_size": 8}, ceilings)
    assert r8["ceiling_ideal_tok_s"] < r1["ceiling_ideal_tok_s"]
    assert 8 * r8["ceiling_ideal_tok_s"] > 7 * r1["ceiling_ideal_tok_s"]
    assert r8["measured_throughput_tok_s"] == 320.0
    # 150 TFLOP/s over 2 x 6.53e9 FLOPs per token is ~11.5k tokens/s.
    assert r1["compute_ceiling_throughput_tok_s"] == pytest.approx(11494, rel=0.01)
