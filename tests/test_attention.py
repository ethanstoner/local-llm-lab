"""Attention backend probing and the no-GQA SDPA path.

The behaviour under test is the reason long-context benchmarking works at all on this
platform: transformers asks PyTorch for native grouped-query attention, PyTorch's
memory-efficient kernel does not implement it, and the Windows build has no flash
kernel, so SDPA silently falls back to the quadratic math kernel. See
``src/models/attention.py`` for the measurements.
"""

from __future__ import annotations

import pytest
import torch

from src.models.attention import (
    SDPA_NO_GQA,
    backend_report,
    flash_attention_available,
    memory_efficient_available,
    recommended_attn_implementation,
    register_attention_backends,
    sdpa_no_gqa_attention_forward,
)
from src.utils.config import VALID_ATTN_IMPLEMENTATIONS


class StubAttention(torch.nn.Module):
    """Stands in for a decoder block's attention module."""

    def __init__(self, groups: int) -> None:
        super().__init__()
        self.num_key_value_groups = groups
        self.is_causal = True


def test_registration_is_idempotent() -> None:
    first = register_attention_backends()
    second = register_attention_backends()
    assert SDPA_NO_GQA in first
    assert first == second


def test_registered_name_is_a_valid_config_value() -> None:
    """The registry and the config whitelist must not drift apart."""
    assert SDPA_NO_GQA in VALID_ATTN_IMPLEMENTATIONS


def test_probes_return_booleans() -> None:
    assert isinstance(flash_attention_available(), bool)
    assert isinstance(memory_efficient_available(), bool)


def test_recommendation_is_a_valid_implementation() -> None:
    assert recommended_attn_implementation() in VALID_ATTN_IMPLEMENTATIONS


def test_recommendation_follows_the_probes() -> None:
    """Prefer stock sdpa when flash exists; only work around it when it does not."""
    recommended = recommended_attn_implementation()
    if flash_attention_available():
        assert recommended == "sdpa"
    elif memory_efficient_available():
        assert recommended == SDPA_NO_GQA


def test_backend_report_shape() -> None:
    report = backend_report()
    assert set(report) == {
        "flash_kernel_available",
        "memory_efficient_kernel_available",
        "recommended_attn_implementation",
        "torch_version",
    }


@pytest.mark.parametrize("groups", [1, 7])
def test_no_gqa_matches_reference_attention(groups: int) -> None:
    """The workaround must compute the same thing, not merely run faster.

    Checked against an explicit softmax(QK^T/sqrt(d))V with the KV heads expanded by
    hand, on CPU so the test needs no GPU.
    """
    torch.manual_seed(0)
    kv_heads, ctx, dim = 2, 6, 8
    heads = kv_heads * groups

    query = torch.randn(1, heads, ctx, dim, dtype=torch.float32)
    key = torch.randn(1, kv_heads, ctx, dim, dtype=torch.float32)
    value = torch.randn(1, kv_heads, ctx, dim, dtype=torch.float32)

    out, weights = sdpa_no_gqa_attention_forward(
        StubAttention(groups), query, key, value, attention_mask=None
    )
    assert weights is None
    # The forward transposes to (batch, seq, heads, dim) as transformers expects.
    assert out.shape == (1, ctx, heads, dim)

    k_full = key.repeat_interleave(groups, dim=1)
    v_full = value.repeat_interleave(groups, dim=1)
    scores = (query @ k_full.transpose(-1, -2)) / (dim**0.5)
    causal = torch.triu(torch.full((ctx, ctx), float("-inf")), diagonal=1)
    expected = (torch.softmax(scores + causal, dim=-1) @ v_full).transpose(1, 2)

    assert torch.allclose(out, expected, atol=1e-5)


def test_no_gqa_honours_an_explicit_mask() -> None:
    """A supplied 4-D mask must still be applied, and trimmed to the key length."""
    torch.manual_seed(1)
    heads, ctx, dim = 4, 5, 8
    query = torch.randn(1, heads, ctx, dim)
    key = torch.randn(1, heads, ctx, dim)
    value = torch.randn(1, heads, ctx, dim)

    # Mask out the final key position entirely.
    mask = torch.zeros(1, 1, ctx, ctx)
    mask[..., -1] = float("-inf")

    out, _ = sdpa_no_gqa_attention_forward(
        StubAttention(1), query, key, value, attention_mask=mask, is_causal=False
    )

    k_full, v_full = key, value
    scores = (query @ k_full.transpose(-1, -2)) / (dim**0.5) + mask
    expected = (torch.softmax(scores, dim=-1) @ v_full).transpose(1, 2)
    assert torch.allclose(out, expected, atol=1e-5)


@pytest.mark.gpu
def test_no_gqa_keeps_attention_memory_linear() -> None:
    """Doubling the context must not quadruple activation memory.

    This is the regression that matters: with ``enable_gqa=True`` the math kernel runs
    and memory goes as the square of the sequence length, which on a 24 GB card turns a
    one-second prefill into a two-minute one once the driver starts paging.
    """
    if not torch.cuda.is_available() or not memory_efficient_available():
        pytest.skip("needs a CUDA device with the memory-efficient kernel")

    from src.monitoring.memory import empty_cache

    heads, kv_heads, dim = 8, 2, 64
    module = StubAttention(heads // kv_heads)
    peaks = {}

    for ctx in (1024, 2048):
        empty_cache()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        query = torch.randn(1, heads, ctx, dim, dtype=torch.bfloat16, device="cuda")
        key = torch.randn(1, kv_heads, ctx, dim, dtype=torch.bfloat16, device="cuda")
        value = torch.randn(1, kv_heads, ctx, dim, dtype=torch.bfloat16, device="cuda")
        with torch.inference_mode():
            sdpa_no_gqa_attention_forward(module, query, key, value, attention_mask=None)
        torch.cuda.synchronize()
        peaks[ctx] = torch.cuda.max_memory_allocated() - base
        del query, key, value

    # Linear growth would be ~2x. Quadratic would be ~4x. Allow generous headroom and
    # still catch a regression to the math kernel.
    ratio = peaks[2048] / max(peaks[1024], 1)
    assert ratio < 3.0, f"attention memory grew {ratio:.1f}x for a 2x context"
