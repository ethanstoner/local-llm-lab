"""Attention-backend selection, and a fix for a platform-specific performance trap.

## The trap

PyTorch's scaled dot-product attention dispatches to one of several kernels. Two matter
here: the *memory-efficient* kernel, whose memory is linear in sequence length, and the
*math* kernel, which materialises the full ``(heads, seq, seq)`` attention matrix and is
therefore quadratic.

Transformers 4.5x asks PyTorch to handle grouped-query attention natively by passing
``enable_gqa=True``, which it does whenever there is no attention mask. Its own comment
explains the reasoning: passing a mask "will fall back to the math kernel", so avoiding
one is supposed to keep attention on a fast path.

That reasoning assumes the flash-attention kernel exists. PyTorch's Windows wheels are
**not compiled with flash attention** (``UserWarning: Torch was not compiled with flash
attention``), and the memory-efficient kernel does not implement ``enable_gqa`` at all.
So on Windows the two fast kernels are both unavailable and SDPA silently falls back to
math - the exact outcome the flag was meant to avoid.

Measured on this machine with Qwen2.5-7B-Instruct at bf16, prompt processing only:

| context | activations (stock sdpa) | prefill |
|---------|--------------------------|---------|
| 2048    | 1322 MiB                 | 0.61 s  |
| 4096    | 4682 MiB                 | 1.27 s  |
| 8192    | 17549 MiB                | 142 s   |

At 8192 the allocator's peak reached 32078 MiB on a 24564 MiB card. It did not raise
OOM: the Windows display driver pages VRAM to host memory instead, so the run survives
and merely becomes about two hundred times slower. A benchmark that did not check for
this would publish that number as the model's prefill latency.

Isolating the cause, at 4096 tokens with this model's head configuration:

```
enable_gqa=True   mem_efficient: FAIL  No available kernel
enable_gqa=True   math         : OK    4344 MiB
enable_gqa=False  mem_efficient: OK      28 MiB
enable_gqa=False  math         : OK    4320 MiB
```

## The fix

``sdpa_no_gqa`` is the stock SDPA path with one change: key and value heads are expanded
with ``repeat_kv`` and ``enable_gqa`` is never passed. That costs a modest amount of
memory for the expanded KV tensors and lets the memory-efficient kernel run, which is
worth several orders of magnitude at long context.

This is a workaround for a platform limitation, not an improvement on transformers. On a
build that has flash attention, stock ``sdpa`` is the better choice, which is why
:func:`recommended_attn_implementation` probes the build rather than assuming.

## A second trap: custom names get no padding mask

Transformers builds the attention mask through a *separate* registry from the attention
function itself. For any implementation name missing from that registry,
``masking_utils`` concludes that the backend "doesn't need a mask" and passes ``None``.
Registering only the attention function therefore silently drops the padding mask: in
a left-padded batch every real token attends to the pad tokens. Batch-size-1 runs are
unaffected, which is why the benchmarks never showed it; every batched forward pass
was. :func:`register_attention_backends` now registers the stock SDPA mask builder
under each custom name, and ``tests/test_attention.py`` checks a padded batch against
the same prompt run alone.

## Decode: read the KV cache once

``sdpa_no_gqa`` fixes prefill but is wasteful for decoding. With one query token, the
memory-efficient kernel parallelises over batch x heads - 28 thread blocks at batch 1 on
a 128-SM card - and ``repeat_kv`` first copies the whole cache ``groups`` times per
layer per token. At 16k context that is ~6.6 GB of extra traffic per generated token.

``sdpa_grouped_decode`` keeps the ``sdpa_no_gqa`` prefill and replaces the decode step
with two grouped matmuls: the query heads that share a KV head are stacked as rows of
one matrix, so each key and value is read exactly once, with no expansion and with
cuBLAS tiling over the sequence dimension. The arithmetic follows transformers' eager
attention (bf16 scores, float32 softmax).
"""

from __future__ import annotations

import warnings
from functools import lru_cache
from typing import Any, Optional

import torch

from src.utils.logging import get_logger

logger = get_logger(__name__)

#: Name this module registers with transformers' attention interface.
SDPA_NO_GQA = "sdpa_no_gqa"


@lru_cache(maxsize=1)
def flash_attention_available() -> bool:
    """Whether this torch build has a usable flash-attention kernel.

    Probed by running a tiny attention under the flash backend rather than by checking
    flags: ``torch.backends.cuda.flash_sdp_enabled()`` reports True on Windows even
    though the kernel was never compiled in.
    """
    if not torch.cuda.is_available():
        return False
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        q = torch.zeros(1, 2, 8, 64, dtype=torch.bfloat16, device="cuda")
        with warnings.catch_warnings():
            # Probing for an absent kernel is the point; PyTorch narrating each
            # rejection would otherwise head every run's log.
            warnings.simplefilter("ignore", UserWarning)
            with torch.inference_mode(), sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                torch.nn.functional.scaled_dot_product_attention(q, q, q, is_causal=True)
        return True
    except Exception as exc:
        logger.debug("flash attention unavailable: %s", exc)
        return False


@lru_cache(maxsize=1)
def memory_efficient_available() -> bool:
    """Whether this torch build has a usable memory-efficient attention kernel."""
    if not torch.cuda.is_available():
        return False
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        q = torch.zeros(1, 2, 8, 64, dtype=torch.bfloat16, device="cuda")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with torch.inference_mode(), sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
                torch.nn.functional.scaled_dot_product_attention(q, q, q, is_causal=True)
        return True
    except Exception as exc:
        logger.debug("memory-efficient attention unavailable: %s", exc)
        return False


def sdpa_no_gqa_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    is_causal: Optional[bool] = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    """SDPA attention that expands KV heads instead of using ``enable_gqa``.

    Mirrors ``transformers.integrations.sdpa_attention.sdpa_attention_forward``, minus
    the ``enable_gqa`` branch. See the module docstring for why that one difference
    matters by two orders of magnitude on this platform.
    """
    from transformers.integrations.sdpa_attention import repeat_kv

    if kwargs.get("output_attentions", False) or kwargs.get("head_mask") is not None:
        logger.warning(
            "sdpa_no_gqa does not support output_attentions or head_mask; use eager."
        )

    if hasattr(module, "num_key_value_groups"):
        key = repeat_kv(key, module.num_key_value_groups)
        value = repeat_kv(value, module.num_key_value_groups)

    if attention_mask is not None and attention_mask.ndim == 4:
        attention_mask = attention_mask[:, :, :, : key.shape[-2]]

    if is_causal is None:
        is_causal = query.shape[2] > 1 and attention_mask is None and getattr(module, "is_causal", True)

    if torch.jit.is_tracing() and isinstance(is_causal, torch.Tensor):
        is_causal = is_causal.item()

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attention_mask,
        dropout_p=dropout,
        scale=scaling,
        is_causal=is_causal,
    )
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, None


#: Name for the variant with a grouped, expansion-free decode step.
SDPA_GROUPED_DECODE = "sdpa_grouped_decode"

#: Every custom implementation this module registers.
CUSTOM_IMPLEMENTATIONS = (SDPA_NO_GQA, SDPA_GROUPED_DECODE)


def grouped_decode_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: Optional[float] = None,
) -> torch.Tensor:
    """Single-query attention without expanding the KV heads.

    Args:
        query: ``(batch, heads, 1, dim)``.
        key: ``(batch, kv_heads, kv_len, dim)``.
        value: ``(batch, kv_heads, kv_len, dim)``.
        attention_mask: ``None``, a boolean ``(batch, 1, 1, kv_len)`` mask where True
            means attend, or an additive float mask of the same shape.
        scaling: Softmax temperature; ``1/sqrt(dim)`` when omitted.

    Returns:
        ``(batch, 1, heads, dim)``, the layout transformers expects back.

    Heads are grouped the way ``repeat_kv`` expands them - query head ``h`` reads KV head
    ``h // groups`` - so reshaping ``(heads,)`` to ``(kv_heads, groups)`` pairs each
    query with its own key and value.
    """
    batch, heads, q_len, dim = query.shape
    kv_heads = key.shape[1]
    if q_len != 1:
        raise ValueError(f"grouped decode needs one query position, got {q_len}")
    if heads % kv_heads:
        raise ValueError(f"{heads} query heads do not divide into {kv_heads} KV heads")
    groups = heads // kv_heads
    scale = scaling if scaling is not None else dim**-0.5

    q = query.reshape(batch, kv_heads, groups, dim)
    scores = torch.matmul(q, key.transpose(-1, -2)) * scale
    if attention_mask is not None:
        mask = attention_mask[..., : key.shape[-2]]
        if mask.dtype == torch.bool:
            scores = scores.masked_fill(~mask, float("-inf"))
        else:
            scores = scores + mask
    probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(value.dtype)
    out = torch.matmul(probs, value)
    return out.reshape(batch, heads, 1, dim).transpose(1, 2).contiguous()


def sdpa_grouped_decode_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    is_causal: Optional[bool] = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    """``sdpa_no_gqa`` for prefill, :func:`grouped_decode_attention` for decode steps."""
    if query.shape[2] == 1 and dropout == 0.0:
        return grouped_decode_attention(query, key, value, attention_mask, scaling), None
    return sdpa_no_gqa_attention_forward(
        module, query, key, value, attention_mask,
        dropout=dropout, scaling=scaling, is_causal=is_causal, **kwargs,
    )


def register_attention_backends() -> list[str]:
    """Register this project's custom attention implementations with transformers.

    Two registries are updated, and both matter. The attention-function registry makes
    the name loadable; the mask registry makes transformers build a padding mask for
    it. Without the second, ``masking_utils`` passes ``None`` for any unknown name and
    padded batches silently attend to their pad tokens.

    Idempotent, so it is safe to call from every entry point.

    Returns:
        The names now available to ``attn_implementation``.
    """
    from transformers.masking_utils import AttentionMaskInterface, sdpa_mask
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    functions = {
        SDPA_NO_GQA: sdpa_no_gqa_attention_forward,
        SDPA_GROUPED_DECODE: sdpa_grouped_decode_attention_forward,
    }
    for name, fn in functions.items():
        if name not in ALL_ATTENTION_FUNCTIONS.valid_keys():
            ALL_ATTENTION_FUNCTIONS[name] = fn
            logger.debug("Registered attention implementation %r", name)
        # The mask builder must go into the class-level mapping: masking_utils checks
        # ``_global_mapping`` directly, so an instance-level assignment is invisible.
        if name not in AttentionMaskInterface._global_mapping:
            AttentionMaskInterface.register(name, sdpa_mask)
    return list(ALL_ATTENTION_FUNCTIONS.valid_keys())


def recommended_attn_implementation() -> str:
    """Pick the attention implementation that suits this build.

    ``sdpa`` when a flash kernel exists, ``sdpa_no_gqa`` when it does not but the
    memory-efficient kernel does, and ``sdpa`` as a last resort so behaviour matches
    stock transformers on an unknown platform.
    """
    if flash_attention_available():
        return "sdpa"
    if memory_efficient_available():
        return SDPA_NO_GQA
    return "sdpa"


def backend_report() -> dict[str, Any]:
    """Describe the available attention kernels, for run metadata.

    Worth recording per run: the same config on a Linux build would take a different
    kernel and produce different long-context numbers.
    """
    return {
        "flash_kernel_available": flash_attention_available(),
        "memory_efficient_kernel_available": memory_efficient_available(),
        "recommended_attn_implementation": recommended_attn_implementation(),
        "torch_version": torch.__version__,
    }
