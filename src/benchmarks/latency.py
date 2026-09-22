"""Timing primitives for generation benchmarks.

Three properties make these numbers trustworthy, and all three are easy to get wrong:

1. CUDA work is asynchronous, so every timed region ends with an explicit
   ``torch.cuda.synchronize``. A timer stopped after the launch measures enqueue time.
2. Prefill and decode are different regimes and are measured separately. A single
   tokens-per-second figure averages a compute-bound phase with a bandwidth-bound one.
3. Generation is forced to emit exactly ``max_new_tokens`` tokens. A run that stops
   early at an end-of-sequence marker produces fewer tokens in less time, and the ratio
   flatters whichever configuration happened to stop soonest.
"""

from __future__ import annotations

import time
from typing import Any

import torch
from transformers import LogitsProcessor, LogitsProcessorList

from src.benchmarks.metrics import GenerationTiming
from src.utils.logging import get_logger

logger = get_logger(__name__)


class TokenTimestampProcessor(LogitsProcessor):
    """Records a timestamp as each token's logits become available.

    Transformers calls the logits processors once per generated token, immediately after
    the forward pass that produced those logits. Stamping the clock there gives time to
    first token and the full inter-token latency distribution from a single generation
    call, with no second pass and no streaming thread.

    Because CUDA is asynchronous, the processor synchronizes before stamping - otherwise
    it would record when the work was queued rather than when it completed. That costs a
    small amount of CPU/GPU overlap; :func:`measure_sync_overhead` quantifies it so the
    cost is documented rather than assumed negligible.
    """

    def __init__(self, device: torch.device | str | None = None, synchronize: bool = True) -> None:
        """
        Args:
            device: Device to synchronize on. Ignored when ``synchronize`` is False.
            synchronize: Whether to wait for the GPU before stamping.
        """
        self.timestamps: list[float] = []
        self._device = device
        self._synchronize = synchronize and torch.cuda.is_available()

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self._synchronize:
            torch.cuda.synchronize(self._device)
        self.timestamps.append(time.perf_counter())
        return scores

    def reset(self) -> None:
        """Clear recorded timestamps so the processor can be reused."""
        self.timestamps.clear()


def sync(device: torch.device | str | None = None) -> None:
    """Block until all queued CUDA work on ``device`` has completed."""
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)


@torch.inference_mode()
def measure_prefill(
    model: Any,
    inputs: dict[str, torch.Tensor],
    device: torch.device | str | None = None,
) -> float:
    """Time a single forward pass over the prompt.

    This is prompt processing in isolation: the cost of filling the KV cache before any
    token is generated. It is measured with its own call rather than inferred from time
    to first token, and the two are reported side by side as a cross-check.

    Returns:
        Elapsed seconds.
    """
    sync(device)
    start = time.perf_counter()
    model(**inputs, use_cache=True)
    sync(device)
    return time.perf_counter() - start


@torch.inference_mode()
def measure_generation(
    model: Any,
    tokenizer: Any,
    inputs: dict[str, torch.Tensor],
    max_new_tokens: int,
    device: torch.device | str | None = None,
    measure_prefill_separately: bool = True,
    synchronize_per_token: bool = True,
) -> tuple[GenerationTiming, torch.Tensor]:
    """Run one generation and return its timings plus the generated token ids.

    Args:
        model: A causal LM in eval mode.
        tokenizer: Its tokenizer, used only for the pad token id.
        inputs: ``input_ids`` and ``attention_mask``, already on the target device.
        max_new_tokens: Exact number of tokens to generate. Enforced with
            ``min_new_tokens`` so an early end-of-sequence cannot shorten the run.
        device: Device to synchronize on.
        measure_prefill_separately: Run an extra forward pass to time prefill alone.
        synchronize_per_token: Wait for the GPU before each per-token timestamp.

    Returns:
        A tuple of the :class:`GenerationTiming` and the generated sequences.
    """
    prompt_tokens = int(inputs["input_ids"].shape[-1])

    prefill_latency = (
        measure_prefill(model, inputs, device) if measure_prefill_separately else None
    )

    stamper = TokenTimestampProcessor(device=device, synchronize=synchronize_per_token)
    processors = LogitsProcessorList([stamper])

    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        # Forces a fixed token budget; also suppresses EOS, so every run is comparable.
        "min_new_tokens": max_new_tokens,
        "do_sample": False,
        "num_beams": 1,
        "use_cache": True,
        "logits_processor": processors,
        "pad_token_id": tokenizer.pad_token_id,
        "return_dict_in_generate": False,
    }

    sync(device)
    start = time.perf_counter()
    sequences = model.generate(**inputs, **generation_kwargs)
    sync(device)
    total = time.perf_counter() - start

    timestamps = stamper.timestamps
    new_tokens = int(sequences.shape[-1] - prompt_tokens)

    if not timestamps:
        # Should not happen with a standard generate loop; recorded rather than guessed.
        logger.warning("No per-token timestamps captured; TTFT will fall back to total time")
        ttft = total
        decode_time = 0.0
        itl_ms: list[float] = []
    else:
        ttft = timestamps[0] - start
        decode_time = timestamps[-1] - timestamps[0]
        itl_ms = [
            (timestamps[i] - timestamps[i - 1]) * 1000.0 for i in range(1, len(timestamps))
        ]

    timing = GenerationTiming(
        prompt_tokens=prompt_tokens,
        new_tokens=new_tokens,
        prefill_latency_s=prefill_latency,
        ttft_s=ttft,
        total_generation_s=total,
        decode_time_s=decode_time,
        inter_token_latency_ms=itl_ms,
    )
    return timing, sequences


@torch.inference_mode()
def measure_sync_overhead(
    model: Any,
    tokenizer: Any,
    inputs: dict[str, torch.Tensor],
    max_new_tokens: int,
    device: torch.device | str | None = None,
    repeats: int = 3,
) -> dict[str, Any]:
    """Quantify what per-token synchronization costs.

    The main harness stamps a synchronized clock once per generated token. That is the
    only way to get a real inter-token latency distribution, but it does give up some
    CPU/GPU overlap. This runs the same configuration with and without the per-token
    synchronization so the size of that effect is a measured quantity in the results
    rather than an assumption in a docstring.

    Returns:
        Median total generation seconds for both modes and the relative difference.
    """
    import statistics

    results: dict[str, list[float]] = {"synced": [], "unsynced": []}
    for mode, flag in (("synced", True), ("unsynced", False)):
        for _ in range(repeats):
            timing, _ = measure_generation(
                model,
                tokenizer,
                inputs,
                max_new_tokens,
                device=device,
                measure_prefill_separately=False,
                synchronize_per_token=flag,
            )
            results[mode].append(timing.total_generation_s)

    synced = statistics.median(results["synced"])
    unsynced = statistics.median(results["unsynced"])
    return {
        "repeats": repeats,
        "max_new_tokens": max_new_tokens,
        "prompt_tokens": int(inputs["input_ids"].shape[-1]),
        "median_total_s_with_per_token_sync": round(synced, 5),
        "median_total_s_without_per_token_sync": round(unsynced, 5),
        "relative_overhead_pct": round((synced - unsynced) / unsynced * 100.0, 3),
    }


@torch.inference_mode()
def warmup(
    model: Any,
    tokenizer: Any,
    inputs: dict[str, torch.Tensor],
    max_new_tokens: int,
    device: torch.device | str | None = None,
    iterations: int = 1,
) -> None:
    """Run and discard generations so kernel selection and cache growth are not measured."""
    for _ in range(max(iterations, 0)):
        measure_generation(
            model,
            tokenizer,
            inputs,
            max_new_tokens,
            device=device,
            measure_prefill_separately=False,
            synchronize_per_token=False,
        )
    sync(device)
