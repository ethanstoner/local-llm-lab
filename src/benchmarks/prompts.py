"""Deterministic prompt construction for the benchmark grid.

Context length is the independent variable in Phase 1, so it has to be controlled
exactly. These prompts are built by tokenising a fixed bundled corpus and slicing it to
a precise token count - not by guessing at a character count and hoping.

Chat templates are deliberately *not* applied here. A template adds a variable number of
tokens per model, which would mean "2048 context" silently meant something different for
each checkpoint. Benchmark prompts are raw token sequences of exactly the stated length;
the interpretability experiments, where instruction formatting genuinely matters, apply
the template instead.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import torch

from src.utils.io import repo_root
from src.utils.logging import get_logger

logger = get_logger(__name__)

CORPUS_PATH = "data/benchmark_corpus.txt"

#: Appended to every benchmark prompt so the model has a concrete continuation task
#: rather than being asked to extend arbitrary filler.
CONTINUATION_CUE = "\n\nSummary of the passage above:"


@lru_cache(maxsize=1)
def load_corpus() -> str:
    """Read the bundled filler corpus.

    Raises:
        FileNotFoundError: If the corpus is missing, which would otherwise show up much
            later as an unexplained change in measured numbers.
    """
    path = repo_root() / CORPUS_PATH
    if not path.is_file():
        raise FileNotFoundError(
            f"benchmark corpus not found at {path}; it is tracked in the repo and is "
            "required for reproducible context-length control"
        )
    return path.read_text(encoding="utf-8")


def build_fixed_length_ids(tokenizer: Any, n_tokens: int) -> torch.Tensor:
    """Return exactly ``n_tokens`` token ids drawn from the bundled corpus.

    The corpus is tiled if it is shorter than the requested length, so the same code
    path serves a 128-token prompt and a 32k one. Special tokens are not added, because
    the count must be exact.

    Args:
        tokenizer: A Hugging Face tokenizer.
        n_tokens: Desired prompt length in tokens. Must be at least 1.

    Returns:
        A 1-D LongTensor of length ``n_tokens``.
    """
    if n_tokens < 1:
        raise ValueError(f"n_tokens must be >= 1, got {n_tokens}")

    cue_ids = tokenizer(CONTINUATION_CUE, add_special_tokens=False)["input_ids"]
    body_target = max(n_tokens - len(cue_ids), 1)

    corpus_ids = tokenizer(load_corpus(), add_special_tokens=False)["input_ids"]
    if not corpus_ids:
        raise ValueError("benchmark corpus tokenised to zero tokens")

    repeats = -(-body_target // len(corpus_ids))  # ceiling division
    body = (corpus_ids * repeats)[:body_target]

    ids = body + cue_ids
    ids = ids[:n_tokens]
    if len(ids) < n_tokens:  # pragma: no cover - only if the cue is longer than n_tokens
        ids = ids + corpus_ids[: n_tokens - len(ids)]

    assert len(ids) == n_tokens, f"built {len(ids)} tokens, wanted {n_tokens}"
    return torch.tensor(ids, dtype=torch.long)


def build_batch(
    tokenizer: Any,
    n_tokens: int,
    batch_size: int,
    device: str | torch.device,
) -> dict[str, torch.Tensor]:
    """Build a uniform-length batch ready for ``generate``.

    Every row is identical, which is intentional: the measurement targets the model's
    throughput at a given shape, not the variance of natural prompts.

    Returns:
        A dict with ``input_ids`` and ``attention_mask`` on ``device``.
    """
    ids = build_fixed_length_ids(tokenizer, n_tokens)
    input_ids = ids.unsqueeze(0).repeat(batch_size, 1).to(device)
    attention_mask = torch.ones_like(input_ids)
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def max_supported_context(model_config: Any, requested: int) -> tuple[int, str | None]:
    """Clamp a requested context length to what the model architecture allows.

    Returns:
        ``(usable_length, reason)`` where ``reason`` is None when nothing was clamped.
    """
    limit = getattr(model_config, "max_position_embeddings", None)
    if limit is None or requested <= limit:
        return requested, None
    return limit, (
        f"requested context {requested} exceeds the model's "
        f"max_position_embeddings ({limit})"
    )
