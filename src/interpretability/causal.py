"""Measurement helpers for the Phase 6 causal test.

A condition is an intervention (or none) applied while the model answers a prompt set.
For each condition this records how often the model refuses, and - because a refusal
rate alone cannot tell "stopped refusing" apart from "stopped producing language" - how
fluent the completions are.

Fluency is scored as each completion's mean per-token negative log-likelihood, by up to
two models and always with the intervention's hooks removed:

* ``self`` - the intact subject model. Sensitive, but biased: greedy output is by
  construction the intact model's own most likely text, so *any* change to the output
  raises this score, fluent or not.
* ``judge`` - an independent smaller model sharing the tokenizer. It has no stake in the
  subject model's particular greedy path, so it separates "different" from "broken".

Harmful-prompt completions are held in memory only for as long as it takes to classify
and score them. Nothing but counts and summary statistics leaves this module.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from statistics import mean, median
from typing import Any

import torch

from src.interpretability.refusal import classify_refusal
from src.interpretability.stats import wilson_interval
from src.utils.logging import get_logger

logger = get_logger(__name__)


def format_chat(tokenizer: Any, prompts: Sequence[str]) -> list[str]:
    """Wrap each instruction in the model's chat template."""
    return [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
        )
        for p in prompts
    ]


def _stop_ids(tokenizer: Any) -> set[int]:
    ids = {tokenizer.eos_token_id, tokenizer.pad_token_id}
    return {i for i in ids if i is not None}


def trim_completion(ids: Sequence[int], stop_ids: set[int]) -> list[int]:
    """Cut a generated row at its first end-of-sequence or padding token."""
    out: list[int] = []
    for token in ids:
        if token in stop_ids:
            break
        out.append(int(token))
    return out


@torch.inference_mode()
def generate_completions(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    max_new_tokens: int,
    batch_size: int = 8,
    device: str | torch.device | None = None,
) -> list[list[int]]:
    """Greedy chat completions, returned as token ids trimmed at end-of-sequence.

    Unlike the benchmark harness this does *not* force a fixed length: whether the model
    stops early is part of the behaviour being measured.
    """
    target = device or next(model.parameters()).device
    stops = _stop_ids(tokenizer)
    texts = format_chat(tokenizer, prompts)
    results: list[list[int]] = []
    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        encoded = tokenizer(chunk, return_tensors="pt", padding=True, add_special_tokens=False)
        encoded = {k: v.to(target) for k, v in encoded.items()}
        prompt_len = encoded["input_ids"].shape[-1]
        sequences = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
        )
        for row in sequences[:, prompt_len:].tolist():
            results.append(trim_completion(row, stops))
    return results


@torch.inference_mode()
def completion_nll(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    completions: Sequence[Sequence[int]],
    device: str | torch.device | None = None,
) -> list[float | None]:
    """Mean per-token negative log-likelihood of each completion given its prompt.

    Scored one sequence at a time, which avoids any interaction between padding and
    position ids; at a few dozen short sequences per condition the cost is negligible.
    Empty completions score ``None``.
    """
    if len(prompts) != len(completions):
        raise ValueError(f"{len(prompts)} prompts but {len(completions)} completions")
    target = device or next(model.parameters()).device
    scores: list[float | None] = []
    for text, completion in zip(format_chat(tokenizer, prompts), completions, strict=True):
        if not completion:
            scores.append(None)
            continue
        prompt_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        ids = torch.tensor([list(prompt_ids) + list(completion)], device=target)
        logits = model(input_ids=ids, use_cache=False).logits[0].float()
        logprobs = torch.log_softmax(logits[len(prompt_ids) - 1 : -1], dim=-1)
        targets = ids[0, len(prompt_ids) :]
        nll = -logprobs.gather(1, targets.unsqueeze(1)).squeeze(1)
        scores.append(float(nll.mean()))
    return scores


def distinct_ratio(ids: Sequence[int]) -> float | None:
    """Fraction of a completion's tokens that are distinct. Degenerate loops score low."""
    return len(set(ids)) / len(ids) if ids else None


@dataclass
class ConditionResult:
    """What one intervention did to one prompt set."""

    condition: str
    prompt_set: str
    n: int
    refusals: int
    nll: dict[str, tuple[float | None, float | None]]
    mean_length: float
    empty_rate: float
    mean_distinct_ratio: float | None
    extra: dict[str, Any] = field(default_factory=dict)
    examples: list[str] = field(default_factory=list)

    @property
    def refusal_rate(self) -> float:
        return self.refusals / self.n if self.n else float("nan")

    def to_row(self) -> dict[str, Any]:
        low, high = wilson_interval(self.refusals, self.n)
        row = {
            "condition": self.condition,
            "prompt_set": self.prompt_set,
            "n": self.n,
            "refusals": self.refusals,
            "refusal_rate": round(self.refusal_rate, 4),
            "refusal_ci_low": round(low, 4),
            "refusal_ci_high": round(high, 4),
            "mean_length": round(self.mean_length, 2),
            "empty_rate": round(self.empty_rate, 4),
            "mean_distinct_ratio": (
                None if self.mean_distinct_ratio is None else round(self.mean_distinct_ratio, 4)
            ),
        }
        for scorer, (mean_value, median_value) in self.nll.items():
            row[f"mean_nll_{scorer}"] = None if mean_value is None else round(mean_value, 4)
            row[f"median_nll_{scorer}"] = None if median_value is None else round(median_value, 4)
        row.update(self.extra)
        return row

    def to_dict(self) -> dict[str, Any]:
        payload = self.to_row()
        if self.examples:
            payload["example_openings"] = self.examples
        return payload


def evaluate_condition(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    intervention: Callable[[], Any],
    condition: str,
    prompt_set: str,
    max_new_tokens: int,
    batch_size: int = 8,
    keep_examples: int = 0,
    extra: dict[str, Any] | None = None,
    device: str | torch.device | None = None,
    scorers: dict[str, Any] | None = None,
) -> ConditionResult:
    """Generate under an intervention, then classify and score with the intact model.

    Args:
        intervention: Zero-argument factory returning a context manager that applies the
            intervention. A factory rather than an instance, so the hooks exist only for
            the generation step and are provably gone before fluency is scored.
        keep_examples: How many completion openings to retain verbatim. Callers pass 0
            for harmful prompt sets; nothing from those completions is kept.
        scorers: Name -> model used to score fluency. Defaults to ``{"self": model}``.
            Every scorer must share ``tokenizer``'s vocabulary.
    """
    with intervention():
        completions = generate_completions(
            model, tokenizer, prompts, max_new_tokens, batch_size=batch_size, device=device
        )

    texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in completions]
    flags = [classify_refusal(t) for t in texts]
    nll: dict[str, tuple[float | None, float | None]] = {}
    for scorer_name, scorer in (scorers or {"self": model}).items():
        scorer_device = next(scorer.parameters()).device
        values = [
            v
            for v in completion_nll(scorer, tokenizer, prompts, completions, scorer_device)
            if v is not None
        ]
        nll[scorer_name] = (mean(values), median(values)) if values else (None, None)
    distinct = [d for d in (distinct_ratio(c) for c in completions) if d is not None]

    examples = [t.strip()[:160] for t in texts[:keep_examples]] if keep_examples else []
    result = ConditionResult(
        condition=condition,
        prompt_set=prompt_set,
        n=len(prompts),
        refusals=sum(flags),
        nll=nll,
        mean_length=mean(len(c) for c in completions) if completions else 0.0,
        empty_rate=sum(1 for c in completions if not c) / len(completions) if completions else 0.0,
        mean_distinct_ratio=mean(distinct) if distinct else None,
        extra=dict(extra or {}),
        examples=examples,
    )
    del texts, completions
    logger.info(
        "%-24s %-12s refusal %2d/%-2d  nll %s",
        condition,
        prompt_set,
        result.refusals,
        result.n,
        " ".join(
            f"{k}={'-' if v[0] is None else f'{v[0]:.3f}'}" for k, v in result.nll.items()
        ),
    )
    return result


def select_causal_layer(
    candidates: Sequence[dict[str, Any]],
    max_perplexity_ratio: float,
    max_layer_exclusive: float | None = None,
) -> int | None:
    """Pick the layer whose direction best removes refusal, among directions that work.

    The selection criteria of Arditi et al. (2024), applied to measured quantities. A
    candidate layer is eligible only if all three hold:

    * **It is not near the output.** Layers at or beyond ``max_layer_exclusive`` are
      excluded: a direction read from the last few blocks can suppress refusal by
      disrupting the output distribution directly, and adding it there produces
      degenerate text rather than refusal.
    * **Removing it is cheap.** Ablating it keeps corpus perplexity within
      ``max_perplexity_ratio`` of the intact model.
    * **Adding it induces refusal.** The 95% Wilson lower bound of the refusal rate on
      harmless prompts with the direction added exceeds the unmodified model's rate on
      the same prompts. A direction that removes refusal but cannot produce it is not
      evidence of a refusal representation.

    Among eligible layers the lowest refusal rate on harmful prompts with the direction
    ablated wins; ties go to the smaller perplexity cost, then the earlier layer. Any
    missing measurement makes a candidate ineligible rather than assumed acceptable.

    Args:
        candidates: Rows with ``layer``, ``refusals`` and ``n`` (harmful prompts, direction
            ablated), ``perplexity_ratio``, ``induced_refusals`` and ``induced_n``
            (harmless prompts, direction added) and ``harmless_baseline_rate``.
        max_perplexity_ratio: The capability budget.
        max_layer_exclusive: Layers from here on are excluded; ``None`` disables the
            depth filter.

    Returns:
        The selected layer, or ``None`` if nothing is eligible.
    """

    def induces(c: dict[str, Any]) -> bool:
        if c.get("induced_n") in (None, 0) or c.get("harmless_baseline_rate") is None:
            return False
        low, _ = wilson_interval(int(c["induced_refusals"]), int(c["induced_n"]))
        return low > float(c["harmless_baseline_rate"])

    eligible = [
        c
        for c in candidates
        if c.get("n")
        and c.get("perplexity_ratio") is not None
        and c["perplexity_ratio"] <= max_perplexity_ratio
        and (max_layer_exclusive is None or c["layer"] < max_layer_exclusive)
        and induces(c)
    ]
    if not eligible:
        return None
    best = min(eligible, key=lambda c: (c["refusals"] / c["n"], c["perplexity_ratio"], c["layer"]))
    return int(best["layer"])
