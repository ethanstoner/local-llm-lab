"""Quality measurement for quantized models.

The question Phase 2 has to answer is "what did the memory saving cost in output
quality", and it has to answer it with numbers that are actually computable on one card.
That rules out an LLM judge (a second model, unvalidated, and unreproducible) and rules
out benchmark suites that need hours per configuration.

What is used instead is a set of deterministic comparisons against the full-precision
reference:

* **Greedy continuation agreement** - with sampling disabled, the reference and the
  candidate should emit the same tokens. Where they first diverge, and how often they
  agree, is a direct measure of behavioural drift.
* **Next-token distribution divergence** - exact KL and Jensen-Shannon between the two
  models' final-position distributions on a fixed prompt set.
* **Teacher-forced agreement and perplexity** over a fixed corpus - about a thousand
  positions rather than a few dozen. It is a paired comparison over identical positions,
  so relative differences between precisions are far better determined than the absolute
  perplexity is.

The reference is measured once and reduced to a small set of artifacts, so the two
models are never resident on the card at the same time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import torch

from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TeacherForcedTrace:
    """Per-position outputs of a teacher-forced pass over a fixed corpus.

    Deliberately compact: storing full vocabulary distributions for every
    position would run to gigabytes, so only the argmax and the log-probability of the
    actual next token are kept. Both are enough to compute the comparisons that matter
    and neither is an approximation.
    """

    argmax_ids: torch.Tensor
    target_logprobs: torch.Tensor
    n_positions: int

    def perplexity(self) -> float:
        """Perplexity implied by the stored target log-probabilities."""
        return float(torch.exp(-self.target_logprobs.mean()))


@dataclass
class ReferenceArtifacts:
    """Everything the reference model contributes to the comparison."""

    precision: str
    trace: TeacherForcedTrace
    final_logprobs: torch.Tensor
    continuations: list[list[int]]
    prompts: list[str]

    def summary(self) -> dict[str, Any]:
        """Return a JSON-serialisable description of the reference itself."""
        return {
            "precision": self.precision,
            "perplexity": round(self.trace.perplexity(), 6),
            "n_teacher_forced_positions": self.trace.n_positions,
            "n_prompts": len(self.prompts),
            "continuation_tokens": len(self.continuations[0]) if self.continuations else 0,
        }


@torch.inference_mode()
def teacher_forced_pass(
    model: Any,
    tokenizer: Any,
    text: str,
    max_tokens: int,
    window: int = 1024,
    stride: int = 512,
    device: str | torch.device | None = None,
) -> TeacherForcedTrace:
    """Run a strided teacher-forced pass and record per-position predictions.

    A sliding window with overlap is used so that every scored position has a reasonable
    amount of left context, rather than the first tokens of each chunk being predicted
    from almost nothing.

    Args:
        model: Causal LM in eval mode.
        tokenizer: Its tokenizer.
        text: Corpus to score.
        max_tokens: Truncate the corpus to this many tokens.
        window: Context window per forward pass.
        stride: How far the window advances; ``window - stride`` tokens of each chunk are
            context-only and are not scored.
        device: Device for the inputs.

    Returns:
        A :class:`TeacherForcedTrace` covering every scored position, in order.
    """
    target = device or next(model.parameters()).device
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    ids = ids[:max_tokens]

    if ids.numel() < 2:
        raise ValueError("corpus tokenised to fewer than two tokens")

    argmax_chunks: list[torch.Tensor] = []
    logprob_chunks: list[torch.Tensor] = []
    previous_end = 0

    for start in range(0, ids.numel(), stride):
        end = min(start + window, ids.numel())
        chunk = ids[start:end].unsqueeze(0).to(target)
        if chunk.shape[-1] < 2:
            break

        logits = model(input_ids=chunk, use_cache=False).logits.float()
        logprobs = torch.log_softmax(logits, dim=-1)[0]

        # Position i predicts token i+1, so the last position has no target here.
        predictions = logprobs[:-1]
        targets = chunk[0, 1:]

        # Skip positions already scored by the previous window.
        first_new = max(previous_end - start - 1, 0)
        if first_new >= predictions.shape[0]:
            if end == ids.numel():
                break
            continue

        selected = predictions[first_new:]
        selected_targets = targets[first_new:]

        argmax_chunks.append(selected.argmax(dim=-1).cpu())
        logprob_chunks.append(
            selected.gather(1, selected_targets.unsqueeze(1)).squeeze(1).cpu()
        )
        previous_end = end

        if end == ids.numel():
            break

    argmax_ids = torch.cat(argmax_chunks)
    target_logprobs = torch.cat(logprob_chunks)
    return TeacherForcedTrace(
        argmax_ids=argmax_ids,
        target_logprobs=target_logprobs,
        n_positions=int(argmax_ids.numel()),
    )


@torch.inference_mode()
def final_position_logprobs(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    batch_size: int = 8,
    apply_chat_template: bool = True,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    """Return ``(n_prompts, vocab)`` log-probabilities for the token after each prompt.

    Kept on the host in float32. For a 150k vocabulary and a few dozen prompts this is
    tens of megabytes, which is affordable; one vector per corpus position would not be.
    """
    target = device or next(model.parameters()).device
    texts = _format_prompts(tokenizer, prompts, apply_chat_template)

    outputs: list[torch.Tensor] = []
    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        encoded = tokenizer(
            chunk,
            return_tensors="pt",
            padding=True,
            add_special_tokens=not apply_chat_template,
        )
        encoded = {k: v.to(target) for k, v in encoded.items()}
        logits = model(**encoded, use_cache=False).logits[:, -1, :].float()
        outputs.append(torch.log_softmax(logits, dim=-1).cpu())
    return torch.cat(outputs, dim=0)


@torch.inference_mode()
def greedy_continuations(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    max_new_tokens: int,
    batch_size: int = 8,
    apply_chat_template: bool = True,
    device: str | torch.device | None = None,
) -> list[list[int]]:
    """Generate a fixed-length greedy continuation for each prompt.

    ``min_new_tokens`` matches ``max_new_tokens`` so that two models cannot be compared
    over different numbers of tokens.
    """
    target = device or next(model.parameters()).device
    texts = _format_prompts(tokenizer, prompts, apply_chat_template)

    results: list[list[int]] = []
    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        encoded = tokenizer(
            chunk,
            return_tensors="pt",
            padding=True,
            add_special_tokens=not apply_chat_template,
        )
        encoded = {k: v.to(target) for k, v in encoded.items()}
        prompt_len = encoded["input_ids"].shape[-1]
        sequences = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            min_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
        )
        for row in sequences[:, prompt_len:]:
            results.append([int(t) for t in row.tolist()])
    return results


def _format_prompts(
    tokenizer: Any, prompts: Sequence[str], apply_chat_template: bool
) -> list[str]:
    """Apply the model's chat template when it has one."""
    if not apply_chat_template or not getattr(tokenizer, "chat_template", None):
        return list(prompts)
    return [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
        )
        for p in prompts
    ]


def continuation_agreement(
    reference: Sequence[Sequence[int]],
    candidate: Sequence[Sequence[int]],
) -> dict[str, Any]:
    """Compare two sets of greedy continuations token by token.

    Returns:
        Exact-match rate, mean token-level agreement, and the mean position at which the
        two models first diverge (the most interpretable single number: "they agree for
        the first N tokens").
    """
    if len(reference) != len(candidate):
        raise ValueError(
            f"continuation count mismatch: {len(reference)} vs {len(candidate)}"
        )
    if not reference:
        return {"n": 0}

    exact = 0
    agreements: list[float] = []
    divergence_points: list[int] = []

    for ref, cand in zip(reference, candidate):
        length = min(len(ref), len(cand))
        matches = sum(1 for i in range(length) if ref[i] == cand[i])
        agreements.append(matches / length if length else 0.0)

        first_diff = length
        for i in range(length):
            if ref[i] != cand[i]:
                first_diff = i
                break
        divergence_points.append(first_diff)
        if first_diff == length and len(ref) == len(cand):
            exact += 1

    n = len(reference)
    return {
        "n": n,
        "exact_match_rate": round(exact / n, 6),
        "mean_token_agreement": round(sum(agreements) / n, 6),
        "mean_first_divergence_index": round(sum(divergence_points) / n, 4),
        "max_tokens_compared": min(len(r) for r in reference),
    }


def distribution_divergence(
    reference_logprobs: torch.Tensor,
    candidate_logprobs: torch.Tensor,
) -> dict[str, Any]:
    """Compare two sets of next-token log-probability vectors.

    Args:
        reference_logprobs: ``(n, vocab)`` from the reference model.
        candidate_logprobs: ``(n, vocab)`` from the candidate model.

    Returns:
        Mean and maximum KL(reference || candidate), mean Jensen-Shannon divergence, and
        top-1 agreement. KL is asymmetric and the reference is the correct first
        argument: it measures the cost of using the candidate in place of the reference.
    """
    if reference_logprobs.shape != candidate_logprobs.shape:
        raise ValueError(
            f"shape mismatch: {tuple(reference_logprobs.shape)} vs "
            f"{tuple(candidate_logprobs.shape)}"
        )

    ref = reference_logprobs.double()
    cand = candidate_logprobs.double()
    p = ref.exp()

    kl = (p * (ref - cand)).sum(dim=-1)

    mixture = torch.logaddexp(ref, cand) - math.log(2.0)
    q = cand.exp()
    js = 0.5 * (p * (ref - mixture)).sum(dim=-1) + 0.5 * (q * (cand - mixture)).sum(dim=-1)

    top1_match = (ref.argmax(dim=-1) == cand.argmax(dim=-1)).double()

    return {
        "n": int(ref.shape[0]),
        "vocab_size": int(ref.shape[1]),
        "mean_kl_nats": round(float(kl.mean()), 8),
        "median_kl_nats": round(float(kl.median()), 8),
        "max_kl_nats": round(float(kl.max()), 8),
        "mean_js_nats": round(float(js.mean()), 8),
        "top1_agreement": round(float(top1_match.mean()), 6),
    }


def trace_comparison(
    reference: TeacherForcedTrace,
    candidate: TeacherForcedTrace,
) -> dict[str, Any]:
    """Compare two teacher-forced traces over the same corpus."""
    n = min(reference.n_positions, candidate.n_positions)
    if n == 0:
        return {"n_positions": 0}

    ref_argmax = reference.argmax_ids[:n]
    cand_argmax = candidate.argmax_ids[:n]
    ref_lp = reference.target_logprobs[:n]
    cand_lp = candidate.target_logprobs[:n]

    delta = (cand_lp - ref_lp).double()
    ref_ppl = float(torch.exp(-ref_lp.mean()))
    cand_ppl = float(torch.exp(-cand_lp.mean()))

    return {
        "n_positions": int(n),
        "top1_agreement": round(float((ref_argmax == cand_argmax).double().mean()), 6),
        "reference_perplexity": round(ref_ppl, 6),
        "candidate_perplexity": round(cand_ppl, 6),
        "perplexity_ratio": round(cand_ppl / ref_ppl, 6) if ref_ppl > 0 else None,
        "mean_target_logprob_delta": round(float(delta.mean()), 8),
        "mean_abs_target_logprob_delta": round(float(delta.abs().mean()), 8),
    }


@dataclass
class QualityComparison:
    """The full quality report for one candidate precision."""

    precision: str
    status: str = "ok"
    reason: str | None = None
    continuation: dict[str, Any] = field(default_factory=dict)
    distribution: dict[str, Any] = field(default_factory=dict)
    teacher_forced: dict[str, Any] = field(default_factory=dict)
    load: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        return {
            "precision": self.precision,
            "status": self.status,
            "reason": self.reason,
            "continuation_agreement": self.continuation,
            "next_token_distribution": self.distribution,
            "teacher_forced": self.teacher_forced,
            "load": self.load,
        }

    def to_row(self) -> dict[str, Any]:
        """Flatten into one CSV row."""
        row: dict[str, Any] = {"precision": self.precision, "status": self.status}
        row.update({f"cont_{k}": v for k, v in self.continuation.items()})
        row.update({f"dist_{k}": v for k, v in self.distribution.items()})
        row.update({f"tf_{k}": v for k, v in self.teacher_forced.items()})
        row["load_weights_mib"] = self.load.get("weights_mib")
        row["load_idle_device_used_mib"] = self.load.get("idle_device_used_mib")
        return row
