"""Quality comparison maths and the refusal-direction fitting logic."""

from __future__ import annotations

import math

import pytest
import torch

from src.evaluation.quality import (
    TeacherForcedTrace,
    continuation_agreement,
    distribution_divergence,
    trace_comparison,
)
from src.interpretability.refusal import (
    classify_refusal,
    direction_agreement,
    fit_all_layers,
    fit_layer_direction,
    summarise,
)
from src.utils.datasets import PromptSets, split_prompt_sets


# -- quality ---------------------------------------------------------------------------


def test_continuation_agreement_identical() -> None:
    result = continuation_agreement([[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]])
    assert result["exact_match_rate"] == 1.0
    assert result["mean_token_agreement"] == 1.0
    assert result["mean_first_divergence_index"] == 3.0


def test_continuation_agreement_diverges_midway() -> None:
    result = continuation_agreement([[1, 2, 3, 4]], [[1, 2, 9, 9]])
    assert result["exact_match_rate"] == 0.0
    assert result["mean_token_agreement"] == 0.5
    assert result["mean_first_divergence_index"] == 2.0


def test_continuation_agreement_rejects_mismatched_counts() -> None:
    with pytest.raises(ValueError, match="count mismatch"):
        continuation_agreement([[1]], [[1], [2]])


def test_distribution_divergence_of_identical_is_zero() -> None:
    logits = torch.randn(4, 50)
    logprobs = torch.log_softmax(logits, dim=-1)
    result = distribution_divergence(logprobs, logprobs)
    assert result["mean_kl_nats"] == pytest.approx(0.0, abs=1e-9)
    assert result["mean_js_nats"] == pytest.approx(0.0, abs=1e-9)
    assert result["top1_agreement"] == 1.0


def test_kl_matches_hand_computation() -> None:
    # p = [0.5, 0.5], q = [0.25, 0.75]  =>  KL = 0.5*ln(2) + 0.5*ln(2/3)
    p = torch.log(torch.tensor([[0.5, 0.5]]))
    q = torch.log(torch.tensor([[0.25, 0.75]]))
    expected = 0.5 * math.log(0.5 / 0.25) + 0.5 * math.log(0.5 / 0.75)
    assert distribution_divergence(p, q)["mean_kl_nats"] == pytest.approx(expected, abs=1e-7)


def test_kl_is_asymmetric() -> None:
    """The reference must be the first argument; swapping it changes the answer."""
    p = torch.log(torch.tensor([[0.5, 0.5]]))
    q = torch.log(torch.tensor([[0.25, 0.75]]))
    assert distribution_divergence(p, q)["mean_kl_nats"] != pytest.approx(
        distribution_divergence(q, p)["mean_kl_nats"], abs=1e-6
    )


def test_distribution_divergence_shape_check() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        distribution_divergence(torch.zeros(2, 5), torch.zeros(2, 6))


def test_trace_comparison() -> None:
    reference = TeacherForcedTrace(
        argmax_ids=torch.tensor([1, 2, 3, 4]),
        target_logprobs=torch.tensor([-1.0, -1.0, -1.0, -1.0]),
        n_positions=4,
    )
    candidate = TeacherForcedTrace(
        argmax_ids=torch.tensor([1, 2, 9, 9]),
        target_logprobs=torch.tensor([-2.0, -2.0, -2.0, -2.0]),
        n_positions=4,
    )
    result = trace_comparison(reference, candidate)

    assert result["top1_agreement"] == 0.5
    assert result["reference_perplexity"] == pytest.approx(math.e)
    assert result["candidate_perplexity"] == pytest.approx(math.e**2)
    assert result["perplexity_ratio"] == pytest.approx(math.e)
    assert result["mean_target_logprob_delta"] == pytest.approx(-1.0)


def test_trace_perplexity() -> None:
    trace = TeacherForcedTrace(torch.tensor([0]), torch.tensor([0.0]), 1)
    assert trace.perplexity() == pytest.approx(1.0)


# -- refusal ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "I'm sorry, but I can't help with that.",
        "I cannot provide that information.",
        "As an AI, I must decline.",
        "  Sorry, I won't do that.",
    ],
)
def test_classify_refusal_positive(text: str) -> None:
    assert classify_refusal(text)


@pytest.mark.parametrize(
    "text",
    [
        "Compound interest is interest calculated on the initial principal.",
        "Sure! Here is a short guide to getting started with running.",
        "",
    ],
)
def test_classify_refusal_negative(text: str) -> None:
    assert not classify_refusal(text)


def test_classify_refusal_ignores_late_caveats() -> None:
    """A compliant answer that adds a caveat far below is not a refusal."""
    answer = "Here is how photosynthesis works. " + "Detail. " * 60 + "I cannot go further."
    assert not classify_refusal(answer)


def _separable_activations(n: int, hidden: int, offset: float, seed: int) -> torch.Tensor:
    """Gaussian activations displaced along the first coordinate."""
    generator = torch.Generator().manual_seed(seed)
    data = torch.randn(n, hidden, generator=generator)
    data[:, 0] += offset
    return data


def test_fit_layer_direction_finds_the_planted_axis() -> None:
    harmful_train = _separable_activations(60, 16, offset=8.0, seed=1)
    harmless_train = _separable_activations(60, 16, offset=-8.0, seed=2)
    harmful_test = _separable_activations(30, 16, offset=8.0, seed=3)
    harmless_test = _separable_activations(30, 16, offset=-8.0, seed=4)

    result = fit_layer_direction(harmful_train, harmless_train, harmful_test, harmless_test, 7)

    assert result.layer == 7
    assert abs(result.direction[0]) > 0.95  # the planted axis dominates
    assert result.test_separation.auroc > 0.95
    assert result.test_separation.cohens_d > 2.0


def test_fit_layer_direction_reports_chance_on_noise() -> None:
    """Identically-distributed classes must not produce a confident direction."""
    a = _separable_activations(80, 16, offset=0.0, seed=11)
    b = _separable_activations(80, 16, offset=0.0, seed=12)
    result = fit_layer_direction(a[:50], b[:50], a[50:], b[50:], 0)

    # Held out, an overfitted direction has nothing to hold on to.
    assert 0.2 < result.test_separation.auroc < 0.8
    assert abs(result.test_separation.cohens_d) < 1.0


def test_fit_all_layers_and_summarise() -> None:
    layers = {}
    for index, offset in enumerate((0.2, 6.0, 1.0)):
        layers[index] = {
            "harmful_train": _separable_activations(40, 12, offset, seed=100 + index),
            "harmless_train": _separable_activations(40, 12, -offset, seed=200 + index),
            "harmful_test": _separable_activations(20, 12, offset, seed=300 + index),
            "harmless_test": _separable_activations(20, 12, -offset, seed=400 + index),
        }

    results = fit_all_layers(
        {i: v["harmful_train"] for i, v in layers.items()},
        {i: v["harmless_train"] for i, v in layers.items()},
        {i: v["harmful_test"] for i, v in layers.items()},
        {i: v["harmless_test"] for i, v in layers.items()},
    )

    assert [r.layer for r in results] == [0, 1, 2]
    summary = summarise(results)
    assert summary["best_layer_by_cohens_d"]["layer"] == 1  # the strongest planted signal

    agreement = direction_agreement(results)
    assert agreement["best_layer"] == 1
    assert len(agreement["adjacent_layer_cosine"]) == 2
    assert all(-1.0 <= e["cosine"] <= 1.0 for e in agreement["adjacent_layer_cosine"])


def test_layer_direction_row_is_flat() -> None:
    result = fit_layer_direction(
        _separable_activations(20, 8, 3.0, 1),
        _separable_activations(20, 8, -3.0, 2),
        _separable_activations(10, 8, 3.0, 3),
        _separable_activations(10, 8, -3.0, 4),
        2,
    )
    row = result.to_row()
    assert row["layer"] == 2
    assert all(not isinstance(v, (dict, list)) for v in row.values())


# -- splitting -------------------------------------------------------------------------


def test_split_is_disjoint_and_balanced() -> None:
    sets = PromptSets(
        harmful=[f"h{i}" for i in range(50)],
        harmless=[f"b{i}" for i in range(50)],
    )
    splits = split_prompt_sets(sets, test_fraction=0.3, seed=7)

    assert not set(splits.harmful_train) & set(splits.harmful_test)
    assert not set(splits.harmless_train) & set(splits.harmless_test)
    assert len(splits.harmful_train) + len(splits.harmful_test) == 50
    assert splits.counts()["harmful_test"] == 15


def test_split_is_seeded() -> None:
    sets = PromptSets(harmful=[str(i) for i in range(20)], harmless=[str(-i) for i in range(20)])
    first = split_prompt_sets(sets, 0.25, seed=3)
    second = split_prompt_sets(sets, 0.25, seed=3)
    assert first.harmful_test == second.harmful_test


def test_split_rejects_degenerate_fraction() -> None:
    sets = PromptSets(harmful=["a", "b"], harmless=["c", "d"])
    with pytest.raises(ValueError, match="test_fraction"):
        split_prompt_sets(sets, 1.0)


def test_split_always_leaves_training_data() -> None:
    sets = PromptSets(harmful=["a", "b"], harmless=["c", "d"])
    splits = split_prompt_sets(sets, 0.9, seed=1)
    assert len(splits.harmful_train) >= 1
    assert len(splits.harmful_test) >= 1
