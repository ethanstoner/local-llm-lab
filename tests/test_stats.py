"""Activation statistics, checked against hand-computable values."""

from __future__ import annotations

import math

import pytest
import torch

from src.interpretability.stats import (
    auroc,
    cohens_d,
    cosine_similarity,
    difference_in_means,
    layer_norm_profile,
    pairwise_cosine_matrix,
    pca,
    project,
    separation,
    vector_norms,
)


def test_cosine_similarity_known_values() -> None:
    a = torch.tensor([1.0, 0.0])
    assert cosine_similarity(a, torch.tensor([2.0, 0.0])) == pytest.approx(1.0)
    assert cosine_similarity(a, torch.tensor([0.0, 1.0])) == pytest.approx(0.0, abs=1e-6)
    assert cosine_similarity(a, torch.tensor([-1.0, 0.0])) == pytest.approx(-1.0)


def test_cosine_similarity_handles_zero_vector() -> None:
    assert cosine_similarity(torch.zeros(4), torch.ones(4)) == 0.0


def test_difference_in_means_direction_is_unit() -> None:
    positive = torch.tensor([[2.0, 0.0], [4.0, 0.0]])
    negative = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
    raw, unit = difference_in_means(positive, negative)
    assert raw.tolist() == [3.0, 0.0]
    assert float(unit.norm()) == pytest.approx(1.0)
    assert unit.tolist() == pytest.approx([1.0, 0.0])


def test_difference_in_means_identical_classes() -> None:
    same = torch.ones(5, 3)
    _, unit = difference_in_means(same, same.clone())
    assert float(unit.norm()) == 0.0


def test_project() -> None:
    activations = torch.tensor([[3.0, 4.0], [1.0, 0.0]])
    direction = torch.tensor([1.0, 0.0])
    assert project(activations, direction).tolist() == [3.0, 1.0]


def test_auroc_perfect_separation() -> None:
    assert auroc(torch.tensor([3.0, 4.0, 5.0]), torch.tensor([0.0, 1.0, 2.0])) == 1.0


def test_auroc_reversed_separation() -> None:
    assert auroc(torch.tensor([0.0, 1.0]), torch.tensor([3.0, 4.0])) == 0.0


def test_auroc_all_ties_is_chance() -> None:
    """Complete ties must land exactly on 0.5, not on an arbitrary side."""
    assert auroc(torch.ones(4), torch.ones(4)) == pytest.approx(0.5)


def test_auroc_partial_overlap() -> None:
    # positives {1, 3}, negatives {2, 4}: pairs won = (1>2)F (1>4)F (3>2)T (3>4)F = 1/4.
    assert auroc(torch.tensor([1.0, 3.0]), torch.tensor([2.0, 4.0])) == pytest.approx(0.25)


def test_cohens_d_sign_and_scale() -> None:
    a = torch.tensor([2.0, 2.0, 2.0, 2.0])
    b = torch.tensor([0.0, 0.0, 0.0, 0.0])
    assert math.isnan(cohens_d(a, b))  # zero pooled variance

    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([4.0, 5.0, 6.0])
    assert cohens_d(a, b) < 0
    assert cohens_d(b, a) == pytest.approx(-cohens_d(a, b))


def test_separation_reports_both_measures() -> None:
    stats = separation(torch.tensor([5.0, 6.0, 7.0]), torch.tensor([1.0, 2.0, 3.0]))
    payload = stats.to_dict()
    assert payload["auroc"] == 1.0
    assert payload["mean_gap"] == pytest.approx(4.0)
    assert payload["n_positive"] == 3


def test_vector_norms() -> None:
    stats = vector_norms(torch.tensor([[3.0, 4.0], [6.0, 8.0]]))
    assert stats["mean"] == pytest.approx(7.5)
    assert stats["min"] == pytest.approx(5.0)
    assert stats["max"] == pytest.approx(10.0)


def test_pairwise_cosine_matrix_diagonal_is_one() -> None:
    matrix = pairwise_cosine_matrix(torch.randn(5, 8))
    assert torch.allclose(matrix.diagonal(), torch.ones(5), atol=1e-5)
    assert torch.allclose(matrix, matrix.T, atol=1e-5)


def test_pca_shapes_and_dominant_axis() -> None:
    torch.manual_seed(0)
    # Variance almost entirely along the first coordinate.
    data = torch.zeros(60, 4)
    data[:, 0] = torch.randn(60) * 10.0
    data[:, 1] = torch.randn(60) * 0.01

    result = pca(data, n_components=2)
    assert result["projected"].shape == (60, 2)
    assert result["components"].shape == (2, 4)
    assert result["explained_variance_ratio"][0] > 0.99
    assert abs(result["components"][0][0]) > 0.99


def test_layer_norm_profile_is_ordered() -> None:
    profile = layer_norm_profile({2: torch.ones(3, 4), 0: torch.ones(3, 4) * 2})
    assert [row["layer"] for row in profile] == [0, 2]
    assert profile[0]["mean"] == pytest.approx(4.0)
