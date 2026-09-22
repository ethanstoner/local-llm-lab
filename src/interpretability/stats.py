"""Statistics over captured activations.

Everything here operates on ``(n_prompts, hidden_size)`` float32 tensors and returns
plain Python numbers or small tensors, so results serialise directly into JSON. PCA uses
``torch.pca_lowrank`` rather than scikit-learn: it is accurate enough for a two- or
three-component visualisation and avoids a dependency that would otherwise exist solely
for one figure.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

from src.utils.logging import get_logger

logger = get_logger(__name__)


def mean_vector(activations: torch.Tensor) -> torch.Tensor:
    """Return the mean activation vector over prompts."""
    return activations.mean(dim=0)


def vector_norms(activations: torch.Tensor) -> dict[str, float]:
    """Summarise the L2 norms of a set of activation vectors.

    Residual-stream norm grows substantially with depth in most models, so comparing raw
    projection magnitudes across layers without this context is misleading.
    """
    norms = activations.norm(dim=-1)
    return {
        "mean": float(norms.mean()),
        "std": float(norms.std(unbiased=False)),
        "min": float(norms.min()),
        "max": float(norms.max()),
    }


def cosine_similarity(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> float:
    """Cosine similarity between two 1-D vectors."""
    denominator = a.norm() * b.norm()
    if float(denominator) < eps:
        return 0.0
    return float(torch.dot(a, b) / denominator)


def pairwise_cosine_matrix(vectors: torch.Tensor) -> torch.Tensor:
    """Return the ``(n, n)`` cosine-similarity matrix of a stack of row vectors."""
    normalised = vectors / vectors.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return normalised @ normalised.T


def difference_in_means(
    positive: torch.Tensor,
    negative: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the difference-in-means direction between two activation sets.

    Args:
        positive: ``(n_pos, hidden)`` activations for the first class.
        negative: ``(n_neg, hidden)`` activations for the second class.

    Returns:
        ``(raw_difference, unit_direction)``. The raw difference keeps its magnitude,
        which carries information about how far apart the class means are; the unit
        vector is what projections are taken against.
    """
    raw = mean_vector(positive) - mean_vector(negative)
    norm = raw.norm()
    if float(norm) < 1e-8:
        logger.warning("Difference-in-means vector is ~zero; direction is undefined")
        return raw, torch.zeros_like(raw)
    return raw, raw / norm


def project(activations: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    """Project each activation onto a direction.

    Args:
        activations: ``(n, hidden)``.
        direction: ``(hidden,)``, expected to be unit norm.

    Returns:
        ``(n,)`` scalar projections.
    """
    return activations @ direction


def cohens_d(a: torch.Tensor, b: torch.Tensor) -> float:
    """Standardised mean difference between two 1-D samples.

    Uses the pooled standard deviation. Reported alongside AUROC because the two answer
    different questions: effect size versus rank separability.
    """
    n_a, n_b = a.numel(), b.numel()
    if n_a < 2 or n_b < 2:
        return float("nan")
    var_a = float(a.var(unbiased=True))
    var_b = float(b.var(unbiased=True))
    pooled = math.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    if pooled < 1e-12:
        return float("nan")
    return (float(a.mean()) - float(b.mean())) / pooled


def auroc(positive: torch.Tensor, negative: torch.Tensor) -> float:
    """Area under the ROC curve for separating two samples by a scalar score.

    Computed via the Mann-Whitney U statistic with tie correction, which is exact and
    needs no threshold sweep.

    Returns:
        A value in [0, 1]; 0.5 means the score carries no information.
    """
    n_pos, n_neg = positive.numel(), negative.numel()
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    scores = torch.cat([positive, negative]).to(torch.float64)
    order = torch.argsort(scores)
    sorted_scores = scores[order]

    # Average ranks within tied groups so ties contribute 0.5 rather than 0 or 1.
    ranks = torch.empty_like(sorted_scores)
    i = 0
    while i < sorted_scores.numel():
        j = i
        while j + 1 < sorted_scores.numel() and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        average_rank = (i + j) / 2.0 + 1.0
        ranks[i : j + 1] = average_rank
        i = j + 1

    original_ranks = torch.empty_like(ranks)
    original_ranks[order] = ranks
    rank_sum_pos = float(original_ranks[:n_pos].sum())
    u = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
    return u / (n_pos * n_neg)


@dataclass
class SeparationStats:
    """How well a scalar score separates two classes."""

    mean_positive: float
    mean_negative: float
    std_positive: float
    std_negative: float
    cohens_d: float
    auroc: float
    n_positive: int
    n_negative: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        return {
            "mean_positive": round(self.mean_positive, 6),
            "mean_negative": round(self.mean_negative, 6),
            "std_positive": round(self.std_positive, 6),
            "std_negative": round(self.std_negative, 6),
            "mean_gap": round(self.mean_positive - self.mean_negative, 6),
            "cohens_d": round(self.cohens_d, 6),
            "auroc": round(self.auroc, 6),
            "n_positive": self.n_positive,
            "n_negative": self.n_negative,
        }


def separation(positive: torch.Tensor, negative: torch.Tensor) -> SeparationStats:
    """Summarise how far apart two sets of scalar scores are."""
    return SeparationStats(
        mean_positive=float(positive.mean()),
        mean_negative=float(negative.mean()),
        std_positive=float(positive.std(unbiased=False)),
        std_negative=float(negative.std(unbiased=False)),
        cohens_d=cohens_d(positive, negative),
        auroc=auroc(positive, negative),
        n_positive=int(positive.numel()),
        n_negative=int(negative.numel()),
    )


def pca(
    activations: torch.Tensor,
    n_components: int = 2,
    center: bool = True,
) -> dict[str, Any]:
    """Project activations onto their leading principal components.

    Args:
        activations: ``(n, hidden)``.
        n_components: How many components to return.
        center: Subtract the mean first, as PCA normally requires.

    Returns:
        A dict with ``components`` ``(n_components, hidden)``, ``projected``
        ``(n, n_components)``, and ``explained_variance_ratio``.
    """
    matrix = activations.to(torch.float32)
    if center:
        matrix = matrix - matrix.mean(dim=0, keepdim=True)

    q = min(n_components + 4, min(matrix.shape))
    _, s, v = torch.pca_lowrank(matrix, q=q, center=False)

    components = v[:, :n_components].T
    projected = matrix @ components.T

    total_variance = float((matrix**2).sum())
    explained = [(float(sv) ** 2) / total_variance if total_variance > 0 else 0.0 for sv in s[:n_components]]

    return {
        "components": components,
        "projected": projected,
        "explained_variance_ratio": explained,
        "n_components": n_components,
    }


def layer_norm_profile(activations_by_layer: dict[int, torch.Tensor]) -> list[dict[str, Any]]:
    """Return per-layer activation-norm statistics, ordered by depth."""
    profile = []
    for index in sorted(activations_by_layer):
        stats = vector_norms(activations_by_layer[index])
        profile.append({"layer": index, **{k: round(v, 4) for k, v in stats.items()}})
    return profile


def wilson_interval(successes: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion.

    Used for refusal rates, which are measured on a few dozen prompts. The normal
    approximation collapses to a zero-width interval at 0% or 100%, which is exactly
    where the interesting ablation results land; Wilson does not.
    """
    if n <= 0:
        return (float("nan"), float("nan"))
    if not 0 <= successes <= n:
        raise ValueError(f"successes={successes} outside [0, {n}]")
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))
