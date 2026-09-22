"""Seeding and determinism control.

Full bitwise determinism is not achievable for every CUDA kernel a transformer touches,
so this module does two things: it seeds everything it can, and it *reports* what level
of determinism was actually reached. The report is stored in run metadata, which is what
makes a later "why don't these numbers match" question answerable.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class DeterminismReport:
    """What determinism settings were requested and what actually took effect."""

    seed: int
    requested_deterministic: bool
    deterministic_algorithms: bool = False
    cudnn_deterministic: bool = False
    cudnn_benchmark: bool = True
    cublas_workspace_config: str | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view for run metadata."""
        return {
            "seed": self.seed,
            "requested_deterministic": self.requested_deterministic,
            "deterministic_algorithms": self.deterministic_algorithms,
            "cudnn_deterministic": self.cudnn_deterministic,
            "cudnn_benchmark": self.cudnn_benchmark,
            "cublas_workspace_config": self.cublas_workspace_config,
            "warnings": list(self.warnings),
        }


def set_seed(seed: int, deterministic: bool = True) -> DeterminismReport:
    """Seed every RNG this project touches and optionally request deterministic kernels.

    Args:
        seed: The seed applied to ``random``, ``numpy`` and ``torch`` (CPU and CUDA).
        deterministic: When True, ask PyTorch for deterministic algorithms and disable
            cuDNN autotuning. This costs throughput, so benchmark configs may turn it
            off — in which case the returned report records that choice.

    Returns:
        A :class:`DeterminismReport` describing the settings that took effect.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    report = DeterminismReport(seed=seed, requested_deterministic=deterministic)

    if not deterministic:
        torch.backends.cudnn.benchmark = True
        report.cudnn_benchmark = True
        logger.info("Seeded with %d; deterministic algorithms NOT requested", seed)
        return report

    # cuBLAS needs this set before the first CUDA context use to make GEMMs reproducible.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    report.cublas_workspace_config = os.environ.get("CUBLAS_WORKSPACE_CONFIG")

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    report.cudnn_deterministic = True
    report.cudnn_benchmark = False

    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
        report.deterministic_algorithms = True
    except Exception as exc:  # pragma: no cover - depends on the local torch build
        report.warnings.append(f"use_deterministic_algorithms failed: {exc}")
        logger.warning("Could not enable deterministic algorithms: %s", exc)

    logger.info("Seeded with %d; deterministic algorithms requested (warn_only)", seed)
    return report
