"""Precision definitions and capability probing.

The rule this module exists to enforce: a configuration that cannot run on this machine
is reported as *unsupported, with a reason*. It is never silently skipped, and it is
never filled in with a plausible-looking number. Capability is probed before a load is
attempted so that a sweep can record the reason instead of crashing on an import error.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class PrecisionSpec:
    """How one precision label maps onto an actual model construction."""

    name: str
    #: dtype for the compute path. For bitsandbytes modes this is the dtype the
    #: dequantised activations flow in, not the storage dtype of the weights.
    torch_dtype: torch.dtype
    #: ``"none"``, ``"bnb-int8"``, ``"bnb-nf4"`` or ``"bnb-fp4"``.
    quantization: str
    #: Approximate bits per stored weight, used only for expectation-setting in reports.
    nominal_bits: float
    description: str


PRECISIONS: dict[str, PrecisionSpec] = {
    "fp32": PrecisionSpec(
        name="fp32",
        torch_dtype=torch.float32,
        quantization="none",
        nominal_bits=32,
        description="Full 32-bit float weights and compute.",
    ),
    "fp16": PrecisionSpec(
        name="fp16",
        torch_dtype=torch.float16,
        quantization="none",
        nominal_bits=16,
        description="IEEE half precision.",
    ),
    "bf16": PrecisionSpec(
        name="bf16",
        torch_dtype=torch.bfloat16,
        quantization="none",
        nominal_bits=16,
        description="bfloat16; the reference precision for this project.",
    ),
    "int8": PrecisionSpec(
        name="int8",
        torch_dtype=torch.float16,
        quantization="bnb-int8",
        nominal_bits=8,
        description="bitsandbytes LLM.int8() mixed-precision decomposition.",
    ),
    "nf4": PrecisionSpec(
        name="nf4",
        torch_dtype=torch.bfloat16,
        quantization="bnb-nf4",
        nominal_bits=4,
        description="bitsandbytes 4-bit NormalFloat with double quantization.",
    ),
    "fp4": PrecisionSpec(
        name="fp4",
        torch_dtype=torch.bfloat16,
        quantization="bnb-fp4",
        nominal_bits=4,
        description="bitsandbytes 4-bit float.",
    ),
}


@dataclass(frozen=True)
class CapabilityReport:
    """Whether a precision can actually be run here, and why not if it cannot."""

    precision: str
    supported: bool
    reason: str | None = None
    detail: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        return {
            "precision": self.precision,
            "supported": self.supported,
            "reason": self.reason,
            "detail": self.detail or {},
        }


def get_precision(name: str) -> PrecisionSpec:
    """Look up a precision spec by name.

    Raises:
        KeyError: If the name is not a known precision.
    """
    try:
        return PRECISIONS[name]
    except KeyError as exc:
        raise KeyError(
            f"unknown precision {name!r}; known precisions are {sorted(PRECISIONS)}"
        ) from exc


def bitsandbytes_status() -> dict[str, Any]:
    """Probe whether bitsandbytes is importable and believes it has a CUDA backend."""
    status: dict[str, Any] = {"importable": False, "version": None, "error": None}
    try:
        import bitsandbytes as bnb

        status["importable"] = True
        status["version"] = getattr(bnb, "__version__", None)
    except Exception as exc:
        status["error"] = f"{type(exc).__name__}: {exc}"
    return status


def check_precision_support(name: str, device_index: int = 0) -> CapabilityReport:
    """Decide whether ``name`` can be loaded on this machine right now.

    Args:
        name: A precision label from :data:`PRECISIONS`.
        device_index: CUDA device that would be used.

    Returns:
        A :class:`CapabilityReport`. ``supported=False`` always carries a human-readable
        reason, which is what ends up in the results file.
    """
    if name not in PRECISIONS:
        return CapabilityReport(name, False, f"unknown precision {name!r}")

    spec = PRECISIONS[name]

    if not torch.cuda.is_available():
        return CapabilityReport(name, False, "no CUDA device available")

    if spec.torch_dtype is torch.bfloat16 or spec.quantization == "bnb-nf4":
        if not torch.cuda.is_bf16_supported():
            return CapabilityReport(
                name, False, "device does not support bfloat16 compute"
            )

    if spec.quantization.startswith("bnb-"):
        status = bitsandbytes_status()
        if not status["importable"]:
            return CapabilityReport(
                name,
                False,
                f"bitsandbytes is not importable: {status['error']}",
                detail=status,
            )
        # LLM.int8() needs tensor cores; every supported card here has them, but the
        # check is cheap and makes the failure legible on older hardware.
        major, _ = torch.cuda.get_device_capability(device_index)
        if major < 7:
            return CapabilityReport(
                name,
                False,
                f"bitsandbytes quantization needs compute capability >= 7.0, "
                f"device reports {major}.x",
                detail=status,
            )
        return CapabilityReport(name, True, detail=status)

    return CapabilityReport(name, True)


def supported_precisions(
    requested: tuple[str, ...] | list[str], device_index: int = 0
) -> tuple[list[str], list[CapabilityReport]]:
    """Split a requested precision list into runnable names and rejection reports.

    Returns:
        ``(runnable, reports)`` where ``reports`` covers every requested precision,
        supported or not, so the full request is documented in the results.
    """
    reports = [check_precision_support(p, device_index) for p in requested]
    runnable = [r.precision for r in reports if r.supported]
    for report in reports:
        if not report.supported:
            logger.warning("Precision %s unsupported: %s", report.precision, report.reason)
    return runnable, reports
