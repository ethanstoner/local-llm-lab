"""Graceful CUDA out-of-memory handling.

A sweep over precisions and context lengths will find the edge of the card - that is
partly the point. When it does, the run that failed should be recorded as a failure with
its allocator state attached, and the sweep should carry on to the next cell. Losing
forty minutes of completed measurements because the last cell was too big is not
acceptable behaviour for a benchmark harness.
"""

from __future__ import annotations

import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

import torch

from src.monitoring.memory import allocator_snapshot, device_memory, empty_cache
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class GuardOutcome:
    """The result of running a guarded block."""

    status: str = "ok"
    error_type: str | None = None
    error_message: str | None = None
    traceback: str | None = None
    memory_at_failure: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """Whether the guarded block completed without error."""
        return self.status == "ok"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        return {
            "status": self.status,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "traceback": self.traceback,
            "memory_at_failure": self.memory_at_failure,
        }


def is_oom_error(exc: BaseException) -> bool:
    """Return whether an exception represents a CUDA allocation failure.

    ``torch.cuda.OutOfMemoryError`` covers the common case, but allocation failures also
    surface as a plain ``RuntimeError`` from some kernels and from bitsandbytes, so the
    message is checked as well.
    """
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    if isinstance(exc, RuntimeError):
        message = str(exc).lower()
        return "out of memory" in message or "cuda error: out of memory" in message
    return False


@contextmanager
def oom_guard(
    label: str,
    device: int = 0,
    reraise_non_oom: bool = True,
) -> Iterator[GuardOutcome]:
    """Run a block, converting OOM into a recorded outcome instead of a crash.

    Args:
        label: Human-readable name of the guarded work, used in log messages.
        device: CUDA device whose memory state is captured on failure.
        reraise_non_oom: When True (the default) any non-OOM exception propagates.
            A genuine bug should not be quietly filed as an experimental result.

    Yields:
        A :class:`GuardOutcome` that is populated by the time the block exits.

    Example:
        >>> with oom_guard("generate@8192") as outcome:  # doctest: +SKIP
        ...     result = model.generate(**inputs)
        >>> if not outcome.ok:  # doctest: +SKIP
        ...     record["status"] = outcome.status
    """
    outcome = GuardOutcome()
    try:
        yield outcome
    except BaseException as exc:  # noqa: BLE001 - deliberately broad, then re-narrowed
        if not is_oom_error(exc):
            outcome.status = "error"
            outcome.error_type = type(exc).__name__
            outcome.error_message = str(exc)
            outcome.traceback = traceback.format_exc()
            if reraise_non_oom:
                raise
            logger.error("%s failed: %s", label, exc)
            return

        outcome.status = "oom"
        outcome.error_type = type(exc).__name__
        outcome.error_message = str(exc).strip().splitlines()[0] if str(exc) else "OOM"
        outcome.memory_at_failure = {
            "allocator": allocator_snapshot(device),
            "device": device_memory(device),
        }
        logger.warning(
            "%s hit CUDA OOM (device had %.0f MiB free); recording and continuing",
            label,
            outcome.memory_at_failure["device"].get("free_mib") or -1,
        )
        empty_cache()


def release_memory(device: int = 0) -> dict[str, Any]:
    """Return cached CUDA memory to the driver and report what is free afterwards.

    Callers must drop their own references to the model and tokenizer first; this
    function cannot do that for them. It exists so that the collect/empty/synchronize
    sequence is written once and always in the same order.

    Args:
        device: CUDA device to report on.

    Returns:
        The device memory state after the cache has been released.
    """
    empty_cache()
    return device_memory(device)
