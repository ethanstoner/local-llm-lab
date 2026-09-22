"""PyTorch allocator accounting.

Complements :mod:`src.monitoring.gpu`. NVML reports what the *device* holds; the numbers
here report what PyTorch's caching allocator holds. Both are reported for every run
because the gap between them (CUDA context, cuBLAS workspaces, fragmentation, other
processes) is itself informative - on this machine it is consistently around 0.6-1.0 GiB.
"""

from __future__ import annotations

import gc
from typing import Any

import torch

from src.utils.logging import get_logger

logger = get_logger(__name__)

_MIB = 1024**2


def cuda_available() -> bool:
    """Return whether a CUDA device is usable."""
    return torch.cuda.is_available()


def reset_peak_stats(device: int | str = 0) -> None:
    """Reset PyTorch's peak-memory counters so the next region measures cleanly."""
    if cuda_available():
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.reset_accumulated_memory_stats(device)


def clear_cublas_workspaces() -> bool:
    """Free cuBLAS's per-stream workspaces.

    Worth its own function because of how much memory it unlocks. cuBLAS allocates a
    workspace (~8 MiB by default on this architecture) *through PyTorch's caching
    allocator*, and that allocation is held by C++ rather than by any Python object. The
    caching allocator can only return a segment to the driver when the whole segment is
    free, so those few megabytes pin entire multi-gigabyte segments: after unloading a
    3 GiB model this project measured 8.1 MiB allocated but 2946 MiB still reserved, and
    ``empty_cache`` alone could not shift it.

    Returns:
        True if the workspaces were cleared, False if this build of torch has no such
        private API (in which case the caller still gets a normal ``empty_cache``).
    """
    if not cuda_available():
        return False
    clear = getattr(torch._C, "_cuda_clearCublasWorkspaces", None)
    if clear is None:  # pragma: no cover - depends on the torch build
        logger.debug("torch._C._cuda_clearCublasWorkspaces is unavailable")
        return False
    clear()
    return True


def empty_cache(collect: bool = True, clear_workspaces: bool = True) -> None:
    """Release cached blocks back to the driver.

    Args:
        collect: Run a Python garbage collection first, so tensors held only by
            unreachable cycles are actually freed rather than merely uncached.
        clear_workspaces: Also drop cuBLAS workspaces. See
            :func:`clear_cublas_workspaces` for why this matters so much.
    """
    if collect:
        gc.collect()
    if not cuda_available():
        return
    if clear_workspaces:
        clear_cublas_workspaces()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def allocator_snapshot(device: int | str = 0) -> dict[str, float | None]:
    """Return current and peak allocator figures in MiB.

    Returns:
        A dictionary with ``allocated``, ``reserved``, ``max_allocated`` and
        ``max_reserved`` in MiB, or all-None when there is no CUDA device.
    """
    if not cuda_available():
        return {
            "allocated_mib": None,
            "reserved_mib": None,
            "max_allocated_mib": None,
            "max_reserved_mib": None,
        }
    return {
        "allocated_mib": round(torch.cuda.memory_allocated(device) / _MIB, 1),
        "reserved_mib": round(torch.cuda.memory_reserved(device) / _MIB, 1),
        "max_allocated_mib": round(torch.cuda.max_memory_allocated(device) / _MIB, 1),
        "max_reserved_mib": round(torch.cuda.max_memory_reserved(device) / _MIB, 1),
    }


def device_memory(device: int = 0) -> dict[str, float | None]:
    """Return free/used/total device memory in MiB as the driver sees it."""
    if not cuda_available():
        return {"free_mib": None, "used_mib": None, "total_mib": None}
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    return {
        "free_mib": round(free_bytes / _MIB, 1),
        "used_mib": round((total_bytes - free_bytes) / _MIB, 1),
        "total_mib": round(total_bytes / _MIB, 1),
    }


def memory_report(device: int = 0) -> dict[str, Any]:
    """Return both allocator and driver views of memory in one block."""
    return {"allocator": allocator_snapshot(device), "device": device_memory(device)}
