"""Run metadata capture.

Every result file this project writes carries a metadata block produced here. The point
is that a JSON file sitting in ``results/`` six months from now should be able to answer
"what hardware, what software, what commit, what settings" without anyone having to
remember. Numbers without this block are not reproducible, so they are not published.
"""

from __future__ import annotations

import importlib.metadata
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from src.utils.logging import get_logger

logger = get_logger(__name__)

#: Packages whose versions materially affect measured numbers.
_TRACKED_PACKAGES = (
    "torch",
    "transformers",
    "accelerate",
    "bitsandbytes",
    "safetensors",
    "numpy",
    "datasets",
    "nvidia-ml-py",
)


def utc_timestamp() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(UTC).isoformat(timespec="seconds")


def run_slug() -> str:
    """Return a filesystem-safe, sortable timestamp for naming run directories."""
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def package_versions() -> dict[str, str | None]:
    """Return installed versions of the packages that affect results."""
    versions: dict[str, str | None] = {}
    for name in _TRACKED_PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def git_commit(repo_root: Path | None = None) -> dict[str, Any]:
    """Return the current commit SHA and whether the working tree is dirty.

    A dirty tree means the code that produced a result is not fully captured by the SHA,
    which is worth knowing when a number looks surprising.
    """
    root = repo_root or Path(__file__).resolve().parents[2]
    info: dict[str, Any] = {"commit": None, "dirty": None, "branch": None}
    try:
        info["commit"] = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        info["branch"] = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        status = subprocess.check_output(
            ["git", "-C", str(root), "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        info["dirty"] = bool(status.strip())
    except (subprocess.CalledProcessError, FileNotFoundError, OSError) as exc:
        logger.debug("git metadata unavailable: %s", exc)
    return info


def gpu_info(device_index: int = 0) -> dict[str, Any]:
    """Describe the CUDA device, including how much VRAM other processes already hold.

    The baseline occupancy matters on a desktop machine: a browser or a game launcher
    sitting on two gigabytes changes what "peak VRAM headroom" means for a run.
    """
    if not torch.cuda.is_available():
        return {"available": False}

    props = torch.cuda.get_device_properties(device_index)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
    info: dict[str, Any] = {
        "available": True,
        "index": device_index,
        "name": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "total_memory_mib": round(props.total_memory / 1024**2, 1),
        "multi_processor_count": props.multi_processor_count,
        "free_memory_mib_before_load": round(free_bytes / 1024**2, 1),
        "used_by_other_processes_mib": round((total_bytes - free_bytes) / 1024**2, 1),
        "cuda_runtime": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
    }

    try:
        import pynvml

        pynvml.nvmlInit()
        pynvml.nvmlDeviceGetHandleByIndex(device_index)  # raises if the index is invalid
        driver = pynvml.nvmlSystemGetDriverVersion()
        info["driver_version"] = driver.decode() if isinstance(driver, bytes) else driver
        pynvml.nvmlShutdown()
    except Exception as exc:  # pragma: no cover - NVML is optional at import time
        logger.debug("NVML driver query failed: %s", exc)
        info["driver_version"] = None

    return info


def host_info() -> dict[str, Any]:
    """Describe the machine and interpreter."""
    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": sys.version.split()[0],
        "python_executable": sys.executable,
    }


def collect_metadata(
    experiment_name: str,
    extra: dict[str, Any] | None = None,
    device_index: int = 0,
) -> dict[str, Any]:
    """Assemble the full metadata block stamped into every result file.

    Args:
        experiment_name: Logical name of the experiment being run.
        extra: Additional fields to merge in (config dump, model details, ...).
        device_index: CUDA device to describe.

    Returns:
        A JSON-serialisable metadata dictionary.
    """
    meta: dict[str, Any] = {
        "experiment": experiment_name,
        "timestamp_utc": utc_timestamp(),
        "host": host_info(),
        "gpu": gpu_info(device_index),
        "packages": package_versions(),
        "git": git_commit(),
    }
    if extra:
        meta.update(extra)
    return meta
