"""OOM handling and precision capability probing."""

from __future__ import annotations

import pytest
import torch

from src.models.oom import is_oom_error, oom_guard
from src.models.registry import (
    PRECISIONS,
    bitsandbytes_status,
    check_precision_support,
    get_precision,
    supported_precisions,
)


def test_is_oom_error_recognises_torch_error() -> None:
    assert is_oom_error(torch.cuda.OutOfMemoryError("CUDA out of memory."))


def test_is_oom_error_recognises_runtime_message() -> None:
    """bitsandbytes and some kernels raise a plain RuntimeError for allocation failures."""
    assert is_oom_error(RuntimeError("CUDA error: out of memory"))
    assert is_oom_error(RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"))


def test_is_oom_error_rejects_other_errors() -> None:
    assert not is_oom_error(RuntimeError("shape mismatch"))
    assert not is_oom_error(ValueError("nope"))


def test_oom_guard_records_oom_and_continues() -> None:
    with oom_guard("unit-test") as outcome:
        raise torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 40.00 GiB")

    assert outcome.status == "oom"
    assert not outcome.ok
    assert "out of memory" in (outcome.error_message or "").lower()
    assert "allocator" in outcome.memory_at_failure


def test_oom_guard_leaves_success_alone() -> None:
    with oom_guard("unit-test") as outcome:
        value = 1 + 1
    assert outcome.ok
    assert outcome.status == "ok"
    assert value == 2


def test_oom_guard_reraises_real_bugs() -> None:
    """A genuine bug must not be filed away as an experimental result."""
    with pytest.raises(ValueError, match="a real bug"), oom_guard("unit-test"):
        raise ValueError("a real bug")


def test_oom_guard_can_swallow_non_oom_when_asked() -> None:
    with oom_guard("unit-test", reraise_non_oom=False) as outcome:
        raise ValueError("recorded instead")
    assert outcome.status == "error"
    assert outcome.error_type == "ValueError"
    assert outcome.traceback is not None


def test_get_precision_unknown() -> None:
    with pytest.raises(KeyError, match="unknown precision"):
        get_precision("int3")


def test_every_precision_has_a_spec() -> None:
    for name, spec in PRECISIONS.items():
        assert spec.name == name
        assert spec.description
        assert spec.quantization in ("none", "bnb-int8", "bnb-nf4", "bnb-fp4")


def test_unknown_precision_is_unsupported_not_crashing() -> None:
    report = check_precision_support("int3")
    assert not report.supported
    assert "unknown precision" in (report.reason or "")


def test_capability_report_always_explains_refusal() -> None:
    """Whatever this machine supports, an unsupported verdict must carry a reason."""
    for name in PRECISIONS:
        report = check_precision_support(name)
        if not report.supported:
            assert report.reason, f"{name} was rejected without a reason"


def test_supported_precisions_splits_the_request() -> None:
    runnable, reports = supported_precisions(["bf16", "int3"])
    assert "int3" not in runnable
    # Every requested precision is documented, supported or not.
    assert {r.precision for r in reports} == {"bf16", "int3"}


def test_bitsandbytes_status_is_structured() -> None:
    status = bitsandbytes_status()
    assert set(status) == {"importable", "version", "error"}
    if status["importable"]:
        assert status["version"]
    else:
        assert status["error"]


def test_clear_cublas_workspaces_is_safe_to_call() -> None:
    """Must never raise, whatever the torch build offers.

    This is what lets a multi-gigabyte model actually be returned to the driver between
    precisions: cuBLAS workspaces are allocated through PyTorch's caching allocator and
    pin whole segments, so `empty_cache` alone leaves gigabytes reserved.
    """
    from src.monitoring.memory import clear_cublas_workspaces, empty_cache

    result = clear_cublas_workspaces()
    assert isinstance(result, bool)
    if torch.cuda.is_available():
        assert result, "this torch build should expose _cuda_clearCublasWorkspaces"
    empty_cache()


@pytest.mark.gpu
def test_empty_cache_returns_reserved_memory_to_the_driver() -> None:
    """A large allocation must be fully released, not left reserved by the allocator."""
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")

    from src.monitoring.memory import allocator_snapshot, empty_cache

    empty_cache()
    block = torch.empty(256 * 1024 * 1024 // 2, dtype=torch.float16, device="cuda")
    torch.matmul(block[:1024].view(32, 32), block[:1024].view(32, 32))  # forces a cuBLAS workspace
    del block
    empty_cache()

    assert allocator_snapshot()["reserved_mib"] == 0.0
