"""Model construction with measured load cost.

One entry point builds every precision variant, so that "bf16 vs nf4" differs in exactly
one place and not in a dozen scattered ``from_pretrained`` calls. Loading is itself
measured: wall time, peak device memory during construction, and the settled idle
footprint afterwards. Those three numbers answer different questions and are reported
separately rather than collapsed into one "VRAM" figure.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.models.attention import (
    backend_report,
    recommended_attn_implementation,
    register_attention_backends,
)
from src.models.registry import PrecisionSpec, get_precision
from src.monitoring.gpu import GpuSampler
from src.monitoring.memory import allocator_snapshot, device_memory, empty_cache, reset_peak_stats
from src.utils.config import ModelConfig
from src.utils.logging import get_logger

logger = get_logger(__name__)

_MIB = 1024**2


def _dtype_kwarg_name() -> str:
    """Return the keyword ``from_pretrained`` currently uses for the compute dtype.

    Transformers renamed ``torch_dtype`` to ``dtype`` in 4.56. Both are accepted in 4.57
    but only one is accepted either side of that window, so the name is resolved from
    the installed version rather than assumed.
    """
    from packaging.version import Version
    from transformers import __version__ as transformers_version

    return "dtype" if Version(transformers_version) >= Version("4.56.0") else "torch_dtype"


@dataclass
class LoadMetrics:
    """Everything measured about constructing the model."""

    load_time_s: float
    baseline_device_used_mib: float | None
    peak_device_used_mib: float | None
    idle_device_used_mib: float | None
    device_delta_mib: float | None
    weights_mib: float | None
    peak_allocator_mib: float | None
    allocator_after_load: dict[str, Any] = field(default_factory=dict)
    param_count: int | None = None
    dtype_histogram: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        return {
            "load_time_s": round(self.load_time_s, 4),
            "baseline_device_used_mib": self.baseline_device_used_mib,
            "peak_device_used_mib": self.peak_device_used_mib,
            "idle_device_used_mib": self.idle_device_used_mib,
            "device_delta_mib": self.device_delta_mib,
            "weights_mib": self.weights_mib,
            "peak_allocator_mib": self.peak_allocator_mib,
            "allocator_after_load": self.allocator_after_load,
            "param_count": self.param_count,
            "dtype_histogram": self.dtype_histogram,
        }


@dataclass
class LoadedModel:
    """A constructed model plus the measurements taken while constructing it."""

    model: Any
    tokenizer: Any
    precision: str
    spec: PrecisionSpec
    model_id: str
    revision: str
    resolved_commit: str | None
    metrics: LoadMetrics
    config_summary: dict[str, Any]
    device: str
    #: What was handed to ``from_pretrained``: a Hub id or a local directory.
    source: str = ""
    #: The attention implementation actually used, after resolving ``auto``.
    attn_implementation: str = ""

    @property
    def n_layers(self) -> int:
        """Number of transformer blocks in the loaded model."""
        return int(self.model.config.num_hidden_layers)

    @property
    def hidden_size(self) -> int:
        """Residual stream width."""
        return int(self.model.config.hidden_size)

    def describe(self) -> dict[str, Any]:
        """Return a metadata block describing the model, for result files."""
        return {
            "model_id": self.model_id,
            "revision": self.revision,
            "source": self.source,
            "resolved_commit": self.resolved_commit,
            "precision": self.precision,
            "precision_description": self.spec.description,
            "quantization": self.spec.quantization,
            "compute_dtype": str(self.spec.torch_dtype).replace("torch.", ""),
            "device": self.device,
            "attn_implementation": self.attn_implementation,
            "attention_backends": backend_report(),
            "config": self.config_summary,
            "load": self.metrics.to_dict(),
        }


def _dtype_histogram(model: Any) -> dict[str, int]:
    """Count parameters by storage dtype.

    For quantized models this is the honest picture: the bulk of the weights sit in
    ``uint8`` while norms and embeddings stay in the compute dtype.
    """
    histogram: dict[str, int] = {}
    for param in model.parameters():
        key = str(param.dtype).replace("torch.", "")
        histogram[key] = histogram.get(key, 0) + param.numel()
    return histogram


def _quantization_config(spec: PrecisionSpec) -> BitsAndBytesConfig | None:
    """Build the bitsandbytes config for a precision, or None for unquantized modes."""
    if spec.quantization == "none":
        return None
    if spec.quantization == "bnb-int8":
        return BitsAndBytesConfig(load_in_8bit=True)
    if spec.quantization in ("bnb-nf4", "bnb-fp4"):
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4" if spec.quantization == "bnb-nf4" else "fp4",
            bnb_4bit_compute_dtype=spec.torch_dtype,
            bnb_4bit_use_double_quant=True,
        )
    raise ValueError(f"unhandled quantization mode {spec.quantization!r}")


def _config_summary(config: Any) -> dict[str, Any]:
    """Extract the architecture fields worth recording alongside results."""
    keys = (
        "model_type",
        "num_hidden_layers",
        "hidden_size",
        "intermediate_size",
        "num_attention_heads",
        "num_key_value_heads",
        "vocab_size",
        "max_position_embeddings",
        "rope_theta",
        "tie_word_embeddings",
    )
    return {k: getattr(config, k, None) for k in keys}


def load_tokenizer(model_cfg: ModelConfig) -> Any:
    """Load the tokenizer for a model config.

    Left padding is set because batched generation must align the final prompt token
    across the batch; with right padding the "last token" is padding, which silently
    corrupts both generation and activation capture.
    """
    source, is_local = model_cfg.resolve_source()
    kwargs: dict[str, Any] = {"trust_remote_code": model_cfg.trust_remote_code}
    if not is_local:
        kwargs["revision"] = model_cfg.revision

    tokenizer = AutoTokenizer.from_pretrained(model_cfg.tokenizer_id or source, **kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def load_model(
    model_cfg: ModelConfig,
    precision: str,
    device_index: int = 0,
    monitor: bool = True,
    sample_interval_s: float = 0.05,
) -> LoadedModel:
    """Load a causal LM at a given precision, measuring the cost of doing so.

    Args:
        model_cfg: Which checkpoint and how to construct it.
        precision: A key from :data:`src.models.registry.PRECISIONS`. Callers are
            expected to have checked support first with ``check_precision_support``.
        device_index: Target CUDA device.
        monitor: Sample NVML during the load to capture true peak device memory.
        sample_interval_s: Sampling period when ``monitor`` is set.

    Returns:
        A :class:`LoadedModel`.

    Raises:
        torch.cuda.OutOfMemoryError: If the weights do not fit. Callers running a sweep
            should wrap this in :func:`src.models.oom.oom_guard`.
    """
    spec = get_precision(precision)
    device = f"cuda:{device_index}" if torch.cuda.is_available() else "cpu"

    empty_cache()
    reset_peak_stats(device_index)
    baseline = device_memory(device_index)["used_mib"]

    source, is_local = model_cfg.resolve_source()
    tokenizer = load_tokenizer(model_cfg)

    register_attention_backends()
    attn_implementation = model_cfg.attn_implementation
    if attn_implementation == "auto":
        attn_implementation = recommended_attn_implementation()
        logger.info("attn_implementation=auto resolved to %r", attn_implementation)

    common: dict[str, Any] = {"trust_remote_code": model_cfg.trust_remote_code}
    if not is_local:
        common["revision"] = model_cfg.revision

    config = AutoConfig.from_pretrained(source, **common)

    kwargs: dict[str, Any] = {
        **common,
        "attn_implementation": attn_implementation,
        _dtype_kwarg_name(): spec.torch_dtype,
    }
    quant_config = _quantization_config(spec)
    if quant_config is not None:
        kwargs["quantization_config"] = quant_config
        # bitsandbytes modules must be placed as they are built; device_map does that.
        kwargs["device_map"] = {"": device_index}
    elif torch.cuda.is_available():
        kwargs["device_map"] = {"": device_index}

    logger.info(
        "Loading %s at %s (%s) onto %s%s",
        model_cfg.id,
        precision,
        spec.description,
        device,
        " from local files" if is_local else " from the Hub",
    )

    sampler = GpuSampler(device_index=device_index, interval_s=sample_interval_s, enabled=monitor)
    sampler.start()
    sampler.mark("load:start")
    start = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(source, **kwargs)
    model.eval()
    if torch.cuda.is_available():
        torch.cuda.synchronize(device_index)
    load_time = time.perf_counter() - start
    sampler.mark("load:end")
    sampler.stop()

    load_summary = sampler.summary()
    peak_device = load_summary.get("peak_memory_used_mib")
    idle_device = device_memory(device_index)["used_mib"]
    allocator = allocator_snapshot(device_index)

    try:
        weights_mib = round(model.get_memory_footprint() / _MIB, 1)
    except Exception as exc:  # pragma: no cover - some quantized paths lack the helper
        logger.debug("get_memory_footprint unavailable: %s", exc)
        weights_mib = None

    metrics = LoadMetrics(
        load_time_s=load_time,
        baseline_device_used_mib=baseline,
        peak_device_used_mib=peak_device,
        idle_device_used_mib=idle_device,
        device_delta_mib=(
            round(idle_device - baseline, 1)
            if idle_device is not None and baseline is not None
            else None
        ),
        weights_mib=weights_mib,
        peak_allocator_mib=allocator["max_reserved_mib"],
        allocator_after_load=allocator,
        param_count=sum(p.numel() for p in model.parameters()),
        dtype_histogram=_dtype_histogram(model),
    )

    resolved_commit = getattr(config, "_commit_hash", None)

    logger.info(
        "Loaded in %.2fs | weights %.0f MiB | device idle %.0f MiB (baseline %.0f MiB)",
        load_time,
        weights_mib or -1,
        idle_device or -1,
        baseline or -1,
    )

    return LoadedModel(
        model=model,
        tokenizer=tokenizer,
        precision=precision,
        spec=spec,
        model_id=model_cfg.id,
        revision=model_cfg.revision,
        resolved_commit=resolved_commit,
        metrics=metrics,
        config_summary=_config_summary(config),
        device=device,
        source=source,
        attn_implementation=attn_implementation,
    )
