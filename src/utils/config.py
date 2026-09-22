"""Typed, validated experiment configuration.

Configs are YAML files parsed into frozen dataclasses. Validation is strict and happens
up front: an unknown key or an impossible value raises immediately, rather than forty
minutes into a sweep when the offending field is finally read.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar

import yaml

#: Precision identifiers the loader understands. Anything else is a config error.
VALID_PRECISIONS = ("fp32", "fp16", "bf16", "int8", "nf4", "fp4")

#: How per-prompt activations are reduced to a single vector.
VALID_AGGREGATIONS = ("last_token", "mean")

#: Attention backends a config may request. ``auto`` resolves per torch build - see
#: :mod:`src.models.attention` for why that matters on Windows.
VALID_ATTN_IMPLEMENTATIONS = (
    "auto",
    "sdpa",
    "sdpa_no_gqa",
    "sdpa_grouped_decode",
    "fp32_reference",
    "eager",
    "flash_attention_2",
)

T = TypeVar("T")


class ConfigError(ValueError):
    """Raised when a configuration file is malformed or internally inconsistent."""


@dataclass(frozen=True)
class ExperimentConfig:
    """Identity and reproducibility settings for a run."""

    name: str
    seed: int = 1234
    output_root: str = "results"
    notes: str = ""
    deterministic: bool = True


@dataclass(frozen=True)
class ModelConfig:
    """Which checkpoint to load and how to construct it.

    ``local_path`` points at a directory fetched by ``scripts/fetch_model.ps1``. When it
    is set the weights are loaded from disk and nothing is downloaded, but ``id`` is
    still what gets recorded in run metadata, so provenance survives either way.
    """

    id: str
    revision: str = "main"
    trust_remote_code: bool = False
    attn_implementation: str = "sdpa"
    tokenizer_id: str | None = None
    local_path: str | None = None

    def __post_init__(self) -> None:
        if self.attn_implementation not in VALID_ATTN_IMPLEMENTATIONS:
            raise ConfigError(
                f"model.attn_implementation must be one of "
                f"{list(VALID_ATTN_IMPLEMENTATIONS)}, got {self.attn_implementation!r}"
            )

    def resolve_source(self) -> tuple[str, bool]:
        """Return what to hand ``from_pretrained``, and whether it is a local directory.

        Raises:
            ConfigError: If ``local_path`` is set but does not exist. Falling back to the
                Hub silently would turn a typo into an unexpected multi-gigabyte
                download.
        """
        if not self.local_path:
            return self.id, False

        from src.utils.io import resolve_under_repo

        path = resolve_under_repo(self.local_path)
        if not path.is_dir():
            raise ConfigError(
                f"model.local_path does not exist: {path}. Fetch it first with "
                f"scripts/fetch_model.ps1 -Repo {self.id}"
            )
        return str(path), True


@dataclass(frozen=True)
class BenchmarkConfig:
    """The sweep grid and generation settings for Phase 1 / Phase 2."""

    precisions: tuple[str, ...] = ("bf16",)
    context_lengths: tuple[int, ...] = (128, 512, 2048)
    max_new_tokens: int = 128
    repeats: int = 3
    warmup: int = 1
    batch_size: int = 1

    def __post_init__(self) -> None:
        bad = [p for p in self.precisions if p not in VALID_PRECISIONS]
        if bad:
            raise ConfigError(
                f"benchmark.precisions contains unknown entries {bad}; "
                f"valid values are {list(VALID_PRECISIONS)}"
            )
        if not self.precisions:
            raise ConfigError("benchmark.precisions must not be empty")
        if any(c <= 0 for c in self.context_lengths):
            raise ConfigError("benchmark.context_lengths must all be positive")
        if self.max_new_tokens <= 1:
            # Decode throughput is (new_tokens - 1) / (total - ttft); one token has no rate.
            raise ConfigError("benchmark.max_new_tokens must be >= 2 to measure a decode rate")
        if self.repeats < 1:
            raise ConfigError("benchmark.repeats must be >= 1")
        if self.warmup < 0:
            raise ConfigError("benchmark.warmup must be >= 0")
        if self.batch_size < 1:
            raise ConfigError("benchmark.batch_size must be >= 1")


@dataclass(frozen=True)
class MonitoringConfig:
    """Background GPU telemetry settings."""

    enabled: bool = True
    sample_interval_s: float = 0.05

    def __post_init__(self) -> None:
        if self.sample_interval_s <= 0:
            raise ConfigError("monitoring.sample_interval_s must be positive")


@dataclass(frozen=True)
class EvaluationConfig:
    """Phase 2 quality comparison against a full-precision reference."""

    reference_precision: str = "bf16"
    compare_precisions: tuple[str, ...] = ("int8", "nf4")
    n_prompts: int = 32
    max_new_tokens: int = 64
    perplexity_tokens: int = 4096
    perplexity_stride: int = 512

    def __post_init__(self) -> None:
        for name, value in (
            ("reference_precision", self.reference_precision),
            *[("compare_precisions", p) for p in self.compare_precisions],
        ):
            if value not in VALID_PRECISIONS:
                raise ConfigError(f"evaluation.{name} has unknown precision {value!r}")
        if self.n_prompts < 1:
            raise ConfigError("evaluation.n_prompts must be >= 1")


@dataclass(frozen=True)
class InterpretabilityConfig:
    """Phase 4 / Phase 5 activation capture and direction analysis."""

    precision: str = "bf16"
    layers: str | tuple[int, ...] = "all"
    aggregation: str = "last_token"
    n_prompts_per_class: int = 100
    test_fraction: float = 0.3
    save_activations: bool = True
    harmful_source: str = "jbb"
    harmless_source: str = "jbb"
    run_refusal_behaviour_check: bool = True
    refusal_check_max_new_tokens: int = 24
    refusal_check_n_prompts: int = 32

    def __post_init__(self) -> None:
        if self.aggregation not in VALID_AGGREGATIONS:
            raise ConfigError(
                f"interpretability.aggregation must be one of {list(VALID_AGGREGATIONS)}, "
                f"got {self.aggregation!r}"
            )
        if isinstance(self.layers, str) and self.layers != "all":
            raise ConfigError("interpretability.layers must be 'all' or a list of ints")
        if not 0.0 < self.test_fraction < 1.0:
            raise ConfigError("interpretability.test_fraction must be strictly between 0 and 1")
        if self.precision not in VALID_PRECISIONS:
            raise ConfigError(f"interpretability.precision is unknown: {self.precision!r}")


@dataclass(frozen=True)
class InterventionConfig:
    """Phase 6 causal test of the Phase 5 direction.

    ``direction_run`` names the Phase 5 run whose fitted directions are used - either a
    run directory or ``"latest"``. The prompt split is rebuilt from this config's
    ``experiment.seed`` and ``interpretability`` section and checked against that run's
    recorded config, so the held-out prompts here are exactly the ones the direction
    never saw.
    """

    direction_run: str = "latest"
    direction_experiment: str = "phase5_refusal_direction"
    target_layer: int | None = None
    source_layers: tuple[int, ...] = ()
    n_random_controls: int = 3
    addition_coefficients: tuple[float, ...] = (1.0,)
    max_new_tokens: int = 32
    batch_size: int = 8
    include_bundled_prompts: bool = True
    measure_capability: bool = True
    max_perplexity_ratio: float = 1.05
    judge_model_id: str | None = None
    judge_local_path: str | None = None

    def __post_init__(self) -> None:
        if self.max_new_tokens < 1:
            raise ConfigError("intervention.max_new_tokens must be >= 1")
        if self.batch_size < 1:
            raise ConfigError("intervention.batch_size must be >= 1")
        if self.n_random_controls < 0:
            raise ConfigError("intervention.n_random_controls must be >= 0")
        if any(layer < 0 for layer in self.source_layers):
            raise ConfigError("intervention.source_layers must be non-negative block indices")
        if self.target_layer is not None and self.target_layer < 0:
            raise ConfigError("intervention.target_layer must be a non-negative block index")
        if self.max_perplexity_ratio < 1.0:
            raise ConfigError("intervention.max_perplexity_ratio must be >= 1.0")
        if any(c <= 0 for c in self.addition_coefficients):
            raise ConfigError("intervention.addition_coefficients must all be positive")


@dataclass(frozen=True)
class ComparisonConfig:
    """An interleaved comparison over attention backends, batch sizes and contexts.

    Unlike a sweep, which measures one configuration to completion before the next,
    every round visits every cell and alternates the backend order (ABBA), so slow drift
    in the machine's state - thermals, other applications - lands on both arms equally.
    """

    backends: tuple[str, ...] = ("sdpa_no_gqa",)
    batch_sizes: tuple[int, ...] = (1,)
    rounds: int = 3
    paging_threshold: float = 0.97
    fidelity_steps: int = 0
    fidelity_reference: str | None = None

    def __post_init__(self) -> None:
        bad = [b for b in self.backends if b not in VALID_ATTN_IMPLEMENTATIONS or b == "auto"]
        if bad:
            raise ConfigError(f"comparison.backends has invalid entries {bad}")
        if not self.backends:
            raise ConfigError("comparison.backends must not be empty")
        if len(set(self.backends)) != len(self.backends):
            raise ConfigError("comparison.backends must not repeat")
        if not self.batch_sizes or any(b < 1 for b in self.batch_sizes):
            raise ConfigError("comparison.batch_sizes must be positive")
        if self.rounds < 1:
            raise ConfigError("comparison.rounds must be >= 1")
        if self.fidelity_reference is not None and (
            self.fidelity_reference not in VALID_ATTN_IMPLEMENTATIONS or self.fidelity_reference == "auto"
        ):
            raise ConfigError(f"comparison.fidelity_reference is invalid: {self.fidelity_reference!r}")
        if self.fidelity_steps < 0:
            raise ConfigError("comparison.fidelity_steps must be >= 0")
        if not 0.5 < self.paging_threshold <= 1.0:
            raise ConfigError("comparison.paging_threshold must be in (0.5, 1]")


@dataclass(frozen=True)
class LabConfig:
    """A whole experiment configuration."""

    experiment: ExperimentConfig
    model: ModelConfig
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    interpretability: InterpretabilityConfig = field(default_factory=InterpretabilityConfig)
    intervention: InterventionConfig = field(default_factory=InterventionConfig)
    comparison: ComparisonConfig = field(default_factory=ComparisonConfig)
    source_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable copy, for embedding in result metadata."""
        return _asdict(self)


_SECTION_TYPES: dict[str, type] = {
    "experiment": ExperimentConfig,
    "model": ModelConfig,
    "benchmark": BenchmarkConfig,
    "monitoring": MonitoringConfig,
    "evaluation": EvaluationConfig,
    "interpretability": InterpretabilityConfig,
    "intervention": InterventionConfig,
    "comparison": ComparisonConfig,
}

#: Fields typed as tuples in the dataclasses but naturally written as YAML lists.
_TUPLE_FIELDS = {
    "precisions",
    "context_lengths",
    "compare_precisions",
    "source_layers",
    "addition_coefficients",
    "backends",
    "batch_sizes",
}


def _build_section(section: str, cls: type, payload: dict[str, Any]) -> Any:
    """Instantiate one config dataclass, rejecting unknown keys."""
    if not isinstance(payload, dict):
        raise ConfigError(f"section {section!r} must be a mapping, got {type(payload).__name__}")

    known = {f.name for f in fields(cls)}
    unknown = set(payload) - known
    if unknown:
        raise ConfigError(
            f"section {section!r} has unknown keys {sorted(unknown)}; "
            f"valid keys are {sorted(known)}"
        )

    coerced: dict[str, Any] = {}
    for key, value in payload.items():
        if key in _TUPLE_FIELDS and isinstance(value, list):
            coerced[key] = tuple(value)
        elif key == "layers" and isinstance(value, list):
            coerced[key] = tuple(int(v) for v in value)
        else:
            coerced[key] = value
    return cls(**coerced)


def load_config(path: str | Path) -> LabConfig:
    """Parse and validate a YAML experiment config.

    Args:
        path: Path to the YAML file.

    Returns:
        A validated :class:`LabConfig`.

    Raises:
        ConfigError: If the file is missing required sections, contains unknown keys,
            or holds values that cannot produce a runnable experiment.
    """
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    if not isinstance(raw, dict):
        raise ConfigError(f"{path} must contain a top-level mapping")

    unknown_sections = set(raw) - set(_SECTION_TYPES)
    if unknown_sections:
        raise ConfigError(
            f"{path} has unknown top-level sections {sorted(unknown_sections)}; "
            f"valid sections are {sorted(_SECTION_TYPES)}"
        )

    for required in ("experiment", "model"):
        if required not in raw:
            raise ConfigError(f"{path} is missing the required {required!r} section")

    sections = {
        name: _build_section(name, cls, raw.get(name, {}))
        for name, cls in _SECTION_TYPES.items()
    }
    return LabConfig(source_path=str(path), **sections)


def _asdict(obj: Any) -> Any:
    """Recursively convert dataclasses to plain JSON-friendly structures.

    ``dataclasses.asdict`` leaves tuples as tuples, which ``json`` handles but which then
    round-trip as lists; doing the conversion here keeps stored metadata stable.
    """
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _asdict(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, (list, tuple)):
        return [_asdict(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _asdict(v) for k, v in obj.items()}
    if isinstance(obj, Path):
        return str(obj)
    return obj
