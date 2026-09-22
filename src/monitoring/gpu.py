"""Background GPU telemetry sampling via NVML.

``torch.cuda.max_memory_allocated`` only sees PyTorch's own allocator. It cannot see the
CUDA context, the cuBLAS workspaces, other processes, or what the driver actually holds -
and on a desktop machine those are not rounding errors. This module samples the real
device from a background thread so that reported peak VRAM is the number ``nvidia-smi``
would have shown, and so that "memory over time" is a measurement rather than a guess.
"""

from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class GpuSample:
    """One instantaneous reading of the device."""

    t_rel_s: float
    memory_used_mib: float
    memory_total_mib: float
    gpu_util_pct: float | None
    memory_util_pct: float | None
    power_w: float | None
    temperature_c: float | None
    sm_clock_mhz: float | None


@dataclass(frozen=True)
class Marker:
    """A labelled point on the sampler's timeline, e.g. ``"generate:start"``."""

    t_rel_s: float
    label: str


class NvmlUnavailable(RuntimeError):
    """Raised when NVML cannot be initialised and sampling was explicitly required."""


def _decode(value: Any) -> Any:
    """NVML returns bytes on some builds and str on others."""
    return value.decode() if isinstance(value, bytes) else value


class GpuSampler:
    """Context manager that records device telemetry on a background thread.

    Example:
        >>> with GpuSampler(interval_s=0.05) as sampler:  # doctest: +SKIP
        ...     sampler.mark("generate:start")
        ...     model.generate(**inputs)
        ...     sampler.mark("generate:end")
        >>> sampler.summary()["peak_memory_used_mib"]  # doctest: +SKIP

    The sampler degrades gracefully: if NVML is missing it logs a warning, collects
    nothing, and reports ``available: False`` rather than taking the experiment down.
    """

    def __init__(
        self,
        device_index: int = 0,
        interval_s: float = 0.05,
        enabled: bool = True,
    ) -> None:
        """
        Args:
            device_index: CUDA device to sample.
            interval_s: Target seconds between samples. The actual interval is recorded
                per sample, so a stalled thread is visible in the data rather than hidden.
            enabled: When False the sampler is a no-op, for CPU-only test runs.
        """
        self.device_index = device_index
        self.interval_s = interval_s
        self.enabled = enabled
        self.samples: list[GpuSample] = []
        self.markers: list[Marker] = []
        self.available = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._handle: Any = None
        self._pynvml: Any = None
        self._t0: float = 0.0
        self._error: str | None = None

    # -- lifecycle ---------------------------------------------------------------

    def __enter__(self) -> GpuSampler:
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.stop()

    def start(self) -> None:
        """Initialise NVML and begin sampling."""
        if not self.enabled:
            logger.debug("GPU sampling disabled by config")
            return
        try:
            import pynvml

            pynvml.nvmlInit()
            self._pynvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
            self.available = True
        except Exception as exc:
            self._error = str(exc)
            logger.warning("NVML unavailable, GPU telemetry will be empty: %s", exc)
            return

        self._t0 = time.perf_counter()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="gpu-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop sampling and release NVML."""
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=5.0)
            self._thread = None
        if self._pynvml is not None:
            try:
                self._pynvml.nvmlShutdown()
            except Exception as exc:  # pragma: no cover - shutdown is best effort
                logger.debug("nvmlShutdown failed: %s", exc)
            self._pynvml = None
            self._handle = None

    # -- recording ---------------------------------------------------------------

    def mark(self, label: str) -> float:
        """Record a labelled instant on the timeline.

        Returns:
            The marker's time in seconds relative to sampler start.
        """
        t_rel = time.perf_counter() - self._t0 if self._t0 else 0.0
        self.markers.append(Marker(t_rel_s=t_rel, label=label))
        return t_rel

    def _read_once(self) -> GpuSample | None:
        """Take a single NVML reading, tolerating per-field failures."""
        nvml = self._pynvml
        if nvml is None or self._handle is None:
            return None

        try:
            mem = nvml.nvmlDeviceGetMemoryInfo(self._handle)
        except Exception as exc:  # pragma: no cover - transient driver errors
            logger.debug("NVML memory read failed: %s", exc)
            return None

        def _try(fn: Any, *args: Any) -> Any:
            try:
                return fn(*args)
            except Exception:
                return None

        util = _try(nvml.nvmlDeviceGetUtilizationRates, self._handle)
        power_mw = _try(nvml.nvmlDeviceGetPowerUsage, self._handle)
        temp = _try(nvml.nvmlDeviceGetTemperature, self._handle, nvml.NVML_TEMPERATURE_GPU)
        clock = _try(nvml.nvmlDeviceGetClockInfo, self._handle, nvml.NVML_CLOCK_SM)

        return GpuSample(
            t_rel_s=time.perf_counter() - self._t0,
            memory_used_mib=mem.used / 1024**2,
            memory_total_mib=mem.total / 1024**2,
            gpu_util_pct=float(util.gpu) if util is not None else None,
            memory_util_pct=float(util.memory) if util is not None else None,
            power_w=power_mw / 1000.0 if power_mw is not None else None,
            temperature_c=float(temp) if temp is not None else None,
            sm_clock_mhz=float(clock) if clock is not None else None,
        )

    def _loop(self) -> None:
        """Sampling thread body."""
        while not self._stop.is_set():
            sample = self._read_once()
            if sample is not None:
                self.samples.append(sample)
            # wait() rather than sleep() so stop() is responsive.
            self._stop.wait(self.interval_s)

    # -- output ------------------------------------------------------------------

    def to_rows(self) -> list[dict[str, Any]]:
        """Return the samples as flat dictionaries, ready for CSV."""
        return [asdict(s) for s in self.samples]

    def marker_rows(self) -> list[dict[str, Any]]:
        """Return the markers as flat dictionaries."""
        return [asdict(m) for m in self.markers]

    def summary(self) -> dict[str, Any]:
        """Aggregate the collected samples.

        Returns:
            Peak and mean statistics, or a record explaining why nothing was collected.
        """
        if not self.samples:
            return {
                "available": self.available,
                "n_samples": 0,
                "error": self._error,
            }

        used = [s.memory_used_mib for s in self.samples]
        utils = [s.gpu_util_pct for s in self.samples if s.gpu_util_pct is not None]
        powers = [s.power_w for s in self.samples if s.power_w is not None]
        temps = [s.temperature_c for s in self.samples if s.temperature_c is not None]
        duration = self.samples[-1].t_rel_s - self.samples[0].t_rel_s

        return {
            "available": True,
            "n_samples": len(self.samples),
            "duration_s": round(duration, 4),
            "effective_interval_s": round(duration / max(len(self.samples) - 1, 1), 5),
            "peak_memory_used_mib": round(max(used), 1),
            "min_memory_used_mib": round(min(used), 1),
            "final_memory_used_mib": round(used[-1], 1),
            "memory_total_mib": round(self.samples[0].memory_total_mib, 1),
            "mean_gpu_util_pct": round(sum(utils) / len(utils), 2) if utils else None,
            "max_gpu_util_pct": round(max(utils), 2) if utils else None,
            "mean_power_w": round(sum(powers) / len(powers), 2) if powers else None,
            "max_power_w": round(max(powers), 2) if powers else None,
            "max_temperature_c": round(max(temps), 1) if temps else None,
        }


@dataclass
class TelemetryBundle:
    """A sampler's full output, ready to be written next to a run's metrics."""

    summary: dict[str, Any] = field(default_factory=dict)
    samples: list[dict[str, Any]] = field(default_factory=list)
    markers: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_sampler(cls, sampler: GpuSampler) -> TelemetryBundle:
        """Snapshot a sampler into a serialisable bundle."""
        return cls(
            summary=sampler.summary(),
            samples=sampler.to_rows(),
            markers=sampler.marker_rows(),
        )
