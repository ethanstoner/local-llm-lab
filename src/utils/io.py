"""Run directories and result serialisation.

One run produces one directory. Everything about that run - metadata, metrics, GPU
telemetry, log - lives inside it, so a result is a self-contained unit that can be
copied, diffed or deleted without hunting for companions.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from src.utils.env import run_slug
from src.utils.logging import get_logger

logger = get_logger(__name__)


def repo_root() -> Path:
    """Return the repository root, regardless of the caller's working directory."""
    return Path(__file__).resolve().parents[2]


def resolve_under_repo(path: str | Path) -> Path:
    """Resolve a possibly-relative config path against the repository root."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else repo_root() / candidate


def create_run_dir(output_root: str | Path, experiment_name: str) -> Path:
    """Create and return a fresh timestamped directory for one run.

    Args:
        output_root: Base results directory, relative to the repo root unless absolute.
        experiment_name: Used as both a grouping directory and part of the run name.

    Returns:
        Path to the created directory, of the form
        ``<output_root>/<experiment_name>/<timestamp>``.
    """
    base = resolve_under_repo(output_root) / experiment_name
    run_dir = base / run_slug()

    # Two runs launched inside the same second must not collide.
    suffix = 1
    while run_dir.exists():
        suffix += 1
        run_dir = base / f"{run_slug()}-{suffix}"

    run_dir.mkdir(parents=True, exist_ok=False)
    logger.info("Run directory: %s", run_dir)
    return run_dir


def _jsonable(obj: Any) -> Any:
    """Coerce dataclasses, tuples, paths and non-finite floats into JSON-safe values."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return _jsonable(asdict(obj))
    if isinstance(obj, Mapping):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, float):
        # NaN/Infinity are not valid JSON; null is honest about "no measurement".
        return obj if math.isfinite(obj) else None
    return obj


def write_json(path: Path, payload: Any) -> Path:
    """Write ``payload`` as pretty-printed JSON, creating parents as needed.

    The write goes to a temporary file first and is then moved into place, so an
    interrupted run cannot leave a half-written result behind.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(payload), handle, indent=2, sort_keys=False)
        handle.write("\n")
    tmp.replace(path)
    return path


def read_json(path: Path) -> Any:
    """Read a JSON file."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str] | None = None) -> Path:
    """Write a list of flat dictionaries as CSV.

    Args:
        path: Destination file.
        rows: Records to write. An empty sequence writes a header-only file when
            ``columns`` is supplied, and nothing at all otherwise.
        columns: Explicit column order; inferred from the union of keys when omitted.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows and not columns:
        logger.warning("No rows and no columns given for %s; skipping write", path)
        return path

    if columns is None:
        seen: list[str] = []
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.append(key)
        columns = seen

    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _jsonable(row.get(k)) for k in columns})
    tmp.replace(path)
    return path


def iter_run_dirs(output_root: str | Path, experiment_name: str | None = None) -> Iterable[Path]:
    """Yield run directories under ``output_root``, newest last.

    Args:
        output_root: Base results directory.
        experiment_name: Restrict to one experiment when given.
    """
    base = resolve_under_repo(output_root)
    if not base.is_dir():
        return []
    groups = [base / experiment_name] if experiment_name else sorted(
        p for p in base.iterdir() if p.is_dir()
    )
    runs: list[Path] = []
    for group in groups:
        if group.is_dir():
            runs.extend(sorted(p for p in group.iterdir() if p.is_dir()))
    return runs


def latest_run_dir(output_root: str | Path, experiment_name: str) -> Path | None:
    """Return the most recent run directory for an experiment, or None if there is none."""
    runs = list(iter_run_dirs(output_root, experiment_name))
    return runs[-1] if runs else None
