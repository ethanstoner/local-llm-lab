"""Config parsing and validation.

The point of these tests is that a bad config fails at parse time. A typo that is only
noticed when the offending field is finally read has already cost a model load.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from src.utils.config import ConfigError, load_config

MINIMAL = """
experiment:
  name: unit_test
model:
  id: some/model
"""


def write(tmp_path: Path, body: str) -> Path:
    """Write a config file and return its path."""
    path = tmp_path / "config.yaml"
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return path


def test_minimal_config_gets_defaults(tmp_path: Path) -> None:
    config = load_config(write(tmp_path, MINIMAL))
    assert config.experiment.name == "unit_test"
    assert config.model.id == "some/model"
    assert config.benchmark.precisions == ("bf16",)
    assert config.monitoring.enabled is True
    assert config.source_path is not None


def test_lists_become_tuples(tmp_path: Path) -> None:
    config = load_config(
        write(
            tmp_path,
            MINIMAL
            + "benchmark:\n"
            + "  precisions: [bf16, nf4]\n"
            + "  context_lengths: [128, 512]\n",
        )
    )
    assert config.benchmark.precisions == ("bf16", "nf4")
    assert config.benchmark.context_lengths == (128, 512)


def test_unknown_key_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="unknown keys"):
        load_config(write(tmp_path, MINIMAL + "\nbenchmark:\n  precisionz: [bf16]\n"))


def test_unknown_section_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="unknown top-level sections"):
        load_config(write(tmp_path, MINIMAL + "\nnonsense:\n  a: 1\n"))


def test_missing_required_section(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="missing the required 'model' section"):
        load_config(write(tmp_path, "experiment:\n  name: x\n"))


def test_unknown_precision_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="unknown entries"):
        load_config(write(tmp_path, MINIMAL + "\nbenchmark:\n  precisions: [int3]\n"))


def test_single_new_token_is_rejected(tmp_path: Path) -> None:
    # Decode rate is (new_tokens - 1) / decode_time; one token has no rate to report.
    with pytest.raises(ConfigError, match="max_new_tokens must be >= 2"):
        load_config(write(tmp_path, MINIMAL + "\nbenchmark:\n  max_new_tokens: 1\n"))


def test_bad_attention_implementation(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="attn_implementation"):
        load_config(write(tmp_path, "experiment:\n  name: x\nmodel:\n  id: y\n  attn_implementation: magic\n"))


def test_bad_test_fraction(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="test_fraction"):
        load_config(write(tmp_path, MINIMAL + "\ninterpretability:\n  test_fraction: 1.0\n"))


def test_missing_file() -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config("does/not/exist.yaml")


def test_to_dict_is_json_friendly(tmp_path: Path) -> None:
    import json

    config = load_config(write(tmp_path, MINIMAL))
    payload = config.to_dict()
    assert isinstance(payload["benchmark"]["precisions"], list)
    json.dumps(payload)  # must not raise


def test_shipped_configs_all_parse() -> None:
    """Every config in configs/ must be valid; a broken one fails a real run."""
    from src.utils.io import repo_root

    paths = sorted((repo_root() / "configs").glob("*.yaml"))
    assert paths, "no configs found"
    for path in paths:
        config = load_config(path)
        assert config.experiment.name
        assert config.model.id
