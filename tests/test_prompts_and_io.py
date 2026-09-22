"""Prompt construction, result serialisation and run directories."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from src.benchmarks.prompts import (
    CONTINUATION_CUE,
    build_fixed_length_ids,
    load_corpus,
    max_supported_context,
)
from src.utils.io import create_run_dir, read_json, write_csv, write_json


class StubTokenizer:
    """Minimal whitespace tokenizer.

    Enough for the length arithmetic in prompt construction, and it keeps these tests
    free of any downloaded model.
    """

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {}

    def __call__(self, text: str, add_special_tokens: bool = True) -> dict[str, list[int]]:
        ids = []
        for word in text.split():
            ids.append(self.vocab.setdefault(word, len(self.vocab) + 1))
        return {"input_ids": ids}


@pytest.fixture
def tokenizer() -> StubTokenizer:
    return StubTokenizer()


@pytest.mark.parametrize("n_tokens", [1, 5, 64, 512, 4096])
def test_prompt_length_is_exact(tokenizer: StubTokenizer, n_tokens: int) -> None:
    """Context length is the independent variable, so it has to be exact, not close."""
    ids = build_fixed_length_ids(tokenizer, n_tokens)
    assert ids.shape == (n_tokens,)
    assert ids.dtype is torch.long


def test_prompt_is_deterministic(tokenizer: StubTokenizer) -> None:
    first = build_fixed_length_ids(tokenizer, 200)
    second = build_fixed_length_ids(StubTokenizer(), 200)
    assert torch.equal(first, second)


def test_long_prompt_tiles_the_corpus(tokenizer: StubTokenizer) -> None:
    corpus_length = len(tokenizer(load_corpus(), add_special_tokens=False)["input_ids"])
    ids = build_fixed_length_ids(tokenizer, corpus_length * 3)
    assert ids.numel() == corpus_length * 3


def test_prompt_ends_with_the_cue(tokenizer: StubTokenizer) -> None:
    cue = tokenizer(CONTINUATION_CUE, add_special_tokens=False)["input_ids"]
    ids = build_fixed_length_ids(tokenizer, 300)
    assert ids[-len(cue) :].tolist() == cue


def test_zero_length_prompt_rejected(tokenizer: StubTokenizer) -> None:
    with pytest.raises(ValueError, match="n_tokens must be >= 1"):
        build_fixed_length_ids(tokenizer, 0)


def test_corpus_is_present() -> None:
    assert len(load_corpus()) > 1000


class StubConfig:
    max_position_embeddings = 2048


def test_context_clamping() -> None:
    usable, reason = max_supported_context(StubConfig(), 1024)
    assert (usable, reason) == (1024, None)

    usable, reason = max_supported_context(StubConfig(), 8192)
    assert usable == 2048
    assert "exceeds" in reason


def test_context_clamping_without_a_limit() -> None:
    class NoLimit:
        pass

    assert max_supported_context(NoLimit(), 99999) == (99999, None)


def test_write_and_read_json(tmp_path: Path) -> None:
    path = write_json(tmp_path / "x.json", {"a": 1, "b": (2, 3)})
    assert read_json(path) == {"a": 1, "b": [2, 3]}


def test_write_json_converts_non_finite(tmp_path: Path) -> None:
    """NaN and Infinity are not valid JSON; null is honest about 'no measurement'."""
    path = write_json(tmp_path / "x.json", {"nan": float("nan"), "inf": float("inf"), "ok": 1.5})
    raw = path.read_text(encoding="utf-8")
    assert "NaN" not in raw and "Infinity" not in raw

    payload = json.loads(raw)
    assert payload["nan"] is None
    assert payload["inf"] is None
    assert payload["ok"] == 1.5


def test_write_json_leaves_no_temp_file(tmp_path: Path) -> None:
    write_json(tmp_path / "x.json", {"a": 1})
    assert [p.name for p in tmp_path.iterdir()] == ["x.json"]


def test_write_csv_infers_columns(tmp_path: Path) -> None:
    path = write_csv(tmp_path / "x.csv", [{"a": 1, "b": 2}, {"a": 3, "c": 4}])
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "a,b,c"
    assert lines[2] == "3,,4"


def test_write_csv_handles_non_finite(tmp_path: Path) -> None:
    """NaN becomes an empty field, not the literal text 'nan'.

    Read back through csv.reader rather than by string comparison: the csv module
    writes a lone empty field as `""` so the row is not an ambiguous blank line.
    """
    import csv

    path = write_csv(tmp_path / "x.csv", [{"a": float("nan")}])
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    assert rows[1] == [""]


def test_create_run_dir_is_unique(tmp_path: Path) -> None:
    first = create_run_dir(tmp_path, "exp")
    second = create_run_dir(tmp_path, "exp")
    assert first != second
    assert first.parent == second.parent == tmp_path / "exp"
