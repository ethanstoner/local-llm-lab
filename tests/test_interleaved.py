"""The interleaved comparison's bookkeeping, and switching backends in place."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
import torch

from src.benchmarks.interleaved import first_divergence, round_order, summarise
from src.models.attention import set_attn_implementation
from src.utils.config import ConfigError, load_config


def test_round_order_alternates() -> None:
    assert round_order(["a", "b"], 0) == ["a", "b"]
    assert round_order(["a", "b"], 1) == ["b", "a"]
    assert round_order(["a", "b"], 2) == ["a", "b"]


def test_first_divergence() -> None:
    a = torch.tensor([[1, 2, 3, 4]])
    assert first_divergence(a, a.clone()) is None
    assert first_divergence(a, torch.tensor([[1, 2, 9, 4]])) == 2


def _row(backend: str, rnd: int, tok_s: float, ctx: int = 128) -> dict:
    return {
        "status": "ok", "backend": backend, "round": rnd, "batch_size": 1, "context_length": ctx,
        "decode_tok_s": tok_s, "ttft_s": 0.1, "peak_allocated_mib": 100.0, "paging_suspected": False,
    }


def test_summary_uses_paired_ratios() -> None:
    # Round 1 is uniformly slow (drift). A ratio of medians would still be ~2, but the
    # paired ratios are exactly 2 in every round, which is the point of pairing.
    rows = [_row("a", 0, 10.0), _row("b", 0, 20.0), _row("a", 1, 5.0), _row("b", 1, 10.0),
            _row("a", 2, 10.0), _row("b", 2, 20.0)]
    (summary,) = summarise(rows, ["a", "b"])
    assert summary["b_vs_a_ratio_median"] == 2.0
    assert summary["b_vs_a_ratio_min"] == 2.0 and summary["b_vs_a_ratio_max"] == 2.0
    assert summary["b_vs_a_n_pairs"] == 3
    assert summary["a_decode_tok_s_median"] == 10.0


def test_summary_skips_failed_rows() -> None:
    rows = [_row("a", 0, 10.0), {**_row("b", 0, 0.0), "status": "oom"}]
    (summary,) = summarise(rows, ["a", "b"])
    assert "b_decode_tok_s_median" not in summary
    assert "b_vs_a_ratio_median" not in summary


def test_set_attn_implementation_switches_the_forward() -> None:
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    config = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64)
    config._attn_implementation = "eager"
    model = LlamaForCausalLM(config).eval()
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    with torch.no_grad():
        eager = model(input_ids=ids).logits
        set_attn_implementation(model, "sdpa_grouped_decode")
        assert model.model.layers[0].self_attn.config._attn_implementation == "sdpa_grouped_decode"
        grouped = model(input_ids=ids).logits
    assert torch.allclose(eager, grouped, atol=1e-5)
    with pytest.raises(ValueError):
        set_attn_implementation(model, "not_a_backend")


def _config(tmp_path: Path, section: str):
    path = tmp_path / "c.yaml"
    path.write_text(textwrap.dedent("""
        experiment:
          name: t
        model:
          id: m
        """) + section, encoding="utf-8")
    return load_config(path)


def test_comparison_section_parses(tmp_path: Path) -> None:
    config = _config(tmp_path, "comparison:\n  backends: [sdpa_no_gqa, sdpa_grouped_decode]\n  batch_sizes: [1, 8]\n")
    assert config.comparison.backends == ("sdpa_no_gqa", "sdpa_grouped_decode")
    assert config.comparison.batch_sizes == (1, 8)


@pytest.mark.parametrize(
    "bad",
    [
        "  backends: [auto]",
        "  backends: [sdpa, sdpa]",
        "  backends: [nope]",
        "  batch_sizes: [0]",
        "  rounds: 0",
        "  paging_threshold: 1.5",
    ],
)
def test_comparison_section_rejects(tmp_path: Path, bad: str) -> None:
    with pytest.raises(ConfigError):
        _config(tmp_path, "comparison:\n" + bad + "\n")
