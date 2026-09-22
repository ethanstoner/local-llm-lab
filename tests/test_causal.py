"""Phase 6 bookkeeping: layer selection, intervals, split guard and config validation.

These are the pieces that decide *which* numbers get reported, so each is pinned down
independently of any model.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from src.interpretability.causal import (
    ConditionResult,
    distinct_ratio,
    select_causal_layer,
    trim_completion,
)
from src.interpretability.intervene import check_split_matches
from src.interpretability.stats import wilson_interval
from src.utils.config import ConfigError, load_config


def _row(layer: int, refusals: int, ratio: float | None, n: int = 30, induced: int | None = 25) -> dict:
    return {
        "layer": layer, "refusals": refusals, "n": n, "perplexity_ratio": ratio,
        "induced_refusals": induced, "induced_n": None if induced is None else 30,
        "harmless_baseline_rate": 0.05,
    }


def test_selection_prefers_lowest_refusal_within_budget() -> None:
    rows = [_row(16, 7, 1.002), _row(20, 5, 1.20), _row(18, 9, 1.01)]
    # Layer 20 refuses least but breaks the capability budget.
    assert select_causal_layer(rows, max_perplexity_ratio=1.05) == 16
    assert select_causal_layer(rows, max_perplexity_ratio=1.5) == 20


def test_selection_breaks_ties_on_cost_then_layer() -> None:
    rows = [_row(18, 7, 1.03), _row(16, 7, 1.01), _row(14, 7, 1.01)]
    assert select_causal_layer(rows, 1.05) == 14


def test_selection_excludes_layers_near_the_output() -> None:
    rows = [_row(16, 2, 0.99), _row(27, 0, 1.03)]
    assert select_causal_layer(rows, 1.05, max_layer_exclusive=22.4) == 16
    assert select_causal_layer(rows, 1.05, max_layer_exclusive=None) == 27


def test_selection_requires_the_direction_to_induce_refusal() -> None:
    # Layer 14 removes refusal best, but adding it does nothing above baseline.
    rows = [_row(14, 0, 1.01, induced=2), _row(16, 2, 0.99, induced=20)]
    assert select_causal_layer(rows, 1.05) == 16
    # Not measured is not the same as passing.
    assert select_causal_layer([_row(16, 2, 0.99, induced=None)], 1.05) is None


def test_selection_rejects_the_case_the_simplified_rule_got_wrong() -> None:
    """The first run: layer 27 won on a perplexity tie-break and its addition degenerated.

    Its measured induction (4/30 at 1x against a 2/30 baseline) is not significantly above
    baseline, and it sits past 80% of a 28-layer network - either criterion excludes it.
    """
    rows = [
        {**_row(14, 0, 1.0457, induced=20), "harmless_baseline_rate": 2 / 30},
        {**_row(16, 2, 0.9947, induced=19), "harmless_baseline_rate": 2 / 30},
        {**_row(27, 0, 1.0327, induced=4), "harmless_baseline_rate": 2 / 30},
    ]
    assert select_causal_layer(rows, 1.05, max_layer_exclusive=22.4) == 14
    assert select_causal_layer(rows, 1.05, max_layer_exclusive=None) == 14


def test_selection_treats_missing_cost_as_ineligible() -> None:
    assert select_causal_layer([_row(16, 0, None)], 1.05) is None
    assert select_causal_layer([], 1.05) is None


def test_wilson_interval_known_values() -> None:
    low, high = wilson_interval(0, 30)
    assert low == 0.0 and high == pytest.approx(0.1135, abs=1e-4)
    low, high = wilson_interval(30, 30)
    assert high == 1.0 and low == pytest.approx(0.8865, abs=1e-4)
    low, high = wilson_interval(15, 30)
    assert low == pytest.approx(1 - high, abs=1e-9)
    with pytest.raises(ValueError):
        wilson_interval(31, 30)


def test_trim_and_distinct() -> None:
    assert trim_completion([5, 6, 2, 7], {2}) == [5, 6]
    assert trim_completion([2, 5], {2}) == []
    assert distinct_ratio([1, 1, 1, 1]) == 0.25
    assert distinct_ratio([]) is None


def test_condition_row_flattens_scorers_and_interval() -> None:
    result = ConditionResult(
        condition="ablate:L16",
        prompt_set="bundled",
        n=10,
        refusals=0,
        nll={"self": (0.5, 0.4), "judge": (None, None)},
        mean_length=31.0,
        empty_rate=0.0,
        mean_distinct_ratio=0.8,
        extra={"source_layer": 16},
    )
    row = result.to_row()
    assert row["refusal_rate"] == 0.0
    assert row["refusal_ci_low"] == 0.0 and row["refusal_ci_high"] > 0.2
    assert row["mean_nll_self"] == 0.5 and row["mean_nll_judge"] is None
    assert row["source_layer"] == 16


class _FakeSource:
    def __init__(self, meta: dict) -> None:
        self.meta = meta


CONFIG = """
experiment:
  name: t
  seed: 1234
model:
  id: Qwen/Qwen2.5-7B-Instruct
interpretability:
  precision: bf16
  n_prompts_per_class: 100
  test_fraction: 0.3
  harmful_source: jbb
  harmless_source: jbb
"""


def _config(tmp_path: Path, body: str = CONFIG):
    path = tmp_path / "c.yaml"
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return load_config(path)


def _recorded(**overrides) -> dict:
    interp = {
        "precision": "bf16",
        "n_prompts_per_class": 100,
        "test_fraction": 0.3,
        "harmful_source": "jbb",
        "harmless_source": "jbb",
    }
    interp.update(overrides)
    return {
        "config": {
            "experiment": {"seed": 1234},
            "model": {"id": "Qwen/Qwen2.5-7B-Instruct"},
            "interpretability": interp,
        }
    }


def test_split_guard_accepts_matching_run(tmp_path: Path) -> None:
    check_split_matches(_config(tmp_path), _FakeSource(_recorded()))


def test_split_guard_rejects_leaky_run(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="test_fraction"):
        check_split_matches(_config(tmp_path), _FakeSource(_recorded(test_fraction=0.2)))
    other_seed = _recorded()
    other_seed["config"]["experiment"]["seed"] = 7
    with pytest.raises(ConfigError, match="seed"):
        check_split_matches(_config(tmp_path), _FakeSource(other_seed))


def test_intervention_section_parses_and_validates(tmp_path: Path) -> None:
    config = _config(
        tmp_path,
        CONFIG
        + """
intervention:
  source_layers: [4, 16]
  addition_coefficients: [0.5, 1.0]
  max_perplexity_ratio: 1.1
""",
    )
    assert config.intervention.source_layers == (4, 16)
    assert config.intervention.addition_coefficients == (0.5, 1.0)

    for bad in (
        "  addition_coefficients: [0.0]",
        "  source_layers: [-1]",
        "  max_perplexity_ratio: 0.9",
        "  max_relative_depth: 1.5",
        "  typo_key: 1",
    ):
        with pytest.raises(ConfigError):
            _config(tmp_path, CONFIG + "\nintervention:\n" + bad + "\n")
