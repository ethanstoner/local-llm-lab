"""Residual-stream interventions, against a tiny randomly-initialised model.

The key property under test is the one the causal experiment rests on: under
directional ablation the residual stream carries *no* component along the direction at
any layer or position, and when the context exits the model is bit-for-bit back to its
original behaviour.
"""

from __future__ import annotations

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from src.interpretability.hooks import find_layers
from src.interpretability.intervention import (
    ActivationAddition,
    DirectionalAblation,
    NoIntervention,
    ablation_sites,
    project_out,
    random_unit_direction,
    residual_component,
    residual_writers,
    unit,
)

HIDDEN = 32
LAYERS = 3


@pytest.fixture(scope="module")
def tiny_model() -> LlamaForCausalLM:
    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=HIDDEN,
        intermediate_size=64,
        num_hidden_layers=LAYERS,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    model = LlamaForCausalLM(config)
    model.eval()
    return model


@pytest.fixture
def input_ids() -> torch.Tensor:
    return torch.tensor([[1, 5, 9, 13, 17, 21, 25, 29]])


def _block_outputs(model: LlamaForCausalLM, input_ids: torch.Tensor) -> list[torch.Tensor]:
    captured: list[torch.Tensor] = []
    handles = [
        block.register_forward_hook(lambda _m, _a, out: captured.append(out[0] if isinstance(out, tuple) else out))
        for block in find_layers(model)
    ]
    try:
        with torch.no_grad():
            model(input_ids=input_ids, use_cache=False)
    finally:
        for h in handles:
            h.remove()
    return captured


def test_unit_and_random_direction() -> None:
    v = unit(torch.tensor([3.0, 4.0]))
    assert torch.allclose(v, torch.tensor([0.6, 0.8]))
    r = random_unit_direction(HIDDEN, seed=7)
    assert r.shape == (HIDDEN,)
    assert abs(float(r.norm()) - 1.0) < 1e-6
    assert torch.equal(r, random_unit_direction(HIDDEN, seed=7))
    assert not torch.equal(r, random_unit_direction(HIDDEN, seed=8))
    with pytest.raises(ValueError):
        unit(torch.zeros(4))


def test_project_out_removes_component() -> None:
    torch.manual_seed(1)
    hidden = torch.randn(2, 5, HIDDEN)
    r = random_unit_direction(HIDDEN, seed=3)
    ablated = project_out(hidden, r)
    assert (ablated @ r).abs().max() < 1e-5
    # The orthogonal part is untouched.
    orthogonal = hidden - (hidden @ r).unsqueeze(-1) * r
    assert torch.allclose(ablated, orthogonal, atol=1e-5)


def test_residual_writers_counts_embedding_and_sublayers(tiny_model: LlamaForCausalLM) -> None:
    assert len(residual_writers(tiny_model)) == 1 + 2 * LAYERS
    assert len(ablation_sites(tiny_model)) == 1 + 3 * LAYERS


def test_ablation_zeroes_direction_at_every_layer(
    tiny_model: LlamaForCausalLM, input_ids: torch.Tensor
) -> None:
    r = random_unit_direction(HIDDEN, seed=11)
    before = _block_outputs(tiny_model, input_ids)
    assert max(float((h @ r).abs().max()) for h in before) > 1e-2

    with DirectionalAblation(tiny_model, r):
        during = _block_outputs(tiny_model, input_ids)
    for hidden in during:
        assert float((hidden @ r).abs().max()) < 1e-4


def test_hooks_are_removed_on_exit(tiny_model: LlamaForCausalLM, input_ids: torch.Tensor) -> None:
    with torch.no_grad():
        reference = tiny_model(input_ids=input_ids).logits.clone()
    r = random_unit_direction(HIDDEN, seed=12)
    with DirectionalAblation(tiny_model, r):
        with torch.no_grad():
            changed = tiny_model(input_ids=input_ids).logits.clone()
    with torch.no_grad():
        after = tiny_model(input_ids=input_ids).logits
    assert not torch.allclose(reference, changed)
    assert torch.equal(reference, after)


def test_hooks_removed_even_on_error(tiny_model: LlamaForCausalLM) -> None:
    r = random_unit_direction(HIDDEN, seed=13)
    with pytest.raises(RuntimeError):
        with DirectionalAblation(tiny_model, r):
            raise RuntimeError("boom")
    for module in ablation_sites(tiny_model):
        assert not module._forward_hooks


def test_addition_shifts_projection_by_coefficient(
    tiny_model: LlamaForCausalLM, input_ids: torch.Tensor
) -> None:
    r = random_unit_direction(HIDDEN, seed=14)
    layer = 1
    base = _block_outputs(tiny_model, input_ids)[layer]
    with ActivationAddition(tiny_model, layer, 5.0 * r):
        shifted = _block_outputs(tiny_model, input_ids)[layer]
    delta = (shifted - base) @ r
    assert torch.allclose(delta, torch.full_like(delta, 5.0), atol=1e-4)


def test_addition_leaves_earlier_layers_alone(
    tiny_model: LlamaForCausalLM, input_ids: torch.Tensor
) -> None:
    r = random_unit_direction(HIDDEN, seed=15)
    base = _block_outputs(tiny_model, input_ids)
    with ActivationAddition(tiny_model, 2, 3.0 * r):
        shifted = _block_outputs(tiny_model, input_ids)
    assert torch.equal(base[0], shifted[0])
    assert torch.equal(base[1], shifted[1])


def test_addition_rejects_bad_layer(tiny_model: LlamaForCausalLM) -> None:
    with pytest.raises(ValueError):
        ActivationAddition(tiny_model, LAYERS, torch.ones(HIDDEN))


def test_no_intervention_is_identity(tiny_model: LlamaForCausalLM, input_ids: torch.Tensor) -> None:
    with torch.no_grad():
        reference = tiny_model(input_ids=input_ids).logits.clone()
        with NoIntervention():
            same = tiny_model(input_ids=input_ids).logits
    assert torch.equal(reference, same)


def test_ablation_applies_during_generation(tiny_model: LlamaForCausalLM, input_ids: torch.Tensor) -> None:
    """Hooks fire on every decode step, not only the prefill forward pass."""
    r = random_unit_direction(HIDDEN, seed=16)
    worst = 0.0

    def watch(_m: object, _a: object, out: object) -> None:
        nonlocal worst
        hidden = out[0] if isinstance(out, tuple) else out
        worst = max(worst, float((hidden @ r).abs().max()))

    last = find_layers(tiny_model)[-1]
    with DirectionalAblation(tiny_model, r):
        handle = last.register_forward_hook(watch)
        try:
            with torch.no_grad():
                tiny_model.generate(input_ids=input_ids, max_new_tokens=4, do_sample=False, pad_token_id=0)
        finally:
            handle.remove()
    assert worst < 1e-4


class _ToyTokenizer:
    """Just enough tokenizer for residual_component: a fixed chat template and ids."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):  # noqa: ANN001
        return messages[0]["content"]

    def __call__(self, texts, return_tensors="pt", padding=True, add_special_tokens=False):  # noqa: ANN001
        rows = [[(ord(c) % 60) + 1 for c in t] for t in texts]
        width = max(len(r) for r in rows)
        ids = torch.tensor([[0] * (width - len(r)) + r for r in rows])
        mask = torch.tensor([[0] * (width - len(r)) + [1] * len(r) for r in rows])
        return {"input_ids": ids, "attention_mask": mask}


def test_residual_component_falls_to_zero_under_ablation(tiny_model: LlamaForCausalLM) -> None:
    r = random_unit_direction(HIDDEN, seed=17)
    prompts = ["hello there", "abc"]
    intact = residual_component(tiny_model, _ToyTokenizer(), prompts, r, device="cpu")
    with DirectionalAblation(tiny_model, r):
        ablated = residual_component(tiny_model, _ToyTokenizer(), prompts, r, device="cpu")
    assert intact["mean_abs_cosine"] > 1e-3
    assert ablated["max_abs_cosine"] < 1e-5
