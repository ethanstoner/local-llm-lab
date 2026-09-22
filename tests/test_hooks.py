"""Activation capture, against a tiny randomly-initialised model.

The model is built from a config in-process: three layers, 32 hidden units, no
downloaded weights. That keeps the suite fast and network-free while still exercising
the real hook path against a real transformer implementation.
"""

from __future__ import annotations

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from src.interpretability.hooks import (
    ActivationRecorder,
    aggregate,
    find_layers,
    load_activations,
    resolve_layer_indices,
    save_activations,
)

HIDDEN = 32
LAYERS = 3


@pytest.fixture(scope="module")
def tiny_model() -> LlamaForCausalLM:
    """A three-layer Llama with random weights, on CPU."""
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


def test_find_layers(tiny_model: LlamaForCausalLM) -> None:
    assert len(find_layers(tiny_model)) == LAYERS


def test_find_layers_rejects_unknown_architecture() -> None:
    with pytest.raises(AttributeError, match="could not locate"):
        find_layers(torch.nn.Linear(2, 2))


def test_resolve_layer_indices() -> None:
    assert resolve_layer_indices(4, "all") == [0, 1, 2, 3]
    assert resolve_layer_indices(4, [0, 2]) == [0, 2]
    assert resolve_layer_indices(4, [-1]) == [3]
    assert resolve_layer_indices(4, [2, 2, 0]) == [0, 2]


def test_resolve_layer_indices_out_of_range() -> None:
    with pytest.raises(ValueError, match="out of range"):
        resolve_layer_indices(4, [9])
    with pytest.raises(ValueError, match="must be 'all'"):
        resolve_layer_indices(4, "last")


def test_aggregate_last_token() -> None:
    hidden = torch.arange(12, dtype=torch.float32).reshape(1, 4, 3)
    assert aggregate(hidden, None, "last_token").tolist() == [[9.0, 10.0, 11.0]]


def test_aggregate_mean_respects_padding() -> None:
    """Masked positions must not contribute; left padding is what the loader configures."""
    hidden = torch.tensor([[[100.0], [2.0], [4.0]]])
    mask = torch.tensor([[0, 1, 1]])
    assert aggregate(hidden, mask, "mean").tolist() == [[3.0]]


def test_aggregate_unknown_mode() -> None:
    with pytest.raises(ValueError, match="unknown aggregation"):
        aggregate(torch.zeros(1, 2, 3), None, "median")


def test_capture_shapes(tiny_model: LlamaForCausalLM) -> None:
    input_ids = torch.randint(0, 64, (5, 7))
    mask = torch.ones_like(input_ids)

    with ActivationRecorder(tiny_model, layers="all") as recorder:
        recorder.capture_batch(input_ids, mask)
        captured = {k: torch.cat(v, dim=0) for k, v in recorder._buffers.items()}

    assert sorted(captured) == [0, 1, 2]
    for tensor in captured.values():
        assert tensor.shape == (5, HIDDEN)
        assert tensor.dtype is torch.float32


def test_capture_selected_layers_only(tiny_model: LlamaForCausalLM) -> None:
    input_ids = torch.randint(0, 64, (2, 5))
    with ActivationRecorder(tiny_model, layers=[0, 2]) as recorder:
        recorder.capture_batch(input_ids, torch.ones_like(input_ids))
        assert sorted(recorder._buffers) == [0, 2]


def test_hooks_are_removed(tiny_model: LlamaForCausalLM) -> None:
    """A leaked hook would silently corrupt every later forward pass."""
    before = sum(len(block._forward_hooks) for block in find_layers(tiny_model))
    with ActivationRecorder(tiny_model, layers="all"):
        during = sum(len(block._forward_hooks) for block in find_layers(tiny_model))
    after = sum(len(block._forward_hooks) for block in find_layers(tiny_model))

    assert during == before + LAYERS
    assert after == before


def test_capture_does_not_change_model_output(tiny_model: LlamaForCausalLM) -> None:
    """Hooks must be read-only: logits with and without them must match exactly."""
    input_ids = torch.randint(0, 64, (2, 6))
    with torch.inference_mode():
        baseline = tiny_model(input_ids=input_ids).logits.clone()
        with ActivationRecorder(tiny_model, layers="all") as recorder:
            recorder.capture_batch(input_ids, torch.ones_like(input_ids))
        after = tiny_model(input_ids=input_ids).logits

    assert torch.equal(baseline, after)


def test_last_token_capture_matches_manual_forward(tiny_model: LlamaForCausalLM) -> None:
    """The captured vector must be the block's actual output, not something adjacent."""
    input_ids = torch.randint(0, 64, (1, 6))
    with torch.inference_mode():
        reference = tiny_model.model(
            input_ids=input_ids, output_hidden_states=True, use_cache=False
        ).hidden_states

        with ActivationRecorder(tiny_model, layers=[1]) as recorder:
            recorder.capture_batch(input_ids, torch.ones_like(input_ids))
            captured = recorder._buffers[1][0]

    # hidden_states[0] is the embedding output, so block i's output is hidden_states[i+1].
    expected = reference[2][:, -1, :].to(torch.float32)
    assert torch.allclose(captured, expected, atol=1e-5)


def test_save_and_load_activations_roundtrip(tmp_path) -> None:
    tensors = {0: torch.randn(4, 8), 5: torch.randn(4, 8)}
    path = save_activations(tmp_path / "acts.safetensors", tensors, metadata={"split": "test"})
    loaded = load_activations(path)

    assert sorted(loaded) == [0, 5]
    for key in tensors:
        # Stored as float16, so compare at half precision's tolerance.
        assert torch.allclose(loaded[key], tensors[key], atol=1e-2)
