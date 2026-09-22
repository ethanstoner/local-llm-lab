"""Inference-time interventions on the residual stream, for testing whether a direction
found by observation is actually used by the model.

Phase 5 established that harmful and harmless prompts separate along a
difference-in-means direction. That is correlational: a direction can separate two
classes without the model reading from it. The interventions here are the causal test
from Arditi et al. (2024):

* **Directional ablation** removes the component along a unit direction ``r`` from every
  write into the residual stream - the embedding output, and the output of every
  attention and MLP sublayer - so the stream never carries any ``r`` component at all.
  If refusal is mediated by ``r``, the model should stop refusing.
* **Activation addition** adds a vector along ``r`` to the residual stream at one layer.
  If ``r`` is sufficient, harmless prompts should start being refused.

Both are implemented as forward hooks. Nothing is written to the weights, nothing is
saved, and the hooks are removed when the context manager exits, so the model returns to
its unmodified state between conditions.
"""

from __future__ import annotations

from typing import Any, Sequence

import torch
from torch import nn

from src.interpretability.hooks import find_layers
from src.utils.logging import get_logger

logger = get_logger(__name__)


def unit(vector: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Return ``vector`` scaled to unit L2 norm, in float32."""
    vector = vector.detach().to(torch.float32).flatten()
    norm = vector.norm()
    if norm < eps:
        raise ValueError("cannot normalise a zero vector")
    return vector / norm


def random_unit_direction(hidden_size: int, seed: int) -> torch.Tensor:
    """A seeded, isotropically random unit vector - the null control for a direction."""
    generator = torch.Generator().manual_seed(seed)
    return unit(torch.randn(hidden_size, generator=generator))


def project_out(hidden: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    """Remove the component of ``hidden`` along a unit ``direction``.

    The projection is computed in float32 and cast back, because in bf16 the subtraction
    of two nearly equal large numbers leaves a residual component that is not negligible
    once it is repeated across every sublayer.
    """
    h32 = hidden.to(torch.float32)
    coefficient = h32 @ direction
    return (h32 - coefficient.unsqueeze(-1) * direction).to(hidden.dtype)


def _replace_first(output: Any, tensor: torch.Tensor) -> Any:
    """Swap the hidden-state tensor into a module output of either shape."""
    if isinstance(output, torch.Tensor):
        return tensor
    if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
        return (tensor, *output[1:])
    raise TypeError(f"unexpected module output type {type(output).__name__}")


def _first_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    raise TypeError(f"unexpected module output type {type(output).__name__}")


def _embedding(model: nn.Module) -> nn.Module:
    embed = model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else None
    if embed is None:
        raise AttributeError(f"{type(model).__name__} exposes no input embedding")
    return embed


def residual_writers(model: nn.Module) -> list[nn.Module]:
    """Every module whose output is added into the residual stream.

    For a pre-norm decoder that is the token embedding plus each block's attention and
    MLP sublayers. Projecting ``r`` out of all of them leaves no path by which ``r`` can
    enter the stream.

    Raises:
        AttributeError: If a block does not expose ``self_attn`` and ``mlp``. Hooking the
            wrong modules would silently leave ``r`` in the stream, which is worse than
            failing.
    """
    writers: list[nn.Module] = [_embedding(model)]
    for index, block in enumerate(find_layers(model)):
        attn = getattr(block, "self_attn", None)
        mlp = getattr(block, "mlp", None)
        if attn is None or mlp is None:
            raise AttributeError(
                f"block {index} ({type(block).__name__}) has no self_attn/mlp submodules"
            )
        writers.extend((attn, mlp))
    return writers


def ablation_sites(model: nn.Module) -> list[nn.Module]:
    """The residual writers plus every block output.

    Projecting the writers alone is exact in real arithmetic, but each block adds its
    sublayer outputs to the stream in the model's own dtype. In bf16 that addition
    rounds, and in a model like Qwen2.5 whose residual stream carries a few dimensions in
    the thousands, the rounding error has a measurable component along ``r``.
    Re-projecting at each block output stops that error accumulating across depth.
    """
    return residual_writers(model) + list(find_layers(model))


class _HookContext:
    """Shared lifecycle for hook-based interventions."""

    def __init__(self) -> None:
        self._handles: list[Any] = []

    def register(self) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    def remove(self) -> None:
        """Detach every hook. Safe to call more than once."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def __enter__(self) -> "_HookContext":
        self.register()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.remove()


class DirectionalAblation(_HookContext):
    """Project a direction out of every residual-stream write, at every position.

    Example:
        >>> with DirectionalAblation(model, direction):  # doctest: +SKIP
        ...     model.generate(...)
    """

    def __init__(self, model: nn.Module, direction: torch.Tensor) -> None:
        super().__init__()
        self.model = model
        self.writers = ablation_sites(model)
        device = next(model.parameters()).device
        self.direction = unit(direction).to(device)

    def register(self) -> None:
        if self._handles:
            return
        direction = self.direction

        def hook(_module: nn.Module, _args: Any, output: Any) -> Any:
            return _replace_first(output, project_out(_first_tensor(output), direction))

        for module in self.writers:
            self._handles.append(module.register_forward_hook(hook))
        logger.debug("Directional ablation on %d sites", len(self._handles))


class ActivationAddition(_HookContext):
    """Add a fixed vector to the residual stream leaving one block, at every position.

    The vector is added to block ``layer``'s output - exactly the point Phase 5 read the
    direction from - so every later block sees the shifted stream.
    """

    def __init__(self, model: nn.Module, layer: int, vector: torch.Tensor) -> None:
        super().__init__()
        blocks = find_layers(model)
        if not 0 <= layer < len(blocks):
            raise ValueError(f"layer {layer} out of range for {len(blocks)} blocks")
        self.block = blocks[layer]
        self.layer = layer
        parameter = next(model.parameters())
        self.vector = vector.detach().flatten().to(parameter.device, torch.float32)

    def register(self) -> None:
        if self._handles:
            return
        vector = self.vector

        def hook(_module: nn.Module, _args: Any, output: Any) -> Any:
            hidden = _first_tensor(output)
            shifted = (hidden.to(torch.float32) + vector).to(hidden.dtype)
            return _replace_first(output, shifted)

        self._handles.append(self.block.register_forward_hook(hook))


class NoIntervention(_HookContext):
    """The baseline condition, so every condition can be run through the same code path."""

    def register(self) -> None:
        return None


def residual_component(
    model: nn.Module,
    tokenizer: Any,
    prompts: Sequence[str],
    direction: torch.Tensor,
    device: str | torch.device | None = None,
) -> dict[str, float]:
    """How much of the residual stream lies along ``direction``, over every block output.

    Reported as the cosine between each real (non-padding) position's residual vector
    and the direction, because the raw component is dominated by the handful of
    positions where Qwen2.5's residual norm runs into the tens of thousands. Under
    :class:`DirectionalAblation` the cosine should sit at the bf16 rounding floor.

    Returns:
        Mean and maximum ``|cos|`` across all blocks and positions, and the maximum raw
        component for reference.
    """
    target = device or next(model.parameters()).device
    blocks = find_layers(model)
    r = unit(direction).to(target)
    cosines: list[torch.Tensor] = []
    raw_peak = 0.0
    mask_holder: dict[str, torch.Tensor] = {}

    def hook(_module: nn.Module, _args: Any, output: Any) -> None:
        nonlocal raw_peak
        hidden = _first_tensor(output).to(torch.float32)
        component = hidden @ r
        cos = component.abs() / hidden.norm(dim=-1).clamp(min=1e-6)
        keep = mask_holder["mask"].bool()
        cosines.append(cos[keep].cpu())
        raw_peak = max(raw_peak, float(component[keep].abs().max()))

    handles = [block.register_forward_hook(hook) for block in blocks]
    try:
        texts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
            )
            for p in prompts
        ]
        encoded = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False)
        encoded = {k: v.to(target) for k, v in encoded.items()}
        mask_holder["mask"] = encoded["attention_mask"]
        with torch.inference_mode():
            model(**encoded, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    flat = torch.cat(cosines)
    return {
        "mean_abs_cosine": float(flat.mean()),
        "max_abs_cosine": float(flat.max()),
        "max_abs_component": raw_peak,
    }
