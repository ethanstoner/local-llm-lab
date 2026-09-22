"""Residual-stream activation capture via PyTorch forward hooks.

Every transformer block writes its output back into a shared residual stream, so the
output of block *i* is the model's internal state at depth *i* for each token position.
Hooking that tensor gives a per-prompt vector without modifying the model in any way -
no weights are touched, no forward behaviour changes, and the hooks are removed when the
recorder's context exits.

Aggregation happens inside the hook rather than afterwards. Keeping every position of
every layer for a batch would cost hundreds of megabytes for no benefit, since the
analyses here only ever use one vector per prompt per layer.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from src.utils.logging import get_logger

logger = get_logger(__name__)

#: Attribute paths searched, in order, for the list of transformer blocks.
_LAYER_PATHS = (
    "model.layers",          # Llama, Qwen2, Mistral, Gemma
    "transformer.h",         # GPT-2, Falcon
    "gpt_neox.layers",       # Pythia
    "model.decoder.layers",  # OPT
)


def find_layers(model: nn.Module) -> nn.ModuleList:
    """Locate a model's list of transformer blocks.

    Args:
        model: A causal language model.

    Returns:
        The ``ModuleList`` of decoder blocks.

    Raises:
        AttributeError: If no known layout matches, which is better than silently
            hooking the wrong modules.
    """
    for path in _LAYER_PATHS:
        node: Any = model
        for part in path.split("."):
            node = getattr(node, part, None)
            if node is None:
                break
        if isinstance(node, (nn.ModuleList, list)):
            return node  # type: ignore[return-value]
    raise AttributeError(
        f"could not locate transformer blocks on {type(model).__name__}; "
        f"tried {list(_LAYER_PATHS)}"
    )


def resolve_layer_indices(n_layers: int, layers: str | Sequence[int]) -> list[int]:
    """Turn a config's layer selection into a validated list of indices.

    Args:
        n_layers: How many blocks the model has.
        layers: ``"all"`` or an explicit sequence of indices. Negative indices count
            from the end.

    Returns:
        Sorted, de-duplicated indices.

    Raises:
        ValueError: If an index is out of range.
    """
    if isinstance(layers, str):
        if layers != "all":
            raise ValueError(f"layer selection must be 'all' or a sequence, got {layers!r}")
        return list(range(n_layers))

    resolved: list[int] = []
    for index in layers:
        actual = index if index >= 0 else n_layers + index
        if not 0 <= actual < n_layers:
            raise ValueError(f"layer index {index} out of range for {n_layers} layers")
        resolved.append(actual)
    return sorted(set(resolved))


def _hidden_states(output: Any) -> torch.Tensor:
    """Extract the hidden-state tensor from a decoder block's output.

    Blocks return either a bare tensor or a tuple whose first element is the hidden
    states, depending on the transformers version and the architecture.
    """
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    raise TypeError(f"unexpected block output type {type(output).__name__}")


def aggregate(
    hidden: torch.Tensor,
    attention_mask: torch.Tensor | None,
    how: str,
) -> torch.Tensor:
    """Reduce ``(batch, seq, hidden)`` activations to ``(batch, hidden)``.

    Args:
        hidden: Block output.
        attention_mask: ``(batch, seq)`` mask; required for ``"mean"``.
        how: ``"last_token"`` or ``"mean"``.

    Returns:
        One vector per batch element, in float32 on the same device.

    Note:
        ``"last_token"`` assumes left padding, which :func:`src.models.loader.load_tokenizer`
        configures. With right padding the final position would be a pad token and the
        captured vector would be meaningless.
    """
    if how == "last_token":
        return hidden[:, -1, :].to(torch.float32)
    if how == "mean":
        if attention_mask is None:
            return hidden.mean(dim=1).to(torch.float32)
        mask = attention_mask.to(hidden.dtype).unsqueeze(-1)
        summed = (hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1)
        return (summed / counts).to(torch.float32)
    raise ValueError(f"unknown aggregation {how!r}")


@dataclass
class CaptureResult:
    """Activations for one batch of prompts."""

    #: layer index -> ``(n_prompts, hidden_size)`` float32 CPU tensor
    activations: dict[int, torch.Tensor]
    n_prompts: int
    aggregation: str

    def layer_indices(self) -> list[int]:
        """Return the captured layer indices in order."""
        return sorted(self.activations)


class ActivationRecorder:
    """Context manager that records residual-stream vectors from selected blocks.

    Example:
        >>> with ActivationRecorder(model, layers="all") as rec:  # doctest: +SKIP
        ...     result = rec.capture_prompts(tokenizer, prompts, batch_size=8)
        >>> result.activations[14].shape  # doctest: +SKIP
        torch.Size([100, 3584])
    """

    def __init__(
        self,
        model: nn.Module,
        layers: str | Sequence[int] = "all",
        aggregation: str = "last_token",
        keep_on_cpu: bool = True,
    ) -> None:
        """
        Args:
            model: The loaded causal LM.
            layers: ``"all"`` or explicit block indices.
            aggregation: ``"last_token"`` or ``"mean"``.
            keep_on_cpu: Move captured vectors to host memory immediately. Capturing
                every layer for a large prompt set otherwise competes with the model
                for VRAM.
        """
        self.model = model
        self.blocks = find_layers(model)
        self.layer_indices = resolve_layer_indices(len(self.blocks), layers)
        self.aggregation = aggregation
        self.keep_on_cpu = keep_on_cpu

        self._handles: list[Any] = []
        self._buffers: dict[int, list[torch.Tensor]] = {}
        self._current_mask: torch.Tensor | None = None

    # -- lifecycle ---------------------------------------------------------------

    def __enter__(self) -> ActivationRecorder:
        self.register()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.remove()

    def register(self) -> None:
        """Attach forward hooks to the selected blocks."""
        if self._handles:
            return
        for index in self.layer_indices:
            handle = self.blocks[index].register_forward_hook(self._make_hook(index))
            self._handles.append(handle)
        logger.debug("Registered %d activation hooks", len(self._handles))

    def remove(self) -> None:
        """Detach every hook. Safe to call more than once."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _make_hook(self, index: int) -> Any:
        """Build the forward hook for one block."""

        def hook(_module: nn.Module, _args: Any, output: Any) -> None:
            hidden = _hidden_states(output)
            vector = aggregate(hidden, self._current_mask, self.aggregation)
            if self.keep_on_cpu:
                vector = vector.cpu()
            self._buffers.setdefault(index, []).append(vector.detach())

        return hook

    # -- capture -----------------------------------------------------------------

    def reset(self) -> None:
        """Discard anything captured so far."""
        self._buffers.clear()

    @torch.inference_mode()
    def capture_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> None:
        """Run one forward pass and record its activations.

        Only the prompt is processed; no tokens are generated, which is both cheaper and
        the correct stimulus for the difference-in-means analysis.
        """
        self._current_mask = attention_mask
        try:
            self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        finally:
            self._current_mask = None

    @torch.inference_mode()
    def capture_prompts(
        self,
        tokenizer: Any,
        prompts: Sequence[str],
        batch_size: int = 8,
        apply_chat_template: bool = True,
        max_length: int | None = None,
        device: str | torch.device | None = None,
    ) -> CaptureResult:
        """Capture one activation vector per prompt per selected layer.

        Args:
            tokenizer: The model's tokenizer, configured with left padding.
            prompts: Raw instruction strings.
            batch_size: Prompts per forward pass.
            apply_chat_template: Wrap each prompt in the model's instruction template.
                This matters: an instruct model's refusal behaviour is a property of the
                chat format, and capturing raw completions would measure something else.
            max_length: Optional truncation length.
            device: Device to place inputs on; defaults to the model's device.

        Returns:
            A :class:`CaptureResult` whose tensors are ``(len(prompts), hidden_size)``.
        """
        self.reset()
        target = device or next(self.model.parameters()).device

        texts: list[str] = []
        for prompt in prompts:
            if apply_chat_template and getattr(tokenizer, "chat_template", None):
                texts.append(
                    tokenizer.apply_chat_template(
                        [{"role": "user", "content": prompt}],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                )
            else:
                texts.append(prompt)

        for start in range(0, len(texts), batch_size):
            chunk = texts[start : start + batch_size]
            encoded = tokenizer(
                chunk,
                return_tensors="pt",
                padding=True,
                truncation=max_length is not None,
                max_length=max_length,
                add_special_tokens=not apply_chat_template,
            )
            encoded = {k: v.to(target) for k, v in encoded.items()}
            self.capture_batch(encoded["input_ids"], encoded["attention_mask"])

        activations = {
            index: torch.cat(chunks, dim=0) for index, chunks in self._buffers.items()
        }
        self.reset()

        for index, tensor in activations.items():
            if tensor.shape[0] != len(prompts):
                raise RuntimeError(
                    f"layer {index} captured {tensor.shape[0]} vectors for "
                    f"{len(prompts)} prompts"
                )

        return CaptureResult(
            activations=activations,
            n_prompts=len(prompts),
            aggregation=self.aggregation,
        )


def save_activations(
    path: Any,
    activations: dict[int, torch.Tensor],
    metadata: dict[str, str] | None = None,
) -> Any:
    """Write per-layer activations to a single safetensors file.

    Stored as float16: the vectors are used for means, cosines and projections, none of
    which need more than half precision, and the file is half the size.
    """
    from pathlib import Path

    from safetensors.torch import save_file

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tensors = {f"layer_{index}": tensor.to(torch.float16) for index, tensor in activations.items()}
    save_file(tensors, str(path), metadata=metadata or {})
    return path


def load_activations(path: Any) -> dict[int, torch.Tensor]:
    """Read per-layer activations written by :func:`save_activations`."""
    from safetensors.torch import load_file

    tensors = load_file(str(path))
    return {
        int(key.split("_", 1)[1]): value.to(torch.float32) for key, value in tensors.items()
    }


def iter_layer_pairs(
    a: dict[int, torch.Tensor], b: dict[int, torch.Tensor]
) -> Iterable[tuple[int, torch.Tensor, torch.Tensor]]:
    """Yield ``(layer, a_tensor, b_tensor)`` for layers present in both dictionaries."""
    for index in sorted(set(a) & set(b)):
        yield index, a[index], b[index]
