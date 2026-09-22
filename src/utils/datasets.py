"""Prompt datasets for the interpretability experiments.

The refusal-direction analysis needs two matched sets of instructions: one the model
should answer and one it should decline. The primary source is JailbreakBench's
``JBB-Behaviors`` benchmark, which ships exactly that pairing - 100 harmful behaviours
and 100 matched benign ones - and is ungated, citable and widely used.

A small bundled fallback exists so the pipeline still runs without network access. The
fallback's harmful entries are deliberately generic, category-level requests: they are
the *stimulus* used to elicit a refusal, they contain no operational detail, and nothing
in this project ever stores or reports the model's answer to them beyond a yes/no
refusal classification.

Reference:
    Chao et al. (2024), "JailbreakBench: An Open Robustness Benchmark for Jailbreaking
    Large Language Models". https://arxiv.org/abs/2404.01318
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from src.utils.io import repo_root
from src.utils.logging import get_logger

logger = get_logger(__name__)

JBB_DATASET = "JailbreakBench/JBB-Behaviors"
JBB_CONFIG = "behaviors"

FALLBACK_HARMFUL_PATH = "data/fallback_harmful_prompts.txt"
FALLBACK_HARMLESS_PATH = "data/fallback_harmless_prompts.txt"


@dataclass
class PromptSets:
    """Two labelled prompt sets and a record of where they came from."""

    harmful: list[str]
    harmless: list[str]
    source: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        """Return counts and provenance for the metadata block."""
        return {
            "n_harmful": len(self.harmful),
            "n_harmless": len(self.harmless),
            "source": self.source,
        }


@dataclass
class SplitSets:
    """A train/test split of both classes.

    The direction is fitted on ``*_train`` and every reported separation statistic is
    computed on ``*_test``. Without this split, "the classes separate along the
    difference of their means" is close to a tautology.
    """

    harmful_train: list[str]
    harmful_test: list[str]
    harmless_train: list[str]
    harmless_test: list[str]

    def counts(self) -> dict[str, int]:
        """Return the size of each split."""
        return {
            "harmful_train": len(self.harmful_train),
            "harmful_test": len(self.harmful_test),
            "harmless_train": len(self.harmless_train),
            "harmless_test": len(self.harmless_test),
        }


def _read_fallback(relative_path: str) -> list[str]:
    """Read one prompt per non-empty, non-comment line from a bundled file."""
    path = repo_root() / relative_path
    if not path.is_file():
        raise FileNotFoundError(f"bundled prompt file missing: {path}")
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    return [line for line in lines if line and not line.startswith("#")]


def _load_jbb() -> tuple[list[str], list[str], dict[str, Any]]:
    """Load the harmful and benign goal strings from JBB-Behaviors.

    Returns:
        ``(harmful, harmless, source_info)``.

    Raises:
        Exception: Propagated from ``datasets.load_dataset`` so the caller can decide
            whether to fall back; the failure reason is logged either way.
    """
    from datasets import load_dataset

    harmful_split = load_dataset(JBB_DATASET, JBB_CONFIG, split="harmful")
    benign_split = load_dataset(JBB_DATASET, JBB_CONFIG, split="benign")

    column = "Goal" if "Goal" in harmful_split.column_names else harmful_split.column_names[0]
    harmful = [str(x) for x in harmful_split[column]]
    harmless = [str(x) for x in benign_split[column]]

    source = {
        "kind": "jbb",
        "dataset": JBB_DATASET,
        "config": JBB_CONFIG,
        "column": column,
        "citation": "Chao et al. 2024, arXiv:2404.01318",
        "n_available_harmful": len(harmful),
        "n_available_harmless": len(harmless),
    }
    return harmful, harmless, source


def load_prompt_sets(
    harmful_source: str = "jbb",
    harmless_source: str = "jbb",
    n_per_class: int = 100,
    seed: int = 1234,
    allow_fallback: bool = True,
) -> PromptSets:
    """Load matched harmful and harmless instruction sets.

    Args:
        harmful_source: ``"jbb"`` or ``"bundled"``.
        harmless_source: ``"jbb"`` or ``"bundled"``.
        n_per_class: Maximum prompts to take from each class. Sampling is seeded, so the
            same seed selects the same prompts.
        seed: Sampling seed.
        allow_fallback: Fall back to the bundled sets if the Hub is unreachable.

    Returns:
        A :class:`PromptSets` whose ``source`` records exactly what was used.
    """
    rng = random.Random(seed)
    source: dict[str, Any] = {}
    harmful: list[str] = []
    harmless: list[str] = []

    wants_jbb = "jbb" in (harmful_source, harmless_source)
    if wants_jbb:
        try:
            jbb_harmful, jbb_harmless, source = _load_jbb()
            if harmful_source == "jbb":
                harmful = jbb_harmful
            if harmless_source == "jbb":
                harmless = jbb_harmless
        except Exception as exc:
            if not allow_fallback:
                raise
            logger.warning(
                "Could not load %s (%s); falling back to the bundled prompt sets",
                JBB_DATASET,
                exc,
            )
            source = {"kind": "bundled", "reason": f"{type(exc).__name__}: {exc}"}

    if not harmful:
        harmful = _read_fallback(FALLBACK_HARMFUL_PATH)
        source.setdefault("kind", "bundled")
        source["harmful_file"] = FALLBACK_HARMFUL_PATH
    if not harmless:
        harmless = _read_fallback(FALLBACK_HARMLESS_PATH)
        source.setdefault("kind", "bundled")
        source["harmless_file"] = FALLBACK_HARMLESS_PATH

    # Matched sizes keep the difference-in-means estimate balanced between classes.
    n = min(n_per_class, len(harmful), len(harmless))
    if n < n_per_class:
        logger.warning(
            "Only %d prompts per class available (requested %d)", n, n_per_class
        )

    harmful = rng.sample(harmful, n)
    harmless = rng.sample(harmless, n)
    source["n_used_per_class"] = n

    logger.info("Loaded %d harmful and %d harmless prompts from %s", n, n, source.get("kind"))
    return PromptSets(harmful=harmful, harmless=harmless, source=source)


def split_prompt_sets(sets: PromptSets, test_fraction: float, seed: int = 1234) -> SplitSets:
    """Split both classes into train and test partitions.

    Args:
        sets: The loaded prompt sets.
        test_fraction: Fraction held out for evaluation, strictly between 0 and 1.
        seed: Shuffling seed.

    Returns:
        A :class:`SplitSets`.
    """
    if not 0.0 < test_fraction < 1.0:
        raise ValueError(f"test_fraction must be in (0, 1), got {test_fraction}")

    rng = random.Random(seed)

    def _split(items: list[str]) -> tuple[list[str], list[str]]:
        shuffled = list(items)
        rng.shuffle(shuffled)
        n_test = max(1, int(round(len(shuffled) * test_fraction)))
        n_test = min(n_test, len(shuffled) - 1)
        return shuffled[n_test:], shuffled[:n_test]

    harmful_train, harmful_test = _split(sets.harmful)
    harmless_train, harmless_test = _split(sets.harmless)
    return SplitSets(harmful_train, harmful_test, harmless_train, harmless_test)


def load_perplexity_text() -> str:
    """Return the fixed text used for perplexity comparisons.

    The bundled benchmark corpus is reused so that the quality evaluation needs no
    network access and cannot drift between runs.
    """
    path = repo_root() / "data/benchmark_corpus.txt"
    if not path.is_file():
        raise FileNotFoundError(f"perplexity corpus missing: {path}")
    return path.read_text(encoding="utf-8")
