"""Baseline and heuristic font slicing strategies and strategy merging."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np

from mlfont.data import parse_codepoints_file
from mlfont.frequency import get_equally_sized_chunks
from mlfont.maths import balance_topics_diff_constrained, balance_topics_greedy
from mlfont.schemas import SlicingStrategy


def generate_single_slice_strategy(
    codepoints: Iterable[int],
    name: str = "single_slice",
) -> SlicingStrategy:
    """Places all codepoints into a single slice (monolithic font download)."""
    sorted_cps = sorted(set(int(cp) for cp in codepoints))
    return SlicingStrategy(
        subsets=[sorted_cps] if sorted_cps else [],
        name=name,
        metadata={"method": "single_slice"},
    )


def generate_min_slice_strategy(
    codepoints: Iterable[int],
    name: str = "min_slice",
) -> SlicingStrategy:
    """Places every codepoint into its own singleton slice (1 glyph per slice)."""
    sorted_cps = sorted(set(int(cp) for cp in codepoints))
    return SlicingStrategy(
        subsets=[[cp] for cp in sorted_cps],
        name=name,
        metadata={"method": "min_slice"},
    )


def generate_equal_bin_strategy(
    codepoints: Iterable[int],
    num_slices: int,
    name: str = "equal_bin",
) -> SlicingStrategy:
    """Partitions sorted codepoints into num_slices contiguous equal-sized bins."""
    if num_slices <= 0:
        raise ValueError("num_slices must be positive")
    sorted_cps = sorted(set(int(cp) for cp in codepoints))
    chunks = get_equally_sized_chunks(sorted_cps, num_slices)
    return SlicingStrategy(
        subsets=[list(chunk) for chunk in chunks if chunk],
        name=name,
        metadata={"method": "equal_bin", "requested_slices": num_slices},
    )


def generate_random_slice_strategy(
    codepoints: Iterable[int],
    num_slices: int,
    seed: Optional[int] = 42,
    name: str = "random_slice",
) -> SlicingStrategy:
    """Randomly permutes codepoints and partitions them into equal-sized slices."""
    if num_slices <= 0:
        raise ValueError("num_slices must be positive")
    sorted_cps = sorted(set(int(cp) for cp in codepoints))
    rng = np.random.default_rng(seed)
    permuted = list(rng.permutation(sorted_cps))
    chunks = get_equally_sized_chunks(permuted, num_slices)
    subsets = [sorted(int(c) for c in chunk) for chunk in chunks if len(chunk) > 0]
    return SlicingStrategy(
        subsets=subsets,
        name=name,
        metadata={"method": "random_slice", "requested_slices": num_slices, "seed": seed},
    )


def generate_language_slice_strategy(
    codepoints: Iterable[int],
    nam_paths: Sequence[str | Path],
    include_unassigned: bool = False,
    name: str = "language_slice",
) -> SlicingStrategy:
    """Partitions codepoints using Unicode script/language .nam subset files.

    Iterates through the provided .nam files in order, assigning each codepoint
    in the target font to the first .nam file that contains it.

    Args:
        codepoints: Target font codepoints.
        nam_paths: Ordered sequence of paths to .nam files.
        include_unassigned: If True, appends one final slice containing any
            font codepoints not covered by the provided .nam files.
        name: Strategy name.
    """
    remaining = set(int(cp) for cp in codepoints)
    subsets: List[List[int]] = []
    slice_names: List[str] = []

    for nam_path in nam_paths:
        p = Path(nam_path)
        nam_cps = parse_codepoints_file(p)
        matched = sorted(remaining.intersection(nam_cps))
        if matched:
            subsets.append(matched)
            slice_names.append(p.stem)
            remaining.difference_update(matched)

    if include_unassigned and remaining:
        subsets.append(sorted(remaining))
        slice_names.append("unassigned")

    return SlicingStrategy(
        subsets=subsets,
        name=name,
        metadata={
            "method": "language_slice",
            "slice_names": slice_names,
            "unassigned_codepoints": len(remaining) if not include_unassigned else 0,
        },
    )


def generate_greedy_topic_strategy(
    codepoints: Sequence[int],
    lambda_matrix: np.ndarray,
    normalization_method: str = "topic_sum",
    max_diff: Optional[int] = None,
    name: str = "greedy_topic",
) -> SlicingStrategy:
    """Constructs a balanced slicing strategy from a topic-codepoint matrix.

    Args:
        codepoints: Sequence of codepoints corresponding to columns of lambda_matrix.
        lambda_matrix: 2D float64 array of shape (n_topics, n_codepoints).
        normalization_method: Matrix normalization ('topic_sum', 'row_sum',
            'global_max', 'spectral_norm', or 'none').
        max_diff: If None, uses strict greedy topic capacity balancing
            (balance_topics_greedy). If an integer >= 0, uses
            balance_topics_diff_constrained.
        name: Strategy name.
    """
    if lambda_matrix.ndim != 2:
        raise ValueError("lambda_matrix must be 2D")
    num_topics, num_cols = lambda_matrix.shape
    if len(codepoints) != num_cols:
        raise ValueError(
            f"len(codepoints) ({len(codepoints)}) != lambda_matrix.shape[1] ({num_cols})"
        )

    if max_diff is None:
        assignment = balance_topics_greedy(
            lambda_matrix, normalization_method=normalization_method
        )
    else:
        assignment = balance_topics_diff_constrained(
            lambda_matrix,
            max_diff=max_diff,
            normalization_method=normalization_method,
        )

    subsets: List[List[int]] = []
    for topic_idx in range(num_topics):
        col_indices = np.flatnonzero(assignment[topic_idx] > 0)
        if col_indices.size > 0:
            subset = sorted(int(codepoints[int(c)]) for c in col_indices)
            subsets.append(subset)

    return SlicingStrategy(
        subsets=subsets,
        name=name,
        metadata={
            "method": "greedy_topic",
            "normalization_method": normalization_method,
            "max_diff": max_diff,
            "num_topics": num_topics,
        },
    )


def merge_partial_strategies(
    strategies: Sequence[SlicingStrategy],
    full_codepoints: Optional[Iterable[int]] = None,
    name: str = "merged_strategy",
) -> SlicingStrategy:
    """Merges multiple partial SlicingStrategy objects into a single disjoint strategy.

    Earlier strategies in the sequence take precedence if codepoints overlap.
    If full_codepoints is provided, any codepoints missing from all partial
    strategies are placed into a final residual slice.
    """
    seen: set[int] = set()
    merged_subsets: List[List[int]] = []

    for strat in strategies:
        for subset in strat.subsets:
            dedup = sorted(int(cp) for cp in subset if int(cp) not in seen)
            if dedup:
                merged_subsets.append(dedup)
                seen.update(dedup)

    if full_codepoints is not None:
        residual = sorted(int(cp) for cp in full_codepoints if int(cp) not in seen)
        if residual:
            merged_subsets.append(residual)
            seen.update(residual)

    return SlicingStrategy(
        subsets=merged_subsets,
        name=name,
        metadata={
            "method": "merged_strategy",
            "component_strategies": [s.name for s in strategies],
        },
    )
