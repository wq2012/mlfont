"""Codepoint frequency slicing and Most-and-Equal-Bin strategy generation."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from mlfont.schemas import SlicingStrategy


def get_equally_sized_chunks(items: Sequence[int], num_chunks: int) -> List[List[int]]:
    """Splits a sequence into num_chunks nearly equal-sized contiguous lists."""
    if num_chunks <= 0:
        raise ValueError("num_chunks must be positive")
    if num_chunks > len(items):
        raise ValueError("num_chunks cannot exceed length of items")
    items_list = list(items)
    n = len(items_list)
    small_size = n // num_chunks
    large_size = small_size + 1
    n_large = n % num_chunks

    chunks: List[List[int]] = []
    idx = 0
    for _ in range(n_large):
        chunks.append(items_list[idx:idx + large_size])
        idx += large_size
    for _ in range(num_chunks - n_large):
        chunks.append(items_list[idx:idx + small_size])
        idx += small_size
    return chunks


def get_slices_based_on_frequency(
    codepoint_list: Sequence[int],
    most_sizes: Sequence[int] = (),
    frequency_array: Optional[np.ndarray] = None,
    least_sizes: Sequence[int] = (),
) -> Tuple[List[int], List[List[int]], List[int], List[List[int]]]:
    """Partitions codepoints into most-frequent slices, remaining, and least-frequent slices.

    Args:
        codepoint_list: Ordered list of Unicode codepoints.
        most_sizes: List of slice sizes for the most frequent codepoints
            (e.g., [2000, 1000]).
        frequency_array: 1D numpy array of frequencies aligned with codepoint_list.
        least_sizes: Optional list of slice sizes for the least frequent codepoints.

    Returns:
        Tuple of:
            - most_freq_indices: Flat list of indices into codepoint_list assigned to 'most'.
            - most_freq_slices: List of codepoint lists for each 'most' slice.
            - remaining_codepoints: Sorted list of remaining codepoints.
            - least_freq_slices: List of codepoint lists for each 'least' slice.
    """
    n = len(codepoint_list)
    if frequency_array is None:
        raise ValueError("frequency_array is required")
    freq = np.asarray(frequency_array)
    if freq.ndim != 1 or len(freq) != n:
        raise ValueError("frequency_array must be 1D with same length as codepoint_list")

    most_sizes = [int(x) for x in most_sizes if int(x) > 0]
    least_sizes = [int(x) for x in least_sizes if int(x) > 0]
    if sum(most_sizes) + sum(least_sizes) > n:
        raise ValueError("Sum of most_sizes and least_sizes exceeds number of codepoints")

    # Sort indices by descending frequency (tie-breaking by codepoint order)
    desc_indices = list(np.argsort(-freq, kind="mergesort"))

    most_freq_indices: List[int] = []
    most_freq_slices: List[List[int]] = []
    cursor = 0
    for size in most_sizes:
        slice_idx = desc_indices[cursor:cursor + size]
        most_freq_indices.extend(slice_idx)
        most_freq_slices.append(sorted(int(codepoint_list[i]) for i in slice_idx))
        cursor += size

    least_freq_indices: List[int] = []
    least_freq_slices: List[List[int]] = []
    end_cursor = n
    for size in least_sizes:
        slice_idx = desc_indices[end_cursor - size:end_cursor]
        least_freq_indices.extend(slice_idx)
        least_freq_slices.append(sorted(int(codepoint_list[i]) for i in slice_idx))
        end_cursor -= size

    excluded = set(most_freq_indices) | set(least_freq_indices)
    remaining_codepoints = sorted(
        int(codepoint_list[i]) for i in range(n) if i not in excluded
    )
    return most_freq_indices, most_freq_slices, remaining_codepoints, least_freq_slices


def preprocess_codepoint_list_and_lambda_matrix(
    most_sizes: Sequence[int],
    frequency_array: np.ndarray,
    codepoint_list: Sequence[int],
    lambda_matrix: np.ndarray,
) -> Tuple[List[int], np.ndarray]:
    """Removes the top most-frequent codepoints from codepoint_list and lambda_matrix columns."""
    sorted_cps = sorted(int(c) for c in codepoint_list)
    if lambda_matrix.ndim != 2 or lambda_matrix.shape[1] != len(sorted_cps):
        raise ValueError(
            "lambda_matrix must be 2D with column count matching len(codepoint_list)"
        )
    most_indices, _, remaining_cps, _ = get_slices_based_on_frequency(
        sorted_cps, most_sizes=most_sizes, frequency_array=frequency_array
    )
    reduced_lambda = np.delete(lambda_matrix, most_indices, axis=1)
    return remaining_cps, reduced_lambda


def generate_most_and_equal_bin_strategy(
    codepoint_list: Sequence[int],
    frequency_array: np.ndarray,
    most_sizes: Sequence[int],
    num_bins: int,
    name: str = "",
) -> SlicingStrategy:
    """Constructs a Most-and-Equal-Bin slicing strategy.

    1. First k slices contain the most frequent codepoints with sizes in most_sizes.
    2. Remaining codepoints are partitioned into num_bins equal contiguous bins.
    """
    sorted_cps = sorted(set(int(c) for c in codepoint_list))
    _, most_slices, remaining_cps, _ = get_slices_based_on_frequency(
        sorted_cps, most_sizes=most_sizes, frequency_array=frequency_array
    )
    equal_bins = get_equally_sized_chunks(remaining_cps, num_bins) if num_bins > 0 else []
    subsets = most_slices + equal_bins
    if not name:
        most_str = "_".join(str(m) for m in most_sizes)
        name = f"most_{most_str}_equal_bin_{num_bins}" if most_str else f"equal_bin_{num_bins}"
    return SlicingStrategy(
        subsets=subsets,
        name=name,
        metadata={"most_sizes": list(most_sizes), "num_bins": num_bins},
    )
