"""Multi-session web font loading simulator and strategy evaluator.

Simulation model: documents are grouped sequentially into browsing sessions
of length L, a browser slice cache is maintained within each session, and
per-glyph RTT latency, download latency, total latency, CSS range count, and
upfront/session glyph utilization are computed across network profiles.

Latency model (per page):
  * Every slice whose ``unicode-range`` intersects the page's codepoints and
    that is not already in the session cache is fetched exactly once.
  * ``rtt_per_glyph_ms = (num_new_slices * rtt_ms) / num_covered_glyphs``.
    This is a conservative *serial-request* model: each new slice is charged
    one full RTT, as if the browser issued the slice requests one after
    another (no HTTP/2 multiplexing or parallel connections). The parallel,
    wave-based model lives in :mod:`mlfont.simulator`.
  * ``download_per_glyph_ms = (bytes_of_new_slices / bandwidth) / num_covered_glyphs``.
  * Codepoints that the strategy does not cover are counted as missing and do
    not contribute to the per-glyph denominator.
"""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set

import numpy as np

from mlfont.schemas import (
    DistributionStats,
    EvaluationParams,
    STANDARD_EVALUATION_PARAMS,
    SlicingStrategy,
    StrategyEvaluationResult,
)


def compute_distribution_stats(values: Sequence[float]) -> DistributionStats:
    """Computes summary statistics (mean, p95, min, max, stddev) for a sequence."""
    if not values:
        return DistributionStats(
            mean=0.0, p95=0.0, minimal=0.0, maximal=0.0, stddev=0.0
        )
    arr = np.asarray(values, dtype=np.float64)
    return DistributionStats(
        mean=float(np.mean(arr)),
        p95=float(np.percentile(arr, 95)),
        minimal=float(np.min(arr)),
        maximal=float(np.max(arr)),
        stddev=float(np.std(arr)),
    )


class FontLoadingEvaluator:
    """Evaluates font slicing strategies over document codepoint samples."""

    def __init__(
        self,
        codepoints: Iterable[int],
        document_samples: Sequence[Set[int]],
    ) -> None:
        """Initializes the evaluator with a target codepoint universe and test samples.

        Args:
            codepoints: Target font codepoints under evaluation.
            document_samples: Sequence of non-empty sets of codepoints (filtered
                to the target font codepoints), one per test document.
        """
        self.codepoint_set: Set[int] = set(int(c) for c in codepoints)
        self.samples: List[Set[int]] = []
        for sample in document_samples:
            filtered = self.codepoint_set.intersection(sample)
            if filtered:
                self.samples.append(filtered)

    def evaluate_strategy(
        self,
        strategy: SlicingStrategy,
        params: EvaluationParams,
        strategy_name: Optional[str] = None,
        slice_byte_sizes: Optional[Sequence[int]] = None,
    ) -> StrategyEvaluationResult:
        """Evaluates a single SlicingStrategy under one EvaluationParams profile.

        Args:
            strategy: SlicingStrategy to evaluate.
            params: Network and session parameters.
            strategy_name: Optional override for strategy name.
            slice_byte_sizes: Optional actual compressed byte sizes for each
                slice. If None, uses len(slice) * params.average_glyph_size_bytes.
        """
        name = strategy_name or strategy.name or "strategy"
        slice_mapping: Dict[int, int] = {}
        slice_glyph_counts: List[int] = []
        for idx, subset in enumerate(strategy.subsets):
            unique_subset = set(int(cp) for cp in subset)
            slice_glyph_counts.append(len(unique_subset))
            for cp in unique_subset:
                slice_mapping[cp] = idx

        if slice_byte_sizes is not None and len(slice_byte_sizes) != len(
            strategy.subsets
        ):
            raise ValueError("slice_byte_sizes length must match strategy.num_slices")

        unassigned_font_cps = len(self.codepoint_set - set(slice_mapping.keys()))

        rtt_list: List[float] = []
        download_time_list: List[float] = []
        upfront_utilization_list: List[float] = []
        session_utilization_list: List[float] = []
        missing_codepoints: Set[int] = set()

        cache_indices_of_slices: Set[int] = set()
        session_sample_codepoints: Set[int] = set()
        session_downloaded_glyph_count = 0

        session_len = max(1, int(params.session_length))

        for i, sample in enumerate(self.samples):
            new_indices_of_slices: Set[int] = set()
            covered_in_sample: Set[int] = set()

            for cp in sample:
                slice_idx = slice_mapping.get(cp)
                if slice_idx is None:
                    missing_codepoints.add(cp)
                else:
                    covered_in_sample.add(cp)
                    if slice_idx not in cache_indices_of_slices:
                        new_indices_of_slices.add(slice_idx)
                        cache_indices_of_slices.add(slice_idx)
                        session_downloaded_glyph_count += slice_glyph_counts[slice_idx]

            session_sample_codepoints.update(covered_in_sample)
            denom = max(1, len(covered_in_sample))
            rtt_ms = (len(new_indices_of_slices) * params.rtt_ms) / denom
            rtt_list.append(rtt_ms)

            if slice_byte_sizes is not None:
                total_bytes = sum(
                    float(slice_byte_sizes[idx]) for idx in new_indices_of_slices
                )
            else:
                total_glyphs = sum(
                    slice_glyph_counts[idx] for idx in new_indices_of_slices
                )
                total_bytes = total_glyphs * params.average_glyph_size_bytes

            download_ms = (total_bytes / params.bandwidth_bytes_per_ms) / denom
            download_time_list.append(download_ms)

            if i % session_len == 0:
                if session_downloaded_glyph_count > 0:
                    upfront_utilization_list.append(
                        100.0
                        * len(session_sample_codepoints)
                        / float(session_downloaded_glyph_count)
                    )

            if (i + 1) % session_len == 0 or (i + 1) == len(self.samples):
                if session_downloaded_glyph_count > 0:
                    session_utilization_list.append(
                        100.0
                        * len(session_sample_codepoints)
                        / float(session_downloaded_glyph_count)
                    )
                cache_indices_of_slices.clear()
                session_sample_codepoints.clear()
                session_downloaded_glyph_count = 0

        total_time_list = [r + d for r, d in zip(rtt_list, download_time_list)]
        upfront_util = (
            float(np.mean(upfront_utilization_list))
            if upfront_utilization_list
            else 0.0
        )
        session_util = (
            float(np.mean(session_utilization_list))
            if session_utilization_list
            else upfront_util
        )

        return StrategyEvaluationResult(
            strategy_name=name,
            params_name=params.name,
            num_slices=strategy.num_slices,
            num_codepoints=len(self.codepoint_set),
            num_samples=len(total_time_list),
            num_missing_codepoints=len(missing_codepoints),
            num_ranges=strategy.count_unicode_ranges(),
            total_per_glyph_ms=compute_distribution_stats(total_time_list),
            rtt_per_glyph_ms=compute_distribution_stats(rtt_list),
            download_per_glyph_ms=compute_distribution_stats(download_time_list),
            upfront_utilization_pct=upfront_util,
            session_utilization_pct=session_util,
            num_unassigned_font_codepoints=unassigned_font_cps,
        )

    def evaluate_all(
        self,
        strategies: Sequence[SlicingStrategy],
        params_list: Optional[Sequence[EvaluationParams]] = None,
        slice_byte_sizes_map: Optional[Dict[str, Sequence[int]]] = None,
    ) -> List[StrategyEvaluationResult]:
        """Evaluates multiple strategies across multiple network/session profiles."""
        profiles = (
            params_list if params_list is not None else STANDARD_EVALUATION_PARAMS
        )
        results: List[StrategyEvaluationResult] = []
        for params in profiles:
            for strat in strategies:
                byte_sizes = (
                    slice_byte_sizes_map.get(strat.name)
                    if slice_byte_sizes_map is not None
                    else None
                )
                results.append(
                    self.evaluate_strategy(
                        strat, params, slice_byte_sizes=byte_sizes
                    )
                )
        return results


def save_evaluation_results_json(
    results: Sequence[StrategyEvaluationResult],
    path: str | Path,
) -> None:
    """Saves a sequence of StrategyEvaluationResult objects to a JSON file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(r) for r in results]
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
