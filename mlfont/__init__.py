"""MLFont: Machine-Learning-Guided Range-Compact Web Font Slicing."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from mlfont.baselines import (
    generate_equal_bin_strategy,
    generate_greedy_topic_strategy,
    generate_language_slice_strategy,
    generate_min_slice_strategy,
    generate_random_slice_strategy,
    generate_single_slice_strategy,
    merge_partial_strategies,
)
from mlfont.configs import (
    MLFontPreset,
    PRESET_REGISTRY,
    chinese_production_config,
    get_preset_config,
    japanese_production_config,
    korean_production_config,
    lgc_production_config,
    traditional_chinese_production_config,
)
from mlfont.data import (
    DocumentCorpusProcessor,
    extract_codepoints_from_font,
    fill_codepoint_gaps,
    parse_codepoints_file,
    save_codepoints_file,
)
from mlfont.dnntm import (
    DNNTMConfig,
    DNNTMLossType,
    DNNTMSolver,
    DnnTmConfig,
    DnnTmLearner,
    DnnTmLossType,
    DnnTopicNetwork,
)
from mlfont.evaluator import (
    FontLoadingEvaluator,
    compute_distribution_stats,
    save_evaluation_results_json,
)
from mlfont.export import (
    PretrainedMLFontBundle,
    export_pretrained_bundle,
    load_pretrained,
    save_safetensors_model,
)
from mlfont.frequency import (
    generate_most_and_equal_bin_strategy,
    get_equally_sized_chunks,
    get_slices_based_on_frequency,
    preprocess_codepoint_list_and_lambda_matrix,
)
from mlfont.maths import (
    ExponentialDecay,
    PowerDecay,
    balance_topics_diff_constrained,
    balance_topics_greedy,
    balance_topics_unbalanced,
    create_params_decay,
    normalize_lambda_matrix,
    sinkhorn_knopp_normalize,
)
from mlfont.olda import (
    OnlineLDAConfig,
    OnlineLDASolver,
    OnlineLdaConfig,
    OnlineLdaLearner,
    align_and_smooth_topic_matrix,
)
from mlfont.range_compact import (
    Gene,
    MUTATION_REGISTRY,
    RangeCompactGASolver,
    RangeCompactGeneticLearner,
    build_island_bridged_topics,
    decrease_max_topic_mutation,
    increase_min_topic_mutation,
    init_gene_by_equal_binning,
    init_gene_from_strategy,
    init_gene_from_topic_matrix,
    left_shift_mutation,
    merge_short_range_mutation,
    potts_viterbi_smooth_topics,
    purely_random_crossover,
    purely_random_mutation,
    remove_duplicate_genes,
    right_shift_mutation,
    swap_topic_mutation,
)
from mlfont.schemas import (
    DecayConfig,
    DistributionStats,
    EvaluationParams,
    GeneticHistoryEvent,
    RangeCompactConfig,
    STANDARD_EVALUATION_PARAMS,
    SlicingStrategy,
    StrategyEvaluationResult,
)
from mlfont.simulator import (
    BrowserNetworkProfile,
    ChromeBrowserPageResult,
    FULL_FONT_RAW_BYTES,
    FULL_FONT_WOFF2_BYTES,
    PageLoadSimulationResult,
    STANDARD_BROWSER_NETWORK_PROFILES,
    StrategyWebsiteSummary,
    WebPageSample,
    WebsiteComparisonReport,
    WebsiteFontSimulator,
    extract_visible_text_from_html,
    extract_webpage_codepoints,
    get_network_profile,
    get_real_website_benchmark_pages,
    load_webpage_sample,
    run_headless_chrome_benchmark,
    save_website_simulation_report_json,
)
from mlfont.subsetter import (
    FontSliceArtifact,
    format_unicode_ranges,
    generate_css,
    slice_font_file,
    slice_font_to_woff2,
)


class FrequencySolver:
    """Convenience wrapper for two-stage head/tail frequency tiering."""

    def __init__(
        self,
        most_sizes: Sequence[int] = (),
        least_sizes: Sequence[int] = (),
        codepoints: Optional[Sequence[int]] = None,
    ) -> None:
        if codepoints is None and (
            len(most_sizes) > 12
            or (len(most_sizes) > 0 and max(int(x) for x in most_sizes) > 2000)
        ):
            self.codepoints_: List[int] = sorted(set(int(c) for c in most_sizes))
            self.most_sizes: List[int] = []
        else:
            self.codepoints_ = sorted(set(int(c) for c in (codepoints or ())))
            self.most_sizes = [int(m) for m in most_sizes if int(m) > 0]
        self.least_sizes = [int(ls) for ls in least_sizes if int(ls) > 0]
        self.frequency_array_: Optional[np.ndarray] = None

    def fit(
        self,
        docs: Sequence[object],
        codepoints: Optional[Sequence[int]] = None,
    ) -> "FrequencySolver":
        """Counts document frequency across the target codepoints."""
        if codepoints is not None:
            self.codepoints_ = sorted(set(int(c) for c in codepoints))
        if not self.codepoints_:
            all_cps: set[int] = set()
            for doc in docs:
                if isinstance(doc, str):
                    all_cps.update(ord(ch) for ch in doc)
                elif isinstance(doc, (set, list, tuple)):
                    all_cps.update(int(c) for c in doc)
            self.codepoints_ = sorted(all_cps)
        proc = (
            docs
            if isinstance(docs, DocumentCorpusProcessor)
            else DocumentCorpusProcessor(docs)
        )
        self.frequency_array_ = proc.count_frequency_array(
            self.codepoints_, deduplicate_within_doc=True
        )
        return self

    def partition_most_frequent(
        self,
        most_counts: Optional[Sequence[int]] = None,
        least_counts: Optional[Sequence[int]] = None,
    ) -> tuple[List[List[int]], List[int]]:
        """Partitions fitted codepoints into (most_slices, remaining_codepoints)."""
        if self.frequency_array_ is None:
            raise ValueError("Call fit(docs) before partition_most_frequent().")
        m_sizes = (
            [int(m) for m in most_counts if int(m) > 0]
            if most_counts is not None
            else self.most_sizes
        )
        l_sizes = (
            [int(ls) for ls in least_counts if int(ls) > 0]
            if least_counts is not None
            else self.least_sizes
        )
        _, most_slices, remaining_cps, _ = get_slices_based_on_frequency(
            self.codepoints_,
            most_sizes=m_sizes,
            frequency_array=self.frequency_array_,
            least_sizes=l_sizes,
        )
        return most_slices, remaining_cps

    def split(
        self,
        codepoints: Sequence[int],
        frequency_array: np.ndarray,
    ) -> tuple[List[int], List[List[int]], List[int], List[List[int]]]:
        return get_slices_based_on_frequency(
            codepoints,
            most_sizes=self.most_sizes,
            frequency_array=frequency_array,
            least_sizes=self.least_sizes,
        )


class FontSubsetter:
    """Convenience wrapper for physical WOFF2/TTF font slicing and CSS generation."""

    def __init__(self, font_path: str | Path) -> None:
        self.font_path = Path(font_path)

    def slice_font(
        self,
        strategy: SlicingStrategy,
        output_dir: str | Path,
        font_family: Optional[str] = None,
        flavor: Optional[str] = "woff2",
        num_workers: int = 1,
    ) -> tuple[List[FontSliceArtifact], Path]:
        return slice_font_file(
            font_path=self.font_path,
            strategy=strategy,
            output_dir=output_dir,
            font_family=font_family,
            flavor=flavor,
            num_workers=num_workers,
        )


# Backward-compatible function aliases
extract_font_codepoints = extract_codepoints_from_font
normalize_topic_matrix = normalize_lambda_matrix

__version__ = "0.1.0"

__all__ = [
    "BrowserNetworkProfile",
    "ChromeBrowserPageResult",
    "DNNTMConfig",
    "DNNTMLossType",
    "DNNTMSolver",
    "DecayConfig",
    "DistributionStats",
    "DnnTmConfig",
    "DnnTmLearner",
    "DnnTmLossType",
    "DnnTopicNetwork",
    "DocumentCorpusProcessor",
    "EvaluationParams",
    "ExponentialDecay",
    "FULL_FONT_RAW_BYTES",
    "FULL_FONT_WOFF2_BYTES",
    "FontLoadingEvaluator",
    "FontSliceArtifact",
    "FontSubsetter",
    "FrequencySolver",
    "Gene",
    "GeneticHistoryEvent",
    "MLFontPreset",
    "MUTATION_REGISTRY",
    "OnlineLDAConfig",
    "OnlineLDASolver",
    "OnlineLdaConfig",
    "OnlineLdaLearner",
    "PRESET_REGISTRY",
    "PageLoadSimulationResult",
    "PowerDecay",
    "PretrainedMLFontBundle",
    "RangeCompactConfig",
    "RangeCompactGASolver",
    "RangeCompactGeneticLearner",
    "STANDARD_BROWSER_NETWORK_PROFILES",
    "STANDARD_EVALUATION_PARAMS",
    "SlicingStrategy",
    "StrategyEvaluationResult",
    "StrategyWebsiteSummary",
    "WebPageSample",
    "WebsiteComparisonReport",
    "WebsiteFontSimulator",
    "align_and_smooth_topic_matrix",
    "balance_topics_diff_constrained",
    "balance_topics_greedy",
    "balance_topics_unbalanced",
    "build_island_bridged_topics",
    "chinese_production_config",
    "compute_distribution_stats",
    "create_params_decay",
    "decrease_max_topic_mutation",
    "export_pretrained_bundle",
    "extract_codepoints_from_font",
    "extract_font_codepoints",
    "extract_visible_text_from_html",
    "extract_webpage_codepoints",
    "fill_codepoint_gaps",
    "format_unicode_ranges",
    "generate_css",
    "generate_equal_bin_strategy",
    "generate_greedy_topic_strategy",
    "generate_language_slice_strategy",
    "generate_min_slice_strategy",
    "generate_most_and_equal_bin_strategy",
    "generate_random_slice_strategy",
    "generate_single_slice_strategy",
    "get_equally_sized_chunks",
    "get_network_profile",
    "get_preset_config",
    "get_real_website_benchmark_pages",
    "get_slices_based_on_frequency",
    "increase_min_topic_mutation",
    "init_gene_by_equal_binning",
    "init_gene_from_strategy",
    "init_gene_from_topic_matrix",
    "japanese_production_config",
    "korean_production_config",
    "left_shift_mutation",
    "lgc_production_config",
    "load_pretrained",
    "load_webpage_sample",
    "merge_partial_strategies",
    "merge_short_range_mutation",
    "normalize_lambda_matrix",
    "normalize_topic_matrix",
    "parse_codepoints_file",
    "potts_viterbi_smooth_topics",
    "preprocess_codepoint_list_and_lambda_matrix",
    "purely_random_crossover",
    "purely_random_mutation",
    "remove_duplicate_genes",
    "right_shift_mutation",
    "run_headless_chrome_benchmark",
    "save_codepoints_file",
    "save_evaluation_results_json",
    "save_safetensors_model",
    "save_website_simulation_report_json",
    "sinkhorn_knopp_normalize",
    "slice_font_file",
    "slice_font_to_woff2",
    "swap_topic_mutation",
    "traditional_chinese_production_config",
]
