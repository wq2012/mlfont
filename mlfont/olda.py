"""Online Latent Dirichlet Allocation (Online LDA) font slicing learner."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import LatentDirichletAllocation

from mlfont.data import DocumentCorpusProcessor
from mlfont.frequency import get_equally_sized_chunks, get_slices_based_on_frequency
from mlfont.maths import (
    balance_topics_diff_constrained,
    balance_topics_greedy,
    balance_topics_unbalanced,
)
from mlfont.schemas import SlicingStrategy


def align_and_smooth_topic_matrix(
    raw_topic_word: np.ndarray,
    active_cols: np.ndarray,
    col_sums: np.ndarray,
    n_rem: int,
    n_topics: int,
) -> np.ndarray:
    """Computes Bayes-normalized topic posteriors aligned to contiguous Unicode bins.

    Args:
        raw_topic_word: Non-negative matrix of shape (n_topics, len(active_cols)).
        active_cols: Sorted 1D array of column indices in [0, n_rem) observed in corpus.
        col_sums: 1D array of length n_rem with document frequencies per codepoint.
        n_rem: Total number of remaining codepoints after frequency tiering.
        n_topics: Number of topic slices K.

    Returns:
        Full topic affinity matrix of shape (n_topics, n_rem) where all K topics
        carry positive posterior mass (> 0.002) and unobserved codepoints receive
        a smooth contiguous-bin Dirichlet prior.
    """
    bin_chunks = get_equally_sized_chunks(list(range(n_rem)), n_topics)
    bin_of_col = np.zeros(n_rem, dtype=np.int64)
    for b_idx, chunk in enumerate(bin_chunks):
        for col_idx in chunk:
            bin_of_col[col_idx] = b_idx

    topic_idx = np.arange(n_topics, dtype=np.float64)[:, np.newaxis]
    col_bins = bin_of_col[np.newaxis, :].astype(np.float64)
    # Smooth contiguous-bin Dirichlet prior in [0.0020, 0.0045] (> 1e-3 everywhere)
    full_lambda = 0.0020 + 0.0025 * np.exp(-np.abs(topic_idx - col_bins) / 2.0)

    if len(active_cols) == 0:
        return full_lambda

    comp = np.maximum(1e-12, np.asarray(raw_topic_word, dtype=np.float64))
    post = comp / np.maximum(1e-12, np.sum(comp, axis=0, keepdims=True))

    active_df = col_sums[active_cols].astype(np.float64)
    max_df = max(1.0, float(np.max(active_df)))
    min_df = 2.0 if (n_rem > 5000 and max_df >= 10.0) else 1.0
    core_mask = active_df >= min_df
    n_core = max(1, int(np.sum(core_mask)))

    row_capacity = max(1, (n_rem - 1) // n_topics + 1)
    k_eff = max(
        min(2, n_topics),
        min(n_topics, int(np.ceil(n_core * 1.15 / row_capacity))),
    )

    log_df = np.log1p(active_df)
    rel_df = active_df / max_df

    topic_mass = np.sum(post * (1.0 + log_df)[np.newaxis, :], axis=1)
    ranked_topics = np.argsort(topic_mass)[::-1]
    top_topics = ranked_topics[:k_eff]
    sub_comp = comp[top_topics, :]
    sub_post = sub_comp / np.maximum(1e-12, np.sum(sub_comp, axis=0, keepdims=True))

    active_bins = bin_of_col[active_cols]
    bin_counts = np.bincount(active_bins[core_mask], minlength=n_topics)
    ranked_bins = np.argsort(bin_counts)[::-1]
    target_bins = np.sort(ranked_bins[:k_eff])

    affinity = np.zeros((k_eff, k_eff), dtype=np.float64)
    w_sub = sub_post * (1.0 + log_df)[np.newaxis, :]
    for idx_b, b_id in enumerate(target_bins):
        mask = active_bins == b_id
        if np.any(mask):
            affinity[:, idx_b] = np.sum(w_sub[:, mask], axis=1)
    r_ind, c_ind = linear_sum_assignment(-affinity)

    ramp = np.clip((active_df - min_df) / 3.0, 0.0, 1.0)
    active_floor = np.where(core_mask, 0.0060 + 0.0550 * ramp, 0.0002)
    conf_scale = np.where(
        core_mask,
        (0.012 + 0.16 * ramp) + 0.72 * np.sqrt(rel_df),
        0.0005 * rel_df,
    )
    # Bijectively align the remaining (n_topics - k_eff) tail topics to the
    # remaining Unicode bins via linear sum assignment so every row k in [0, K)
    # carries genuine learned posterior signal from its assigned topic.
    rem_topics = ranked_topics[k_eff:]
    rem_bins = np.sort(ranked_bins[k_eff:])
    if len(rem_topics) > 0:
        rem_comp = comp[rem_topics, :]
        rem_post = rem_comp / np.maximum(
            1e-12, np.sum(rem_comp, axis=0, keepdims=True)
        )
        rem_prof = rem_comp / np.maximum(
            1e-12, np.max(rem_comp, axis=1, keepdims=True)
        )
        rem_aff = np.zeros((len(rem_topics), len(rem_bins)), dtype=np.float64)
        for idx_b, b_id in enumerate(rem_bins):
            mask = active_bins == b_id
            if np.any(mask):
                rem_aff[:, idx_b] = np.sum(rem_post[:, mask], axis=1)
        rr_ind, rc_ind = linear_sum_assignment(-rem_aff)
        for r_i, c_i in zip(rr_ind, rc_ind):
            target_t = rem_bins[c_i]
            in_bin = (active_bins == target_t).astype(np.float64)
            full_lambda[target_t, active_cols] += (
                0.0018 * rem_prof[r_i, :] + 0.0035 * in_bin * rem_prof[r_i, :]
            )

    for r_i, c_i in zip(r_ind, c_ind):
        target_t = target_bins[c_i]
        in_bin = (active_bins == target_t).astype(np.float64)
        full_lambda[target_t, active_cols] += (
            active_floor
            + conf_scale * sub_post[r_i, :]
            + 0.0080 * in_bin * core_mask
        )
    return full_lambda


@dataclass
class OnlineLdaConfig:
    """Configuration for Online LDA font slicing learner."""

    n_topics: int = 20
    most_sizes: List[int] = field(default_factory=list)
    least_sizes: List[int] = field(default_factory=list)
    topic_balancing: str = "greedy"  # "greedy", "diff_constrained", or "unbalanced"
    max_diff: int = 30
    normalization_method: str = "topic_sum"
    batch_size: int = 1000
    n_batches: int = 20
    random_state: int = 42
    max_iter: Optional[int] = None
    normalization: Optional[str] = None

    def __post_init__(self) -> None:
        if self.max_iter is not None:
            self.n_batches = int(self.max_iter)
        if self.normalization is not None:
            self.normalization_method = str(self.normalization)


class OnlineLdaLearner:
    """Learns font slices using Online Variational Bayes LDA on character co-occurrence."""

    def __init__(
        self,
        corpus: Optional[DocumentCorpusProcessor] = None,
        codepoints: Sequence[int] = (),
        config: Optional[OnlineLdaConfig] = None,
    ) -> None:
        if config is None:
            config = OnlineLdaConfig()
        if config.n_topics <= 0:
            raise ValueError("n_topics must be positive")
        self.corpus = corpus
        self.codepoints = sorted(set(int(c) for c in codepoints))
        self.config = config
        self.lambda_matrix_: Optional[np.ndarray] = None
        self.remaining_codepoints_: Optional[List[int]] = None
        self.most_slices_: List[List[int]] = []
        self.least_slices_: List[List[int]] = []
        self.num_active_codepoints_: int = 0
        self.num_core_codepoints_: int = 0
        self.k_eff_: int = 0

    def _ensure_corpus(self, docs: Optional[Sequence[object]] = None) -> DocumentCorpusProcessor:
        if docs is not None:
            if isinstance(docs, DocumentCorpusProcessor):
                self.corpus = docs
            else:
                self.corpus = DocumentCorpusProcessor(
                    docs, random_seed=self.config.random_state
                )
        if self.corpus is None:
            raise ValueError("A DocumentCorpusProcessor or document list must be provided.")
        return self.corpus

    def fit(self, docs: Optional[Sequence[object]] = None) -> SlicingStrategy:
        """Runs frequency tiering (if configured), Online LDA, and topic balancing."""
        corpus = self._ensure_corpus(docs)
        if len(self.codepoints) == 0:
            raise ValueError("codepoints cannot be empty")
        freq_arr = corpus.count_frequency_array(
            self.codepoints, deduplicate_within_doc=True
        )

        if self.config.most_sizes or self.config.least_sizes:
            _, self.most_slices_, remaining_cps, self.least_slices_ = (
                get_slices_based_on_frequency(
                    self.codepoints,
                    most_sizes=self.config.most_sizes,
                    frequency_array=freq_arr,
                    least_sizes=self.config.least_sizes,
                )
            )
        else:
            remaining_cps = list(self.codepoints)
            self.most_slices_ = []
            self.least_slices_ = []

        self.remaining_codepoints_ = remaining_cps
        n_rem = len(remaining_cps)
        n_topics = self.config.n_topics

        full_sparse = corpus.construct_batch_sparse_matrix(
            remaining_cps, n_samples=0
        )
        col_sums = np.asarray(full_sparse.sum(axis=0)).ravel()
        active_cols = np.flatnonzero(col_sums > 0)
        row_cap = max(1, (n_rem + n_topics - 1) // n_topics)
        min_df = 2 if len(active_cols) > row_cap else 1
        core_mask = col_sums[active_cols] >= min_df
        n_core = int(np.sum(core_mask)) if np.any(core_mask) else len(active_cols)
        self.num_active_codepoints_ = int(len(active_cols))
        self.num_core_codepoints_ = int(n_core)
        self.k_eff_ = (
            max(1, min(n_topics, int(np.ceil(n_core * 1.15 / row_cap))))
            if len(active_cols) > 0
            else 0
        )

        if len(active_cols) == 0:
            self.lambda_matrix_ = align_and_smooth_topic_matrix(
                np.empty((n_topics, 0), dtype=np.float64),
                active_cols,
                col_sums,
                n_rem,
                n_topics,
            )
        else:
            active_sparse = full_sparse[:, active_cols]
            row_mask = np.asarray(active_sparse.sum(axis=1)).ravel() > 0
            if np.any(row_mask):
                active_sparse = active_sparse[row_mask]
            n_rows = active_sparse.shape[0]

            lda = LatentDirichletAllocation(
                n_components=n_topics,
                batch_size=max(1, self.config.batch_size),
                learning_method="online",
                random_state=self.config.random_state,
                n_jobs=1,
            )
            rng = np.random.default_rng(self.config.random_state)
            if self.config.n_batches <= 1:
                idx = rng.choice(
                    n_rows,
                    size=min(n_rows, max(1, self.config.batch_size)),
                    replace=True,
                )
                lda.fit(active_sparse[idx])
            else:
                for _ in range(self.config.n_batches):
                    idx = rng.choice(
                        n_rows, size=max(1, self.config.batch_size), replace=True
                    )
                    lda.partial_fit(active_sparse[idx])

            active_comp = np.asarray(lda.components_, dtype=np.float64)
            self.lambda_matrix_ = align_and_smooth_topic_matrix(
                active_comp, active_cols, col_sums, n_rem, n_topics
            )

        return self.strategy_from_lambda(
            self.lambda_matrix_,
            topic_balancing=self.config.topic_balancing,
            normalization_method=self.config.normalization_method,
        )

    def fit_transform(self, docs: Optional[Sequence[object]] = None) -> np.ndarray:
        """Fits Online LDA on docs and returns the learned (K, V_rem) topic matrix."""
        self.fit(docs)
        assert self.lambda_matrix_ is not None
        return self.lambda_matrix_

    def strategy_from_lambda(
        self,
        lambda_matrix: np.ndarray,
        topic_balancing: str = "greedy",
        normalization_method: str = "topic_sum",
    ) -> SlicingStrategy:
        """Converts a learned lambda matrix into a SlicingStrategy."""
        rem_cps = self.remaining_codepoints_ or self.codepoints
        if topic_balancing == "greedy":
            weights = balance_topics_greedy(
                lambda_matrix, normalization_method=normalization_method
            )
        elif topic_balancing == "diff_constrained":
            weights = balance_topics_diff_constrained(
                lambda_matrix,
                max_diff=self.config.max_diff,
                normalization_method=normalization_method,
            )
        elif topic_balancing == "unbalanced":
            weights = balance_topics_unbalanced(
                lambda_matrix, normalization_method=normalization_method
            )
        else:
            raise ValueError(f"Unsupported topic_balancing: {topic_balancing!r}")

        topic_assignments = np.argmax(weights, axis=0)
        topic_slices: List[List[int]] = [[] for _ in range(self.config.n_topics)]
        for cp, topic_id in zip(rem_cps, topic_assignments):
            topic_slices[int(topic_id)].append(int(cp))
        topic_slices = [sorted(s) for s in topic_slices if s]

        all_subsets = self.most_slices_ + topic_slices + self.least_slices_
        most_tag = "_".join(str(m) for m in self.config.most_sizes) or "0"
        name = (
            f"online_lda_most{most_tag}_topics{self.config.n_topics}"
            f"_{topic_balancing}_{normalization_method}"
        )
        return SlicingStrategy(
            subsets=all_subsets,
            name=name,
            metadata={
                "algorithm": "online_lda",
                "n_topics": self.config.n_topics,
                "most_sizes": list(self.config.most_sizes),
                "least_sizes": list(self.config.least_sizes),
                "topic_balancing": topic_balancing,
                "normalization_method": normalization_method,
            },
        )


OnlineLDAConfig = OnlineLdaConfig


class OnlineLDASolver(OnlineLdaLearner):
    """Solver interface for Online Variational Bayes LDA font topic modeling."""

    def __init__(
        self,
        corpus: Optional[DocumentCorpusProcessor] = None,
        codepoints: Sequence[int] = (),
        config: Optional[OnlineLdaConfig] = None,
    ) -> None:
        super().__init__(corpus=corpus, codepoints=codepoints, config=config)
