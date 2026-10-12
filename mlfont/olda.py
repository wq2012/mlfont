"""Online Latent Dirichlet Allocation (Online LDA) font slicing learner."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np
from sklearn.decomposition import LatentDirichletAllocation

from mlfont.data import DocumentCorpusProcessor
from mlfont.frequency import get_slices_based_on_frequency
from mlfont.maths import (
    balance_topics_diff_constrained,
    balance_topics_greedy,
    balance_topics_unbalanced,
)
from mlfont.schemas import SlicingStrategy


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


class OnlineLdaLearner:
    """Learns font slices using Online Variational Bayes LDA on character co-occurrence."""

    def __init__(
        self,
        corpus: DocumentCorpusProcessor,
        codepoints: Sequence[int],
        config: OnlineLdaConfig,
    ) -> None:
        if config.n_topics <= 0:
            raise ValueError("n_topics must be positive")
        self.corpus = corpus
        self.codepoints = sorted(set(int(c) for c in codepoints))
        self.config = config
        self.lambda_matrix_: Optional[np.ndarray] = None
        self.remaining_codepoints_: Optional[List[int]] = None
        self.most_slices_: List[List[int]] = []
        self.least_slices_: List[List[int]] = []

    def fit(self) -> SlicingStrategy:
        """Runs frequency tiering (if configured), Online LDA, and topic balancing."""
        freq_arr = self.corpus.count_frequency_array(
            self.codepoints, deduplicate_within_doc=True
        )
        if int(np.sum(freq_arr)) == 0:
            raise ValueError("No documents contain any target codepoints")

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

        full_sparse = self.corpus.construct_batch_sparse_matrix(
            remaining_cps, n_samples=0
        )
        col_sums = np.asarray(full_sparse.sum(axis=0)).ravel()
        active_cols = np.flatnonzero(col_sums > 0)

        if len(active_cols) == 0:
            full_lambda = np.full((n_topics, n_rem), 1.0 / n_topics, dtype=np.float64)
            for j in range(n_rem):
                full_lambda[min(n_topics - 1, (j * n_topics) // max(1, n_rem)), j] += 1e-5
            self.lambda_matrix_ = full_lambda
        elif len(active_cols) == n_rem:
            lda = LatentDirichletAllocation(
                n_components=n_topics,
                batch_size=max(1, self.config.batch_size),
                learning_method="online",
                random_state=self.config.random_state,
                n_jobs=1,
            )
            n_rows = full_sparse.shape[0]
            rng = np.random.default_rng(self.config.random_state)
            if self.config.n_batches <= 1:
                idx = rng.choice(
                    n_rows, size=min(n_rows, max(1, self.config.batch_size)), replace=True
                )
                lda.fit(full_sparse[idx])
            else:
                for _ in range(self.config.n_batches):
                    idx = rng.choice(n_rows, size=max(1, self.config.batch_size), replace=True)
                    lda.partial_fit(full_sparse[idx])
            self.lambda_matrix_ = np.asarray(lda.components_, dtype=np.float64)
        else:
            active_sparse = full_sparse[:, active_cols]
            row_mask = np.asarray(active_sparse.sum(axis=1)).ravel() > 0
            active_sparse = active_sparse[row_mask]
            n_rows = active_sparse.shape[0]

            row_capacity = max(1, (n_rem - 1) // n_topics + 1)
            k_active = max(
                min(2, n_topics),
                min(n_topics, int(np.ceil(len(active_cols) * 1.15 / row_capacity))),
            )
            lda = LatentDirichletAllocation(
                n_components=k_active,
                batch_size=max(1, self.config.batch_size),
                learning_method="online",
                random_state=self.config.random_state,
                n_jobs=1,
            )
            rng = np.random.default_rng(self.config.random_state)
            if self.config.n_batches <= 1:
                idx = rng.choice(
                    n_rows, size=min(n_rows, max(1, self.config.batch_size)), replace=True
                )
                lda.fit(active_sparse[idx])
            else:
                for _ in range(self.config.n_batches):
                    idx = rng.choice(n_rows, size=max(1, self.config.batch_size), replace=True)
                    lda.partial_fit(active_sparse[idx])

            active_comp = np.asarray(lda.components_, dtype=np.float64)
            active_probs = active_comp / np.maximum(
                1e-12, np.sum(active_comp, axis=0, keepdims=True)
            )
            active_freq_scale = np.log1p(col_sums[active_cols].astype(np.float64))[
                np.newaxis, :
            ]
            row_decay = 1e-4 * (
                1.0 - 1e-3 * (np.arange(n_topics, dtype=np.float64)[:, np.newaxis] / max(1, n_topics))
            )
            full_lambda = np.broadcast_to(row_decay, (n_topics, n_rem)).copy()
            full_lambda[:k_active, active_cols] = (1e-2 + active_probs) * active_freq_scale
            self.lambda_matrix_ = full_lambda

        return self.strategy_from_lambda(
            self.lambda_matrix_,
            topic_balancing=self.config.topic_balancing,
            normalization_method=self.config.normalization_method,
        )

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
