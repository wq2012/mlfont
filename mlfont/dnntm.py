"""Deep Neural Network Topic Model (DNNTM) for font slicing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np
import torch
from torch import nn

from mlfont.data import DocumentCorpusProcessor
from mlfont.frequency import get_slices_based_on_frequency
from mlfont.maths import (
    balance_topics_diff_constrained,
    balance_topics_greedy,
    balance_topics_unbalanced,
)
from mlfont.schemas import SlicingStrategy


@dataclass
class DnnTmConfig:
    """Configuration for the Deep Neural Network Topic Model (DNNTM)."""

    n_topics: int = 20
    most_sizes: List[int] = field(default_factory=list)
    least_sizes: List[int] = field(default_factory=list)
    topic_balancing: str = "greedy"
    normalization_method: str = "topic_sum"
    batch_size: int = 256
    n_layers: int = 2
    layer_size: int = 128
    fake_doc_word_density: float = 0.05
    training_steps: int = 500
    init_learning_rate: float = 0.01
    learning_rate_decay: float = 0.95
    learning_rate_decay_steps: int = 100
    loss_type: str = "ALL_COMBINED"  # REAL_FAKE_DIFF, REAL_FAKE_AND_REAL_AVERAGE_DIFF, ALL_COMBINED
    activation: str = "ELU"  # RELU, ELU, TANH, SIGMOID
    random_seed: int = 42


def _make_activation(name: str) -> nn.Module:
    upper = name.upper()
    if upper == "RELU":
        return nn.ReLU()
    if upper == "ELU":
        return nn.ELU()
    if upper == "TANH":
        return nn.Tanh()
    if upper == "SIGMOID":
        return nn.Sigmoid()
    raise ValueError(f"Unsupported activation: {name!r}")


def _shannon_entropy(probs: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Computes row-wise Shannon entropy -sum(p * log(p))."""
    clamped = torch.clamp(probs, min=eps, max=1.0)
    return -torch.sum(clamped * torch.log(clamped), dim=-1)


class DnnTopicNetwork(nn.Module):
    """Feedforward network mapping binary document vectors to topic distributions."""

    def __init__(
        self,
        n_words: int,
        n_topics: int,
        n_layers: int = 2,
        layer_size: int = 128,
        activation: str = "ELU",
    ) -> None:
        super().__init__()
        if n_layers < 1:
            raise ValueError("n_layers must be >= 1")
        layers: List[nn.Module] = []
        in_dim = n_words
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(in_dim, layer_size))
            layers.append(_make_activation(activation))
            in_dim = layer_size
        layers.append(nn.Linear(in_dim, n_topics))
        layers.append(nn.Softmax(dim=-1))
        self.net = nn.Sequential(*layers)

    def forward(self, docs: torch.Tensor) -> torch.Tensor:
        return self.net(docs)


class DnnTmLearner:
    """Trains a Deep Neural Network Topic Model using Entropy Difference Loss."""

    def __init__(
        self,
        corpus: DocumentCorpusProcessor,
        codepoints: Sequence[int],
        config: DnnTmConfig,
    ) -> None:
        if config.n_topics <= 0:
            raise ValueError("n_topics must be positive")
        self.corpus = corpus
        self.codepoints = sorted(set(int(c) for c in codepoints))
        self.config = config
        self.word_topic_probs_: Optional[np.ndarray] = None
        self.lambda_matrix_: Optional[np.ndarray] = None
        self.loss_history_: List[float] = []
        self.remaining_codepoints_: Optional[List[int]] = None
        self.most_slices_: List[List[int]] = []
        self.least_slices_: List[List[int]] = []

    def fit(self) -> SlicingStrategy:
        """Trains the DNNTM model and returns the resulting SlicingStrategy."""
        torch.manual_seed(self.config.random_seed)
        np.random.seed(self.config.random_seed)

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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        loss_type = self.config.loss_type.upper()
        if loss_type not in (
            "REAL_FAKE_DIFF",
            "REAL_FAKE_AND_REAL_AVERAGE_DIFF",
            "ALL_COMBINED",
        ):
            raise ValueError(f"Unsupported loss_type: {self.config.loss_type!r}")

        full_sparse = self.corpus.construct_batch_sparse_matrix(
            remaining_cps, n_samples=0
        )
        col_sums = np.asarray(full_sparse.sum(axis=0)).ravel()
        active_cols = np.flatnonzero(col_sums > 0)
        if len(active_cols) == 0:
            active_cols = np.arange(n_rem, dtype=np.int64)
            from scipy import sparse as sp

            active_sparse = sp.csr_matrix(np.ones((1, n_rem), dtype=np.uint8))
            active_freq_scale = np.ones((1, n_rem), dtype=np.float64)
        elif len(active_cols) == n_rem:
            active_sparse = full_sparse
            active_freq_scale = np.log1p(col_sums.astype(np.float64))[np.newaxis, :]
        else:
            active_sparse = full_sparse[:, active_cols]
            row_mask = np.asarray(active_sparse.sum(axis=1)).ravel() > 0
            active_sparse = active_sparse[row_mask]
            active_freq_scale = np.log1p(col_sums[active_cols].astype(np.float64))[
                np.newaxis, :
            ]

        n_words = len(active_cols)
        model = DnnTopicNetwork(
            n_words=n_words,
            n_topics=n_topics,
            n_layers=self.config.n_layers,
            layer_size=self.config.layer_size,
            activation=self.config.activation,
        ).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(), lr=self.config.init_learning_rate
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=max(1, self.config.learning_rate_decay_steps),
            gamma=self.config.learning_rate_decay,
        )

        full_docs_tensor = torch.from_numpy(
            active_sparse.toarray().astype(np.float32)
        ).to(device)
        n_total_docs = full_docs_tensor.shape[0]
        b_size = max(1, self.config.batch_size)
        density = float(self.config.fake_doc_word_density)

        self.loss_history_ = []
        for _ in range(self.config.training_steps):
            batch_idx = torch.randint(0, n_total_docs, (b_size,), device=device)
            real_docs = full_docs_tensor[batch_idx]
            fake_docs = (
                torch.rand((b_size, n_words), device=device) < density
            ).float()

            optimizer.zero_grad()
            real_topics = model(real_docs)
            fake_topics = model(fake_docs)

            loss_real = torch.mean(_shannon_entropy(real_topics))
            loss_fake = torch.mean(_shannon_entropy(fake_topics))

            if loss_type == "REAL_FAKE_DIFF":
                loss = loss_real - loss_fake
            else:
                avg_real_topic = torch.mean(real_topics, dim=0, keepdim=True)
                loss_real_avg = torch.mean(_shannon_entropy(avg_real_topic))
                if loss_type == "REAL_FAKE_AND_REAL_AVERAGE_DIFF":
                    loss = loss_real - loss_fake - loss_real_avg
                else:
                    if n_words <= 128:
                        eye_batch = torch.eye(n_words, device=device)
                    else:
                        idx = torch.randperm(n_words, device=device)[:128]
                        eye_batch = torch.zeros((len(idx), n_words), device=device)
                        eye_batch[torch.arange(len(idx), device=device), idx] = 1.0
                    word_topics = model(eye_batch)
                    loss_word = torch.mean(_shannon_entropy(word_topics))
                    loss = loss_real - loss_fake - loss_real_avg + loss_word

            loss.backward()
            optimizer.step()
            scheduler.step()
            self.loss_history_.append(float(loss.detach().cpu().item()))

        # Compute word-topic probability matrix for active columns in chunks
        model.eval()
        word_probs_chunks: List[np.ndarray] = []
        chunk_size = 1024
        with torch.no_grad():
            for start in range(0, n_words, chunk_size):
                end = min(start + chunk_size, n_words)
                eye_chunk = torch.zeros((end - start, n_words), device=device)
                eye_chunk[
                    torch.arange(end - start, device=device),
                    torch.arange(start, end, device=device),
                ] = 1.0
                probs_chunk = model(eye_chunk).cpu().numpy().astype(np.float64)
                word_probs_chunks.append(probs_chunk)

        active_probs = np.vstack(word_probs_chunks)  # (n_words, n_topics)
        if n_words == n_rem:
            self.word_topic_probs_ = active_probs
            self.lambda_matrix_ = active_probs.T * active_freq_scale
        else:
            full_probs = np.full((n_rem, n_topics), 1.0 / n_topics, dtype=np.float64)
            full_probs[active_cols, :] = active_probs
            self.word_topic_probs_ = full_probs
            row_decay = 1e-4 * (
                1.0 - 1e-3 * (np.arange(n_topics, dtype=np.float64)[:, np.newaxis] / max(1, n_topics))
            )
            full_lambda = np.broadcast_to(row_decay, (n_topics, n_rem)).copy()
            full_lambda[:, active_cols] = (1e-2 + active_probs.T) * active_freq_scale
            self.lambda_matrix_ = full_lambda

        if self.config.topic_balancing == "greedy":
            balanced = balance_topics_greedy(
                self.lambda_matrix_,
                normalization_method=self.config.normalization_method,
            )
        elif self.config.topic_balancing == "diff_constrained":
            balanced = balance_topics_diff_constrained(
                self.lambda_matrix_,
                max_diff=30,
                normalization_method=self.config.normalization_method,
            )
        elif self.config.topic_balancing == "unbalanced":
            balanced = balance_topics_unbalanced(
                self.lambda_matrix_,
                normalization_method=self.config.normalization_method,
            )
        else:
            raise ValueError(f"Unsupported topic_balancing: {self.config.topic_balancing!r}")

        assignments = np.argmax(balanced, axis=0)
        topic_slices: List[List[int]] = [[] for _ in range(self.config.n_topics)]
        for cp, topic_id in zip(remaining_cps, assignments):
            topic_slices[int(topic_id)].append(int(cp))
        topic_slices = [sorted(s) for s in topic_slices if s]

        all_subsets = self.most_slices_ + topic_slices + self.least_slices_
        most_tag = "_".join(str(m) for m in self.config.most_sizes) or "0"
        return SlicingStrategy(
            subsets=all_subsets,
            name=f"dnntm_most{most_tag}_topics{self.config.n_topics}_{self.config.topic_balancing}",
            metadata={
                "algorithm": "dnntm",
                "n_topics": self.config.n_topics,
                "loss_type": self.config.loss_type,
                "training_steps": self.config.training_steps,
                "final_loss": self.loss_history_[-1] if self.loss_history_ else None,
            },
        )
