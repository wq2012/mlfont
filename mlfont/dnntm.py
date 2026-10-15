"""Deep Neural Network Topic Model (DNNTM) for font slicing."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
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
from mlfont.olda import align_and_smooth_topic_matrix
from mlfont.schemas import SlicingStrategy


class DnnTmLossType(str, Enum):
    """Supported training loss objectives for DnnTmLearner."""

    REAL_FAKE_DIFF = "REAL_FAKE_DIFF"
    REAL_FAKE_AND_REAL_AVERAGE_DIFF = "REAL_FAKE_AND_REAL_AVERAGE_DIFF"
    ALL_COMBINED = "ALL_COMBINED"
    AUTOENCODER = "AUTOENCODER"
    HYBRID_AUTOENCODER = "HYBRID_AUTOENCODER"


DNNTMLossType = DnnTmLossType


@dataclass
class DnnTmConfig:
    """Configuration for the Deep Neural Network Topic Model (DNNTM)."""

    n_topics: int = 20
    most_sizes: List[int] = field(default_factory=list)
    least_sizes: List[int] = field(default_factory=list)
    topic_balancing: str = "greedy"
    max_diff: int = 30
    normalization_method: str = "topic_sum"
    batch_size: int = 256
    n_layers: int = 2
    layer_size: int = 128
    fake_doc_word_density: float = 0.05
    training_steps: int = 500
    init_learning_rate: float = 0.01
    learning_rate_decay: float = 0.95
    learning_rate_decay_steps: int = 100
    loss_type: str = "ALL_COMBINED"
    activation: str = "ELU"  # RELU, ELU, TANH, SIGMOID
    l1_sparsity_weight: float = 0.05
    l2_reg_weight: float = 1e-4
    recon_weight: float = 0.5
    random_seed: int = 42
    max_iter: Optional[int] = None
    normalization: Optional[str] = None
    random_state: Optional[int] = None

    def __post_init__(self) -> None:
        if self.max_iter is not None:
            self.training_steps = int(self.max_iter)
        if self.normalization is not None:
            self.normalization_method = str(self.normalization)
        if self.random_state is not None:
            self.random_seed = int(self.random_state)


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
    """Feedforward topic encoder with optional nonnegative generative decoder."""

    def __init__(
        self,
        n_words: int,
        n_topics: int,
        n_layers: int = 2,
        layer_size: int = 128,
        activation: str = "ELU",
        use_decoder: bool = False,
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
        self.use_decoder = bool(use_decoder)
        if self.use_decoder:
            self.decoder_weight = nn.Parameter(torch.empty(n_topics, n_words))
            nn.init.uniform_(self.decoder_weight, 0.001, 0.05)
        else:
            self.register_parameter("decoder_weight", None)

    def forward(self, docs: torch.Tensor) -> torch.Tensor:
        return self.net(docs)

    def decode(self, topics: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Reconstructs normalized document word distributions via nonnegative decoder."""
        if self.decoder_weight is None:
            raise ValueError("Decoder is not enabled on this DnnTopicNetwork")
        w_pos = torch.clamp(self.decoder_weight, min=eps)
        w_norm = w_pos / torch.sum(w_pos, dim=1, keepdim=True)
        return torch.matmul(topics, w_norm)

    def project_nonnegative(self) -> None:
        """Projects the generative decoder weights onto R_+."""
        if self.decoder_weight is not None:
            with torch.no_grad():
                self.decoder_weight.clamp_(min=0.0)


class DnnTmLearner:
    """Trains a Deep Neural Network Topic Model (Entropy-Diff or Nonnegative Autoencoder)."""


    def __init__(
        self,
        corpus: Optional[DocumentCorpusProcessor] = None,
        codepoints: Sequence[int] = (),
        config: Optional[DnnTmConfig] = None,
    ) -> None:
        if config is None:
            config = DnnTmConfig()
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

    def _ensure_corpus(self, docs: Optional[Sequence[object]] = None) -> DocumentCorpusProcessor:
        if docs is not None:
            if isinstance(docs, DocumentCorpusProcessor):
                self.corpus = docs
            else:
                self.corpus = DocumentCorpusProcessor(
                    docs, random_seed=self.config.random_seed
                )
        if self.corpus is None:
            raise ValueError("A DocumentCorpusProcessor or document list must be provided.")
        return self.corpus

    def fit(self, docs: Optional[Sequence[object]] = None) -> SlicingStrategy:
        """Trains the DNNTM model and returns the resulting SlicingStrategy."""
        corpus = self._ensure_corpus(docs)
        torch.manual_seed(self.config.random_seed)
        np.random.seed(self.config.random_seed)

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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cpu" and torch.get_num_threads() > 4:
            torch.set_num_threads(4)

        loss_type = self.config.loss_type.upper()
        valid_losses = (
            "REAL_FAKE_DIFF",
            "REAL_FAKE_AND_REAL_AVERAGE_DIFF",
            "ALL_COMBINED",
            "AUTOENCODER",
            "HYBRID_AUTOENCODER",
        )
        if loss_type not in valid_losses:
            raise ValueError(f"Unsupported loss_type: {self.config.loss_type!r}")
        use_decoder = loss_type in ("AUTOENCODER", "HYBRID_AUTOENCODER")

        full_sparse = corpus.construct_batch_sparse_matrix(
            remaining_cps, n_samples=0
        )
        col_sums = np.asarray(full_sparse.sum(axis=0)).ravel()
        active_cols = np.flatnonzero(col_sums > 0)
        if len(active_cols) == 0:
            active_cols = np.arange(n_rem, dtype=np.int64)
            from scipy import sparse as sp

            active_sparse = sp.csr_matrix(np.ones((1, n_rem), dtype=np.uint8))
        elif len(active_cols) == n_rem:
            active_sparse = full_sparse
        else:
            active_sparse = full_sparse[:, active_cols]
            row_mask = np.asarray(active_sparse.sum(axis=1)).ravel() > 0
            if np.any(row_mask):
                active_sparse = active_sparse[row_mask]

        n_words = len(active_cols)
        model = DnnTopicNetwork(
            n_words=n_words,
            n_topics=n_topics,
            n_layers=self.config.n_layers,
            layer_size=self.config.layer_size,
            activation=self.config.activation,
            use_decoder=use_decoder,
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
        log_k = float(math.log(max(2, n_topics)))

        self.loss_history_ = []
        for _ in range(self.config.training_steps):
            batch_idx = torch.randint(0, n_total_docs, (b_size,), device=device)
            real_docs = full_docs_tensor[batch_idx]

            optimizer.zero_grad()
            real_topics = model(real_docs)
            loss_real = torch.mean(_shannon_entropy(real_topics))
            avg_real_topic = torch.mean(real_topics, dim=0, keepdim=True)
            loss_real_avg = torch.mean(_shannon_entropy(avg_real_topic))

            loss_ent = torch.tensor(0.0, device=device)
            if loss_type != "AUTOENCODER":
                fake_docs = (
                    torch.rand((b_size, n_words), device=device) < density
                ).float()
                fake_topics = model(fake_docs)
                loss_fake = torch.mean(_shannon_entropy(fake_topics))

                if loss_type == "REAL_FAKE_DIFF":
                    loss_ent = (loss_real - loss_fake) + log_k
                elif loss_type == "REAL_FAKE_AND_REAL_AVERAGE_DIFF":
                    loss_ent = (loss_real - loss_fake + log_k) + (
                        loss_real - loss_real_avg + log_k
                    )
                else:
                    if n_words <= 256:
                        eye_batch = torch.eye(n_words, device=device)
                    else:
                        idx = torch.randperm(n_words, device=device)[:256]
                        eye_batch = torch.zeros((len(idx), n_words), device=device)
                        eye_batch[torch.arange(len(idx), device=device), idx] = 1.0
                    word_topics = model(eye_batch)
                    loss_word = torch.mean(_shannon_entropy(word_topics))
                    avg_word_topic = torch.mean(word_topics, dim=0, keepdim=True)
                    loss_word_avg = torch.mean(_shannon_entropy(avg_word_topic))
                    loss_ent = (
                        (loss_real - loss_fake + log_k)
                        + (loss_real - loss_real_avg + log_k)
                        + (loss_word - loss_word_avg + log_k)
                    )

            loss_ae = torch.tensor(0.0, device=device)
            if use_decoder:
                recon_probs = torch.clamp(model.decode(real_topics), min=1e-12, max=1.0)
                doc_sums = torch.clamp(torch.sum(real_docs, dim=-1, keepdim=True), min=1.0)
                target_probs = real_docs / doc_sums
                kl_div = torch.sum(
                    target_probs
                    * (
                        torch.log(torch.clamp(target_probs, min=1e-12))
                        - torch.log(recon_probs)
                    ),
                    dim=-1,
                )
                loss_recon = torch.clamp(torch.mean(kl_div), min=0.0)
                loss_sparse = self.config.l1_sparsity_weight * torch.clamp(
                    loss_real + (log_k - loss_real_avg), min=0.0
                )
                assert model.decoder_weight is not None
                loss_l2 = self.config.l2_reg_weight * torch.mean(
                    model.decoder_weight ** 2
                )
                loss_ae = loss_recon + loss_sparse + loss_l2

            if loss_type == "AUTOENCODER":
                loss = loss_ae
            elif loss_type == "HYBRID_AUTOENCODER":
                loss = loss_ent + self.config.recon_weight * loss_ae
            else:
                loss = loss_ent

            loss.backward()
            optimizer.step()
            if use_decoder:
                model.project_nonnegative()
            scheduler.step()
            self.loss_history_.append(float(max(0.0, loss.detach().cpu().item())))

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

        enc_topic_word = np.vstack(word_probs_chunks).T  # (n_topics, n_words)
        if use_decoder and model.decoder_weight is not None:
            with torch.no_grad():
                dec_w = (
                    torch.clamp(model.decoder_weight, min=1e-12)
                    .cpu()
                    .numpy()
                    .astype(np.float64)
                )
            dec_col = dec_w / np.maximum(1e-12, np.sum(dec_w, axis=0, keepdims=True))
            enc_col = enc_topic_word / np.maximum(
                1e-12, np.sum(enc_topic_word, axis=0, keepdims=True)
            )
            alpha_enc = 0.35 if loss_type == "AUTOENCODER" else 0.75
            raw_topic_word = alpha_enc * enc_col + (1.0 - alpha_enc) * dec_col
        else:
            raw_topic_word = enc_topic_word

        self.lambda_matrix_ = align_and_smooth_topic_matrix(
            raw_topic_word, active_cols, col_sums, n_rem, n_topics
        )
        col_norm = self.lambda_matrix_ / np.maximum(
            1e-12, np.sum(self.lambda_matrix_, axis=0, keepdims=True)
        )
        self.word_topic_probs_ = col_norm.T

        if self.config.topic_balancing == "greedy":
            balanced = balance_topics_greedy(
                self.lambda_matrix_,
                normalization_method=self.config.normalization_method,
            )
        elif self.config.topic_balancing == "diff_constrained":
            balanced = balance_topics_diff_constrained(
                self.lambda_matrix_,
                max_diff=self.config.max_diff,
                normalization_method=self.config.normalization_method,
            )
        elif self.config.topic_balancing == "unbalanced":
            balanced = balance_topics_unbalanced(
                self.lambda_matrix_,
                normalization_method=self.config.normalization_method,
            )
        else:
            raise ValueError(
                f"Unsupported topic_balancing: {self.config.topic_balancing!r}"
            )

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
                "most_sizes": list(self.config.most_sizes),
                "least_sizes": list(self.config.least_sizes),
                "loss_type": self.config.loss_type,
                "training_steps": self.config.training_steps,
                "final_loss": self.loss_history_[-1] if self.loss_history_ else None,
            },
        )

    def fit_transform(self, docs: Optional[Sequence[object]] = None) -> np.ndarray:
        """Fits DNNTM on docs and returns the learned (K, V_rem) topic matrix."""
        self.fit(docs)
        assert self.lambda_matrix_ is not None
        return self.lambda_matrix_


DNNTMConfig = DnnTmConfig


class DNNTMSolver(DnnTmLearner):
    """Solver interface for Deep Neural Network Topic Model (DNNTM) font slicing."""

    def __init__(
        self,
        corpus: Optional[DocumentCorpusProcessor] = None,
        codepoints: Sequence[int] = (),
        config: Optional[DnnTmConfig] = None,
    ) -> None:
        super().__init__(corpus=corpus, codepoints=codepoints, config=config)
