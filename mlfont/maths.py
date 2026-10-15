"""Mathematical utilities: matrix normalization, parameter decay, and topic balancing."""

from __future__ import annotations

import abc

import numpy as np

from mlfont.schemas import DecayConfig

NORMALIZATION_METHODS = (
    "topic_sum",
    "row_sum",
    "global_max",
    "spectral_norm",
    "spectual_norm",  # backward-compatible alias
    "sinkhorn",
    "bistochastic",
    "none",
)


def normalize_lambda_matrix(
    matrix: np.ndarray,
    method: str = "topic_sum",
) -> np.ndarray:
    """Normalizes an (n_topics, n_codepoints) topic-codepoint weight matrix.

    Args:
        matrix: 2D float64 numpy array of shape (n_topics, n_codepoints) with
            non-negative entries and at least one positive entry.
        method: Normalization method:
            - 'topic_sum': Each column (codepoint) sums to 1.0 across topics,
              representing p(topic | codepoint) under a uniform topic prior.
            - 'row_sum': Each row (topic) sums to 1.0 across codepoints,
              representing p(codepoint | topic).
            - 'global_max': Maximum entry in the matrix is scaled to 1.0.
            - 'spectral_norm' (or 'spectual_norm'): Matrix 2-norm is scaled to 1.0.
            - 'sinkhorn' (or 'bistochastic'): Iterative Sinkhorn-Knopp
              bi-stochastic scaling so that each codepoint column sums to 1.0
              and each topic row sums to n_codepoints / n_topics.
            - 'none': Returns a copy without rescaling.
    """
    if method not in NORMALIZATION_METHODS:
        raise ValueError(f"method must be one of {NORMALIZATION_METHODS}, got {method!r}")
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")
    if matrix.ndim != 2:
        raise ValueError("matrix must be 2-dimensional")
    if not np.issubdtype(matrix.dtype, np.floating):
        raise TypeError("matrix must be a floating-point array")
    if np.min(matrix) < 0.0:
        raise ValueError("matrix must not contain negative values")
    if np.max(matrix) <= 0.0:
        raise ValueError("matrix cannot be all zeros")

    normalized = np.array(matrix, dtype=np.float64, copy=True)

    if method == "global_max":
        normalized /= np.max(normalized)
    elif method == "topic_sum":
        col_sums = np.sum(normalized, axis=0, keepdims=True)
        col_sums = np.where(col_sums > 0.0, col_sums, 1.0)
        normalized /= col_sums
    elif method == "row_sum":
        row_sums = np.sum(normalized, axis=1, keepdims=True)
        row_sums = np.where(row_sums > 0.0, row_sums, 1.0)
        normalized /= row_sums
    elif method in ("spectral_norm", "spectual_norm"):
        norm2 = np.linalg.norm(normalized, ord=2)
        if norm2 > 0.0:
            normalized /= norm2
    elif method in ("sinkhorn", "bistochastic"):
        normalized = sinkhorn_knopp_normalize(normalized)
    elif method == "none":
        pass

    return normalized


def sinkhorn_knopp_normalize(
    matrix: np.ndarray,
    n_iters: int = 25,
) -> np.ndarray:
    """Applies iterative Sinkhorn-Knopp bi-stochastic scaling to a topic-codepoint matrix.

    Alternatingly rescales topic rows to uniform slice capacity (n_codepoints / n_topics)
    and codepoint columns to unit mass 1.0.
    """
    normalized = np.array(matrix, dtype=np.float64, copy=True)
    rows, cols = normalized.shape
    target_row_sum = float(cols) / float(max(1, rows))
    eps = max(1e-12, float(np.max(normalized)) * 1e-9)
    normalized = np.maximum(normalized, eps)
    for _ in range(max(1, int(n_iters))):
        r_sums = np.sum(normalized, axis=1, keepdims=True)
        r_sums = np.where(r_sums > 0.0, r_sums, 1.0)
        normalized *= target_row_sum / r_sums
        c_sums = np.sum(normalized, axis=0, keepdims=True)
        c_sums = np.where(c_sums > 0.0, c_sums, 1.0)
        normalized /= c_sums
    return normalized



class ParamsDecayBase(abc.ABC):
    """Base class for decaying hyperparameter schedules."""

    @abc.abstractmethod
    def __call__(self, step: int) -> float:
        """Evaluates the schedule at the given step."""


class PowerDecay(ParamsDecayBase):
    """Polynomial power decay from max_val to min_val over [0, max_step]."""

    def __init__(self, config: DecayConfig, max_step: int) -> None:
        if max_step <= 0:
            raise ValueError("max_step must be positive")
        self.min_val = float(config.min_val)
        self.max_val = float(config.max_val)
        self.decay_power = float(config.decay_power)
        self.max_step = int(max_step)

    def __call__(self, step: int) -> float:
        if step < 0 or step > self.max_step:
            raise ValueError(f"step must be between 0 and {self.max_step}")
        ratio = float(step) / float(self.max_step)
        return self.min_val + (self.max_val - self.min_val) * ((1.0 - ratio) ** self.decay_power)


class ExponentialDecay(ParamsDecayBase):
    """Exponential decay from max_val to min_val over [0, max_step]."""

    def __init__(self, config: DecayConfig, max_step: int) -> None:
        if max_step <= 0:
            raise ValueError("max_step must be positive")
        if not (0.0 <= config.decay_rate <= 1.0):
            raise ValueError("decay_rate must be between 0 and 1")
        self.min_val = float(config.min_val)
        self.max_val = float(config.max_val)
        self.decay_rate = float(config.decay_rate)
        self.max_step = int(max_step)

    def __call__(self, step: int) -> float:
        if step < 0 or step > self.max_step:
            raise ValueError(f"step must be between 0 and {self.max_step}")
        ratio = float(step) / float(self.max_step)
        if abs(ratio - 1.0) <= 1e-3:
            return self.min_val
        return self.min_val + (self.max_val - self.min_val) * (
            self.decay_rate ** (ratio / (1.0 - ratio))
        )


def create_params_decay(config: DecayConfig, max_step: int) -> ParamsDecayBase:
    method = config.decay_method.upper()
    if method == "POWER":
        return PowerDecay(config, max_step)
    if method == "EXPONENTIAL":
        return ExponentialDecay(config, max_step)
    raise ValueError(f"Unknown decay_method: {config.decay_method!r}")


def balance_topics_unbalanced(
    weights: np.ndarray,
    normalization_method: str = "topic_sum",
) -> np.ndarray:
    """Returns normalized topic-codepoint weights without capacity constraints."""
    return normalize_lambda_matrix(weights, method=normalization_method)


def balance_topics_greedy(
    weights: np.ndarray,
    normalization_method: str = "topic_sum",
) -> np.ndarray:
    """Greedily assigns each codepoint to a topic with strict capacity balance.

    Each topic (row) can hold at most floor((n_codepoints - 1) / n_topics) + 1
    codepoints. Entries of the normalized weight matrix are traversed in
    descending order of weight.

    Returns:
        A binary uint8 matrix of shape (n_topics, n_codepoints) where each
        column has a single 1.
    """
    normalized = normalize_lambda_matrix(weights, method=normalization_method)
    rows, cols = normalized.shape
    row_capacity = (cols - 1) // rows + 1
    row_size = np.zeros(rows, dtype=np.int64)
    col_assigned = np.zeros(cols, dtype=bool)
    balanced = np.zeros((rows, cols), dtype=np.uint8)

    # Sort flat indices in descending order of weight (stable with row-major tie-breaking)
    flat_order = np.argsort(-normalized.ravel(), kind="mergesort")
    r_indices = flat_order // cols
    c_indices = flat_order % cols

    assigned_count = 0
    for r, c in zip(r_indices, c_indices):
        if col_assigned[c] or row_size[r] >= row_capacity:
            continue
        balanced[r, c] = 1
        row_size[r] += 1
        col_assigned[c] = True
        assigned_count += 1
        if assigned_count == cols:
            break

    return balanced


def balance_topics_diff_constrained(
    weights: np.ndarray,
    max_diff: int,
    normalization_method: str = "topic_sum",
) -> np.ndarray:
    """Greedily assigns codepoints to topics while bounding slice size difference.

    A topic r can accept a new codepoint as long as:
        row_size[r] < max(row_average_capacity, min(row_size) + max_diff)
    """
    if max_diff < 0:
        raise ValueError("max_diff must be non-negative")
    normalized = normalize_lambda_matrix(weights, method=normalization_method)
    rows, cols = normalized.shape
    row_average_capacity = (cols - 1) // rows + 1
    row_size = np.zeros(rows, dtype=np.int64)
    col_assigned = np.zeros(cols, dtype=bool)
    diff_constrained = np.zeros((rows, cols), dtype=np.uint8)

    flat_order = np.argsort(-normalized.ravel(), kind="mergesort")
    r_indices = flat_order // cols
    c_indices = flat_order % cols

    assigned_count = 0
    min_row_size = 0
    for r, c in zip(r_indices, c_indices):
        if col_assigned[c]:
            continue
        limit = max(row_average_capacity, int(min_row_size) + max_diff)
        if row_size[r] >= limit:
            continue
        diff_constrained[r, c] = 1
        row_size[r] += 1
        if row_size[r] - 1 == min_row_size:
            min_row_size = int(np.min(row_size))
        col_assigned[c] = True
        assigned_count += 1
        if assigned_count == cols:
            break

    return diff_constrained
