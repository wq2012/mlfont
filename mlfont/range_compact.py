"""Range-Compact Genetic Algorithm for joint topic and Unicode range optimization."""

from __future__ import annotations

import collections
import random
import time
from typing import List, Optional, Sequence, Tuple

import numpy as np

from mlfont.frequency import get_equally_sized_chunks, get_slices_based_on_frequency
from mlfont.maths import (
    balance_topics_diff_constrained,
    balance_topics_greedy,
    create_params_decay,
    normalize_lambda_matrix,
)
from mlfont.schemas import (
    GeneticHistoryEvent,
    RangeCompactConfig,
    SlicingStrategy,
)

MAX_MUTATION_ATTEMPTS = 1000


class Gene:
    """Stores, mutates, and evaluates a codepoint-to-topic slicing assignment."""

    def __init__(self, dna: Sequence[Tuple[int, int]], n_topics: int) -> None:
        if not dna:
            raise ValueError("DNA cannot be empty")
        self._dna: List[Tuple[int, int]] = sorted(
            ((int(cp), int(topic)) for cp, topic in dna), key=lambda item: item[0]
        )
        self.n_codepoints: int = len(self._dna)
        self.n_topics: int = int(n_topics)
        max_topic = max(t for _, t in self._dna)
        min_topic = min(t for _, t in self._dna)
        if min_topic < 0 or max_topic >= self.n_topics:
            raise ValueError("DNA contains an invalid topic value")

        self._range_cost: Optional[float] = None
        self._topic_cost: Optional[float] = None
        self._hash_code: Optional[int] = None
        self._has_all_topics: Optional[bool] = None

    def get_dna(self) -> List[Tuple[int, int]]:
        return self._dna

    def convert_to_strategy(self, name: str = "range_compact") -> SlicingStrategy:
        slices: List[List[int]] = [[] for _ in range(self.n_topics)]
        for cp, topic in self._dna:
            slices[topic].append(cp)
        return SlicingStrategy(subsets=slices, name=name)

    def compute_costs(self, matrix: np.ndarray) -> Tuple[float, float]:
        if self._range_cost is None:
            self._range_cost = float(self._compute_cost_range_compactness())
        if self._topic_cost is None:
            self._topic_cost = float(self._compute_cost_topic_model_consistency(matrix))
        return (self._range_cost, self._topic_cost)

    def get_costs(self) -> Tuple[float, float]:
        if self._range_cost is None or self._topic_cost is None:
            raise ValueError("Run compute_costs first before retrieving costs")
        return (self._range_cost, self._topic_cost)

    def check_slice_size_diff(self, max_diff: int = 1) -> bool:
        topics = np.fromiter(
            (t for _, t in self._dna), dtype=np.int64, count=self.n_codepoints
        )
        counts = np.bincount(topics, minlength=self.n_topics)
        return int(np.max(counts) - np.min(counts)) <= max_diff

    def check_no_topic_vanished(self) -> bool:
        if self._has_all_topics is None:
            self._has_all_topics = len({t for _, t in self._dna}) == self.n_topics
        return self._has_all_topics

    def get_hash_code(self) -> int:
        if self._hash_code is None:
            self._hash_code = hash(tuple(t for _, t in self._dna))
        return self._hash_code

    def _compute_cost_topic_model_consistency(self, matrix: np.ndarray) -> float:
        topics = np.fromiter(
            (t for _, t in self._dna), dtype=np.int64, count=self.n_codepoints
        )
        cols = np.arange(self.n_codepoints, dtype=np.int64)
        return float(-np.sum(matrix[topics, cols]))

    def _compute_cost_range_compactness(self) -> int:
        if self._range_cost is not None:
            return int(self._range_cost)
        range_cost = 0
        last_cp: Optional[int] = None
        last_topic: Optional[int] = None
        for cp, topic in self._dna:
            if (last_cp is None) or (cp - 1 != last_cp) or (topic != last_topic):
                range_cost += 1
            last_cp = cp
            last_topic = topic
        return range_cost


def init_gene_from_topic_matrix(
    codepoint_list: Sequence[int],
    matrix: np.ndarray,
) -> Gene:
    """Initializes a Gene by assigning each codepoint to argmax_topic matrix[:, col]."""
    if len(codepoint_list) != matrix.shape[1]:
        raise ValueError("Number of codepoints is not equal to matrix column count")
    topics = np.argmax(matrix, axis=0)
    pairs = [(int(cp), int(t)) for cp, t in zip(codepoint_list, topics)]
    return Gene(pairs, matrix.shape[0])


def init_gene_by_equal_binning(
    codepoint_list: Sequence[int],
    number_of_bins: int,
) -> Gene:
    """Initializes a Gene by partitioning sorted codepoints into equal contiguous bins."""
    if number_of_bins <= 0 or number_of_bins > len(codepoint_list):
        raise ValueError("number_of_bins must be between 1 and len(codepoint_list)")
    sorted_cps = sorted(int(c) for c in codepoint_list)
    chunks = get_equally_sized_chunks(sorted_cps, number_of_bins)
    pairs: List[Tuple[int, int]] = []
    for bin_idx, chunk in enumerate(chunks):
        for cp in chunk:
            pairs.append((cp, bin_idx))
    return Gene(pairs, number_of_bins)


def init_gene_from_strategy(strategy: SlicingStrategy) -> Gene:
    """Initializes a Gene from a SlicingStrategy."""
    dna = strategy.to_dna()
    return Gene(dna, strategy.num_slices)


# --- Mutation Operators ---


def purely_random_mutation(
    ancestor: Gene,
    rng: Optional[random.Random] = None,
) -> Gene:
    """Randomly swaps the topics of two codepoints belonging to different topics."""
    if ancestor.n_topics <= 1:
        raise ValueError("Cannot mutate a gene with <= 1 topic")
    r = rng or random
    dna = ancestor.get_dna()[:]
    idx1 = r.randrange(ancestor.n_codepoints)
    idx2 = r.randrange(ancestor.n_codepoints)
    attempts = 0
    while dna[idx1][1] == dna[idx2][1] and attempts < MAX_MUTATION_ATTEMPTS:
        idx2 = r.randrange(ancestor.n_codepoints)
        attempts += 1
    dna[idx1], dna[idx2] = (dna[idx1][0], dna[idx2][1]), (dna[idx2][0], dna[idx1][1])
    return Gene(dna, ancestor.n_topics)


def left_shift_mutation(
    ancestor: Gene,
    rng: Optional[random.Random] = None,
) -> Gene:
    """Shifts a random contiguous topic run leftward by one codepoint."""
    if ancestor.n_topics <= 1:
        raise ValueError("Cannot mutate a gene with <= 1 topic")
    r = rng or random
    dna = ancestor.get_dna()[:]
    attempt_count = 0
    center_index = 1
    left_index = -1
    while attempt_count <= MAX_MUTATION_ATTEMPTS:
        attempt_count += 1
        center_index = r.randrange(1, ancestor.n_codepoints)
        left_index = center_index
        while left_index > -1 and dna[left_index][1] == dna[center_index][1]:
            left_index -= 1
        if left_index != -1:
            break
    if attempt_count > MAX_MUTATION_ATTEMPTS:
        return purely_random_mutation(ancestor, rng=r)

    right_index = center_index
    while (
        right_index < ancestor.n_codepoints
        and dna[right_index][1] == dna[center_index][1]
    ):
        right_index += 1

    i1, i2 = left_index, right_index - 1
    dna[i1], dna[i2] = (dna[i1][0], dna[i2][1]), (dna[i2][0], dna[i1][1])
    return Gene(dna, ancestor.n_topics)


def right_shift_mutation(
    ancestor: Gene,
    rng: Optional[random.Random] = None,
) -> Gene:
    """Shifts a random contiguous topic run rightward by one codepoint."""
    if ancestor.n_topics <= 1:
        raise ValueError("Cannot mutate a gene with <= 1 topic")
    r = rng or random
    dna = ancestor.get_dna()[:]
    attempt_count = 0
    center_index = 0
    right_index = ancestor.n_codepoints
    while attempt_count <= MAX_MUTATION_ATTEMPTS:
        attempt_count += 1
        center_index = r.randrange(ancestor.n_codepoints - 1)
        right_index = center_index
        while (
            right_index < ancestor.n_codepoints
            and dna[right_index][1] == dna[center_index][1]
        ):
            right_index += 1
        if right_index != ancestor.n_codepoints:
            break
    if attempt_count > MAX_MUTATION_ATTEMPTS:
        return purely_random_mutation(ancestor, rng=r)

    left_index = center_index
    while left_index > -1 and dna[left_index][1] == dna[center_index][1]:
        left_index -= 1

    i1, i2 = left_index + 1, right_index
    dna[i1], dna[i2] = (dna[i1][0], dna[i2][1]), (dna[i2][0], dna[i1][1])
    return Gene(dna, ancestor.n_topics)


def swap_topic_mutation(
    ancestor: Gene,
    rng: Optional[random.Random] = None,
) -> Gene:
    """Swaps two entire topic IDs across all codepoints in the gene."""
    if ancestor.n_topics <= 1:
        raise ValueError("Cannot mutate a gene with <= 1 topic")
    r = rng or random
    dna = ancestor.get_dna()[:]
    t1, t2 = r.sample(range(ancestor.n_topics), 2)
    for idx, (cp, topic) in enumerate(dna):
        if topic == t1:
            dna[idx] = (cp, t2)
        elif topic == t2:
            dna[idx] = (cp, t1)
    return Gene(dna, ancestor.n_topics)


def increase_min_topic_mutation(
    ancestor: Gene,
    rng: Optional[random.Random] = None,
) -> Gene:
    """Expands a random range of the smallest topic by 1 into its neighbor."""
    if ancestor.n_topics <= 1:
        raise ValueError("Cannot mutate a gene with <= 1 topic")
    r = rng or random
    dna = ancestor.get_dna()[:]
    counter = collections.Counter(pair[1] for pair in dna)
    min_topic = counter.most_common()[-1][0]
    min_indices = [i for i, pair in enumerate(dna) if pair[1] == min_topic]
    if not min_indices or len(min_indices) == ancestor.n_codepoints:
        return purely_random_mutation(ancestor, rng=r)
    index = r.choice(min_indices)
    if index == 0:
        direction = 1
    elif index == ancestor.n_codepoints - 1:
        direction = -1
    else:
        direction = r.choice([-1, 1])

    steps = 0
    while dna[index][1] == min_topic and steps < ancestor.n_codepoints:
        index = (index + direction) % ancestor.n_codepoints
        steps += 1
    dna[index] = (dna[index][0], min_topic)
    return Gene(dna, ancestor.n_topics)


def decrease_max_topic_mutation(
    ancestor: Gene,
    rng: Optional[random.Random] = None,
) -> Gene:
    """Shrinks a random range of the largest topic by 1, giving the boundary to its neighbor."""
    if ancestor.n_topics <= 1:
        raise ValueError("Cannot mutate a gene with <= 1 topic")
    r = rng or random
    dna = ancestor.get_dna()[:]
    counter = collections.Counter(pair[1] for pair in dna)
    max_topic = counter.most_common()[0][0]
    max_indices = [i for i, pair in enumerate(dna) if pair[1] == max_topic]
    index = r.choice(max_indices)
    if index == 0:
        direction = "+"
    elif index == ancestor.n_codepoints - 1:
        direction = "-"
    else:
        direction = r.choice(["-", "+"])

    if direction == "-":
        while index > 0 and dna[index - 1][1] == max_topic:
            index -= 1
    else:
        while index < ancestor.n_codepoints - 1 and dna[index + 1][1] == max_topic:
            index += 1

    if direction == "-" and index > 0:
        dna[index] = (dna[index][0], dna[index - 1][1])
    elif direction == "+" and index < ancestor.n_codepoints - 1:
        dna[index] = (dna[index][0], dna[index + 1][1])
    else:
        other_topics = [t for t in range(ancestor.n_topics) if t != max_topic]
        dna[index] = (dna[index][0], r.choice(other_topics))
    return Gene(dna, ancestor.n_topics)


def merge_short_range_mutation(
    ancestor: Gene,
    rng: Optional[random.Random] = None,
) -> Gene:
    """Finds one of the shortest contiguous topic runs and merges it into a neighbor."""
    if ancestor.n_topics <= 1:
        raise ValueError("Cannot mutate a gene with <= 1 topic")
    r = rng or random
    dna = ancestor.get_dna()[:]
    n = ancestor.n_codepoints
    runs: List[Tuple[int, int]] = []
    start_idx = 0
    for i in range(n - 1):
        if dna[i + 1][1] != dna[i][1]:
            runs.append((start_idx, i + 1 - start_idx))
            start_idx = i + 1
    runs.append((start_idx, n - start_idx))
    if len(runs) <= 1:
        return purely_random_mutation(ancestor, rng=r)

    min_len = min(length for _, length in runs)
    shortest_runs = [run for run in runs if run[1] == min_len]
    shortest_range_index, shortest_range_len = r.choice(shortest_runs)

    candidates = [
        x
        for x in (shortest_range_index - 1, shortest_range_index + shortest_range_len)
        if 0 <= x < n
    ]
    if not candidates:
        return purely_random_mutation(ancestor, rng=r)
    neighbour_index = r.choice(candidates)
    merge_to_topic = dna[neighbour_index][1]
    for idx in range(shortest_range_index, shortest_range_index + shortest_range_len):
        dna[idx] = (dna[idx][0], merge_to_topic)
    return Gene(dna, ancestor.n_topics)


MUTATION_REGISTRY = {
    "PURELY_RANDOM": purely_random_mutation,
    "LEFT_SHIFT": left_shift_mutation,
    "RIGHT_SHIFT": right_shift_mutation,
    "SWAP_TOPIC": swap_topic_mutation,
    "INCREASE_MIN_TOPIC": increase_min_topic_mutation,
    "DECREASE_MAX_TOPIC": decrease_max_topic_mutation,
    "MERGE_SHORT_RANGE": merge_short_range_mutation,
}


def purely_random_crossover(
    father: Gene,
    mother: Gene,
    rng: Optional[random.Random] = None,
) -> Gene:
    """Single-point crossover combining a prefix of father with a suffix of mother."""
    r = rng or random
    dna_father = father.get_dna()
    dna_mother = mother.get_dna()
    split_idx = r.randrange(1, father.n_codepoints)
    return Gene(dna_father[:split_idx] + dna_mother[split_idx:], father.n_topics)


def remove_duplicate_genes(genes: Sequence[Gene]) -> List[Gene]:
    unique_genes: List[Gene] = []
    seen_hashes: set[int] = set()
    for g in genes:
        h = g.get_hash_code()
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique_genes.append(g)
    return unique_genes


class RangeCompactGeneticLearner:
    """High-performance Genetic Algorithm for range-compact font slicing.

    Uses vectorized numpy topic arrays internally during evolution while
    maintaining exact mathematical equivalence with the Gene representation.
    """

    def __init__(
        self,
        codepoint_list: Sequence[int],
        lambda_matrix: np.ndarray,
        config: RangeCompactConfig,
        frequency_array: Optional[np.ndarray] = None,
        most_sizes: Sequence[int] = (),
        normalization_method: str = "topic_sum",
    ) -> None:
        if len(codepoint_list) == 0:
            raise ValueError("codepoint_list cannot be empty")
        if lambda_matrix.ndim != 2:
            raise ValueError("lambda_matrix must be 2D")
        if not (0.0 <= config.range_cost_weight <= 1.0):
            raise ValueError("range_cost_weight must be between 0 and 1")

        self.raw_codepoints = sorted(int(c) for c in codepoint_list)
        self.config = config
        self.most_sizes = [int(m) for m in most_sizes if int(m) > 0]
        self.most_slices_: List[List[int]] = []

        if self.most_sizes and frequency_array is not None:
            if len(self.raw_codepoints) == lambda_matrix.shape[1]:
                most_idx, self.most_slices_, rem_cps, _ = get_slices_based_on_frequency(
                    self.raw_codepoints,
                    most_sizes=self.most_sizes,
                    frequency_array=frequency_array,
                )
                reduced_matrix = np.delete(lambda_matrix, most_idx, axis=1)
            else:
                _, self.most_slices_, rem_cps, _ = get_slices_based_on_frequency(
                    self.raw_codepoints,
                    most_sizes=self.most_sizes,
                    frequency_array=frequency_array,
                )
                reduced_matrix = lambda_matrix
            self.codepoint_list = rem_cps
            self.matrix = normalize_lambda_matrix(
                reduced_matrix, method=normalization_method
            )
        else:
            self.codepoint_list = list(self.raw_codepoints)
            self.matrix = normalize_lambda_matrix(
                lambda_matrix, method=normalization_method
            )

        if len(self.codepoint_list) != self.matrix.shape[1]:
            raise ValueError(
                f"Codepoint count ({len(self.codepoint_list)}) != "
                f"matrix columns ({self.matrix.shape[1]})"
            )

        self.n_topics, self.n_codepoints = self.matrix.shape
        self.cp_array = np.asarray(self.codepoint_list, dtype=np.int64)
        # Precompute where consecutive codepoints are NOT adjacent (cp[i] - 1 != cp[i-1])
        self.cp_gap_mask = (self.cp_array[1:] - self.cp_array[:-1]) != 1
        self.col_indices = np.arange(self.n_codepoints, dtype=np.int64)
        self.matrix_flat = np.ascontiguousarray(self.matrix.T, dtype=np.float64).ravel()
        self.col_offset = self.col_indices * self.n_topics

        self.history_: List[GeneticHistoryEvent] = []
        self.winner_gene_: Optional[Gene] = None

    def _topics_cost(self, topics: np.ndarray) -> Tuple[float, float, float]:
        """Vectorized computation of (range_cost, topic_cost, final_cost)."""
        topic_breaks = topics[1:] != topics[:-1]
        range_cost = float(1 + np.count_nonzero(self.cp_gap_mask | topic_breaks))
        topic_cost = float(-np.sum(self.matrix_flat[self.col_offset + topics]))
        w_r = self.config.range_cost_weight
        final_cost = w_r * range_cost + (1.0 - w_r) * topic_cost
        return range_cost, topic_cost, final_cost

    def _valid_topics(self, topics: np.ndarray, max_diff: Optional[int] = None) -> bool:
        counts = np.bincount(topics, minlength=self.n_topics)
        if np.any(counts == 0):
            return False
        if max_diff is not None and int(np.max(counts) - np.min(counts)) > max_diff:
            return False
        return True

    def _repair_max_diff(
        self,
        topics: np.ndarray,
        max_diff: int,
        rng: random.Random,
    ) -> np.ndarray:
        """Deterministically repairs topic counts so no topic vanishes and max-min <= max_diff."""
        t = topics.copy()
        n = self.n_codepoints
        min_achievable = 0 if (n % self.n_topics == 0) else 1
        target_diff = max(min_achievable, int(max_diff))
        counts = np.bincount(t, minlength=self.n_topics)

        # 1. Rescue any vanished topics (count == 0)
        zero_topics = np.flatnonzero(counts == 0)
        for k_zero in zero_topics:
            k_max = int(np.argmax(counts))
            max_idx = np.flatnonzero(t == k_max)
            gains = self.matrix[k_zero, max_idx] - self.matrix[k_max, max_idx]
            chosen = int(max_idx[int(np.argmax(gains))])
            t[chosen] = int(k_zero)
            counts[k_max] -= 1
            counts[k_zero] += 1

        # 2. Enforce max(counts) - min(counts) <= target_diff in fast vectorized batches
        steps = 0
        max_steps = 4 * self.n_topics + 64
        while int(np.max(counts) - np.min(counts)) > target_diff and steps < max_steps:
            steps += 1
            k_max = int(np.argmax(counts))
            k_min = int(np.argmin(counts))
            excess = int(counts[k_max] - counts[k_min]) - target_diff
            if excess <= 0:
                break
            batch_sz = max(1, (excess + 1) // 2)

            # First try expanding k_min at its contiguous run borders
            if batch_sz == 1:
                is_min = t == k_min
                adj_min = np.zeros(n, dtype=bool)
                adj_min[1:] |= is_min[:-1]
                adj_min[:-1] |= is_min[1:]
                border_min = np.flatnonzero(adj_min & ~is_min)
                valid_min = border_min[counts[t[border_min]] > counts[k_min] + 1]
                if len(valid_min) > 0:
                    idx_exp = int(valid_min[int(np.argmax(counts[t[valid_min]]))])
                    old_t = int(t[idx_exp])
                    t[idx_exp] = k_min
                    counts[old_t] -= 1
                    counts[k_min] += 1
                    continue

            # Otherwise transfer up to batch_sz codepoints from k_max (preferring run borders)
            is_max = t == k_max
            not_max = ~is_max
            adj_other = np.zeros(n, dtype=bool)
            adj_other[1:] |= not_max[:-1]
            adj_other[:-1] |= not_max[1:]
            border_max = np.flatnonzero(is_max & adj_other)
            candidates = (
                border_max
                if len(border_max) >= batch_sz
                else np.flatnonzero(is_max)
            )
            take = min(batch_sz, max(1, int(counts[k_max]) - 1), len(candidates))
            losses = self.matrix[k_max, candidates] - self.matrix[k_min, candidates]
            if take == 1:
                min_i = int(np.argmin(losses))
                chosen = candidates[min_i:min_i + 1]
            else:
                part_idx = np.argpartition(losses, take - 1)[:take]
                chosen = candidates[part_idx]
            t[chosen] = k_min
            counts[k_max] -= len(chosen)
            counts[k_min] += len(chosen)

        return t

    def _mutate_topics(
        self,
        anc: np.ndarray,
        mut_name: str,
        rng: random.Random,
        max_diff: Optional[int] = None,
    ) -> np.ndarray:
        t = anc.copy()
        n = self.n_codepoints
        if mut_name == "PURELY_RANDOM":
            i1 = rng.randrange(n)
            i2 = rng.randrange(n)
            attempts = 0
            while t[i1] == t[i2] and attempts < MAX_MUTATION_ATTEMPTS:
                i2 = rng.randrange(n)
                attempts += 1
            t[i1], t[i2] = t[i2], t[i1]
            return t

        if mut_name == "LEFT_SHIFT":
            attempts = 0
            center = 1
            left = -1
            while attempts <= MAX_MUTATION_ATTEMPTS:
                attempts += 1
                center = rng.randrange(1, n)
                left = center
                tc = t[center]
                while left > -1 and t[left] == tc:
                    left -= 1
                if left != -1:
                    break
            if attempts > MAX_MUTATION_ATTEMPTS:
                return self._mutate_topics(anc, "PURELY_RANDOM", rng, max_diff=max_diff)
            right = center
            tc = t[center]
            while right < n and t[right] == tc:
                right += 1
            t[left], t[right - 1] = t[right - 1], t[left]
            return t

        if mut_name == "RIGHT_SHIFT":
            attempts = 0
            center = 0
            right = n
            while attempts <= MAX_MUTATION_ATTEMPTS:
                attempts += 1
                center = rng.randrange(n - 1)
                right = center
                tc = t[center]
                while right < n and t[right] == tc:
                    right += 1
                if right != n:
                    break
            if attempts > MAX_MUTATION_ATTEMPTS:
                return self._mutate_topics(anc, "PURELY_RANDOM", rng, max_diff=max_diff)
            left = center
            tc = t[center]
            while left > -1 and t[left] == tc:
                left -= 1
            t[left + 1], t[right] = t[right], t[left + 1]
            return t

        if mut_name == "SWAP_TOPIC":
            k1, k2 = rng.sample(range(self.n_topics), 2)
            m1 = t == k1
            m2 = t == k2
            t[m1] = k2
            t[m2] = k1
            return t

        if mut_name == "INCREASE_MIN_TOPIC":
            return self._expand_min_topic_border(t, rng)

        if mut_name == "DECREASE_MAX_TOPIC":
            return self._shrink_max_topic_border(t, rng)

        if mut_name == "MERGE_SHORT_RANGE":
            breaks = np.flatnonzero(t[1:] != t[:-1]) + 1
            if len(breaks) == 0:
                return self._mutate_topics(anc, "PURELY_RANDOM", rng, max_diff=max_diff)
            starts = np.concatenate(([0], breaks))
            ends = np.concatenate((breaks, [n]))
            lengths = ends - starts
            min_len = int(np.min(lengths))
            short_mask = lengths <= max(min_len, min(4, min_len + 2))
            short_candidates = np.flatnonzero(short_mask)
            if len(short_candidates) > 256:
                step = len(short_candidates) // 256
                start_off = rng.randrange(step)
                short_candidates = short_candidates[start_off::step][:256]
            s_indices = starts[short_candidates]
            s_topics = t[s_indices]
            left_idx = np.maximum(0, s_indices - 1)
            right_idx = np.minimum(n - 1, ends[short_candidates])
            left_topics = t[left_idx]
            right_topics = t[right_idx]
            s_base = s_indices * self.n_topics
            mat_s = self.matrix_flat[s_base + s_topics]
            loss_left = np.where(
                s_indices > 0,
                mat_s - self.matrix_flat[s_base + left_topics],
                1e9,
            )
            loss_right = np.where(
                ends[short_candidates] < n,
                mat_s - self.matrix_flat[s_base + right_topics],
                1e9,
            )
            bridge_bonus = np.where(
                (s_indices > 0)
                & (ends[short_candidates] < n)
                & (left_topics == right_topics),
                -0.01,
                0.0,
            )
            est_loss = (
                np.minimum(loss_left, loss_right) * lengths[short_candidates]
                + bridge_bonus
            )
            pool_size = min(len(short_candidates), max(8, len(short_candidates) // 2))
            top_order = np.argsort(est_loss)[:pool_size]
            top_runs = [int(short_candidates[i]) for i in top_order]
            n_merges = min(
                len(top_runs),
                rng.randint(1, max(2, min(20, len(top_runs) // 3 + 3))),
            )
            chosen_runs = rng.sample(top_runs, n_merges)
            counts = np.bincount(t, minlength=self.n_topics)
            cur_min = int(np.min(counts))
            cur_max = int(np.max(counts))
            for run_idx in chosen_runs:
                s_idx = int(starts[run_idx])
                s_len = int(lengths[run_idx])
                old_topic = int(t[s_idx])
                if counts[old_topic] <= s_len + 1:
                    continue
                if max_diff is not None and counts[old_topic] - s_len < cur_max - max_diff:
                    continue
                candidates = [x for x in (s_idx - 1, s_idx + s_len) if 0 <= x < n]
                if not candidates:
                    continue
                if max_diff is not None:
                    candidates = [
                        x
                        for x in candidates
                        if counts[t[x]] + s_len <= cur_min + max_diff
                    ]
                    if not candidates:
                        continue
                end_idx = s_idx + s_len
                neigh = min(
                    candidates,
                    key=lambda x: (
                        float(
                            np.sum(
                                self.matrix[old_topic, s_idx:end_idx]
                                - self.matrix[t[x], s_idx:end_idx]
                            )
                        ),
                        int(counts[t[x]]),
                    ),
                )
                new_topic = int(t[neigh])
                if new_topic != old_topic:
                    t[s_idx:end_idx] = new_topic
                    counts[old_topic] -= s_len
                    counts[new_topic] += s_len
                    cur_min = int(np.min(counts))
                    cur_max = int(np.max(counts))
            return t

        raise ValueError(f"Unknown mutation: {mut_name!r}")

    def _expand_min_topic_border(self, t: np.ndarray, rng: random.Random) -> np.ndarray:
        """Expands an existing contiguous run of the smallest topic by 1 codepoint."""
        n = self.n_codepoints
        counts = np.bincount(t, minlength=self.n_topics)
        min_topic = int(np.argmin(counts))
        is_min = t == min_topic
        adj_min = np.zeros(n, dtype=bool)
        adj_min[1:] |= is_min[:-1]
        adj_min[:-1] |= is_min[1:]
        border = np.flatnonzero(adj_min & ~is_min)
        if len(border) == 0:
            return t
        idx = int(border[rng.randrange(len(border))])
        if counts[t[idx]] <= 1:
            return t
        t[idx] = min_topic
        return t

    def _shrink_max_topic_border(self, t: np.ndarray, rng: random.Random) -> np.ndarray:
        """Shrinks an existing contiguous run of the largest topic by 1 codepoint."""
        n = self.n_codepoints
        counts = np.bincount(t, minlength=self.n_topics)
        max_topic = int(np.argmax(counts))
        if counts[max_topic] <= 1:
            return t
        is_max = t == max_topic
        not_max = ~is_max
        adj_other = np.zeros(n, dtype=bool)
        adj_other[1:] |= not_max[:-1]
        adj_other[:-1] |= not_max[1:]
        border = np.flatnonzero(is_max & adj_other)
        if len(border) == 0:
            return t
        idx = int(border[rng.randrange(len(border))])
        neighs = [x for x in (idx - 1, idx + 1) if 0 <= x < n and t[x] != max_topic]
        if neighs:
            chosen = min(neighs, key=lambda x: int(counts[t[x]]))
            t[idx] = t[chosen]
        return t

    def run(self) -> SlicingStrategy:
        """Runs the Range-Compact Genetic Algorithm and returns the winning SlicingStrategy."""
        rng = random.Random(self.config.random_seed)
        np_rng = np.random.default_rng(self.config.random_seed)
        start_time = time.time()

        decay_fn = create_params_decay(
            self.config.decay_max_diff, max(1, self.config.max_iteration)
        )
        initial_max_diff = int(decay_fn(0))
        final_min_diff = max(1, int(self.config.decay_max_diff.min_val))

        # Initialize survival population
        initial_genes: List[Gene] = []
        if self.config.init_from_matrix:
            diff_mat = balance_topics_diff_constrained(
                self.matrix,
                max_diff=final_min_diff,
                normalization_method="none",
            )
            initial_genes.append(
                init_gene_from_topic_matrix(self.codepoint_list, diff_mat)
            )
        if self.config.init_from_topic_balanced_matrix:
            greedy_mat = balance_topics_greedy(self.matrix, normalization_method="none")
            initial_genes.append(
                init_gene_from_topic_matrix(self.codepoint_list, greedy_mat)
            )
        if self.config.init_from_binning or not initial_genes:
            initial_genes.append(
                init_gene_by_equal_binning(self.codepoint_list, self.n_topics)
            )

        # Represent individuals as (topics_array, range_cost, topic_cost, final_cost, hash_id)
        population: List[Tuple[np.ndarray, float, float, float, int]] = []
        for g in initial_genes:
            t_arr = np.array([topic for _, topic in g.get_dna()], dtype=np.int64)
            if not self._valid_topics(t_arr, max_diff=initial_max_diff):
                t_arr = self._repair_max_diff(t_arr, initial_max_diff, rng)
            rc, tc, fc = self._topics_cost(t_arr)
            population.append((t_arr, rc, tc, fc, hash(t_arr.tobytes())))

        mut_names = list(self.config.mutation_probabilities.keys())
        mut_probs = np.array(
            [self.config.mutation_probabilities[k] for k in mut_names], dtype=np.float64
        )
        mut_probs /= np.sum(mut_probs)
        count_changing_muts = {
            "INCREASE_MIN_TOPIC",
            "DECREASE_MAX_TOPIC",
            "MERGE_SHORT_RANGE",
        }

        old_max_diff = -1
        self.history_ = []

        for step in range(self.config.max_iteration + 1):
            max_diff = int(decay_fn(step))
            max_diff_changed = max_diff != old_max_diff
            old_max_diff = max_diff

            alive = list(population)
            n_pop = len(population)

            # Mutations
            chosen_muts = np_rng.choice(
                len(mut_names), size=self.config.n_mutations_per_iteration, p=mut_probs
            )
            for m_idx in chosen_muts:
                m_name = mut_names[int(m_idx)]
                parent_topics = population[rng.randrange(n_pop)][0]
                child_topics = self._mutate_topics(
                    parent_topics, m_name, rng, max_diff=max_diff
                )
                if m_name in count_changing_muts and not self._valid_topics(
                    child_topics, max_diff=max_diff
                ):
                    child_topics = self._repair_max_diff(child_topics, max_diff, rng)
                rc, tc, fc = self._topics_cost(child_topics)
                alive.append((child_topics, rc, tc, fc, hash(child_topics.tobytes())))

            # Crossovers
            if n_pop >= 2:
                for _ in range(self.config.n_crossovers_per_iteration):
                    i_f, i_m = rng.sample(range(n_pop), 2)
                    split = rng.randrange(1, self.n_codepoints)
                    child_topics = np.concatenate(
                        (population[i_f][0][:split], population[i_m][0][split:])
                    )
                    if self._valid_topics(child_topics, max_diff=max_diff):
                        rc, tc, fc = self._topics_cost(child_topics)
                        alive.append(
                            (child_topics, rc, tc, fc, hash(child_topics.tobytes()))
                        )

            # Deduplicate and strictly enforce max_diff and no vanished topics
            deduped: List[Tuple[np.ndarray, float, float, float, int]] = []
            seen: set[int] = set()
            for item in alive:
                t_arr = item[0]
                if max_diff_changed and not self._valid_topics(t_arr, max_diff=max_diff):
                    t_arr = self._repair_max_diff(t_arr, max_diff, rng)
                    rc, tc, fc = self._topics_cost(t_arr)
                    item = (t_arr, rc, tc, fc, hash(t_arr.tobytes()))
                if item[4] not in seen:
                    seen.add(item[4])
                    deduped.append(item)

            deduped.sort(key=lambda item: item[3])
            population = deduped[: self.config.n_survivals]

            hist_int = max(1, self.config.history_interval)
            if step % hist_int == 0 or step == self.config.max_iteration:
                n_cp = float(self.n_codepoints)
                best = population[0]
                avg_rc = float(np.mean([x[1] for x in population])) / n_cp
                avg_tc = float(np.mean([x[2] for x in population])) / n_cp
                w_r = self.config.range_cost_weight
                avg_fc = w_r * avg_rc + (1.0 - w_r) * avg_tc
                self.history_.append(
                    GeneticHistoryEvent(
                        iteration=step,
                        elapsed_sec=time.time() - start_time,
                        best_range_cost=best[1] / n_cp,
                        best_topic_cost=best[2] / n_cp,
                        best_final_cost=best[3] / n_cp,
                        avg_range_cost=avg_rc,
                        avg_topic_cost=avg_tc,
                        avg_final_cost=avg_fc,
                        min_range_cost=float(min(x[1] for x in population)) / n_cp,
                        min_topic_cost=float(min(x[2] for x in population)) / n_cp,
                        max_diff=max_diff,
                        num_survivals=len(population),
                        best_raw_ranges=int(best[1]),
                    )
                )

        winner_topics = population[0][0]
        winner_dna = list(zip(self.codepoint_list, (int(t) for t in winner_topics)))
        self.winner_gene_ = Gene(winner_dna, self.n_topics)
        self.winner_gene_.compute_costs(self.matrix)

        ga_strategy = self.winner_gene_.convert_to_strategy(
            name=f"range_compact_topics{self.n_topics}"
        )
        all_subsets = self.most_slices_ + ga_strategy.subsets
        most_tag = "_".join(str(m) for m in self.most_sizes)
        name = (
            f"range_compact_most{most_tag}_topics{self.n_topics}"
            if most_tag
            else f"range_compact_topics{self.n_topics}"
        )
        return SlicingStrategy(
            subsets=all_subsets,
            name=name,
            metadata={
                "algorithm": "range_compact_ga",
                "n_topics": self.n_topics,
                "most_sizes": self.most_sizes,
                "range_cost_weight": self.config.range_cost_weight,
                "max_iteration": self.config.max_iteration,
                "final_range_cost": float(population[0][1]),
                "final_topic_cost": float(population[0][2]),
            },
        )

    def fit(self) -> SlicingStrategy:
        """Alias for run()."""
        return self.run()


class RangeCompactGASolver(RangeCompactGeneticLearner):
    """Solver interface for the Range-Compact Genetic Algorithm."""

    def __init__(
        self,
        codepoint_list: Optional[Sequence[int]] = None,
        lambda_matrix: Optional[np.ndarray] = None,
        config: Optional[RangeCompactConfig] = None,
        frequency_array: Optional[np.ndarray] = None,
        most_sizes: Sequence[int] = (),
        normalization_method: str = "topic_sum",
        codepoints: Optional[Sequence[int]] = None,
        topic_matrix: Optional[np.ndarray] = None,
    ) -> None:
        cps = codepoint_list if codepoint_list is not None else codepoints
        mat = lambda_matrix if lambda_matrix is not None else topic_matrix
        if cps is None or mat is None:
            raise ValueError("Both codepoints and topic_matrix/lambda_matrix must be provided.")
        mat_arr = np.asarray(mat, dtype=np.float64)
        if mat_arr.ndim == 2 and mat_arr.shape[1] != len(cps) and mat_arr.shape[0] == len(cps):
            mat_arr = mat_arr.T
        if config is None:
            config = RangeCompactConfig()
        super().__init__(
            codepoint_list=cps,
            lambda_matrix=mat_arr,
            config=config,
            frequency_array=frequency_array,
            most_sizes=most_sizes,
            normalization_method=normalization_method,
        )

    def optimize(
        self,
        strategy_name: Optional[str] = None,
        font_name: Optional[str] = None,
        prefix_slices: Optional[Sequence[Sequence[int]]] = None,
        suffix_slices: Optional[Sequence[Sequence[int]]] = None,
    ) -> SlicingStrategy:
        """Runs GA optimization and optionally prepends/appends deterministic slices."""
        strat = self.run()
        subsets = list(strat.subsets)
        if prefix_slices:
            subsets = [sorted(int(c) for c in s) for s in prefix_slices] + subsets
        if suffix_slices:
            subsets = subsets + [sorted(int(c) for c in s) for s in suffix_slices]
        meta = dict(strat.metadata)
        if font_name is not None:
            meta["font_name"] = font_name
        return SlicingStrategy(
            subsets=subsets,
            name=strategy_name or strat.name,
            metadata=meta,
        )
