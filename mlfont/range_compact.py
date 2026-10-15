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


def potts_viterbi_smooth_topics(
    codepoint_list: Sequence[int],
    matrix: np.ndarray,
    range_cost_weight: float = 0.012,
    n_dual_iters: int = 8,
) -> np.ndarray:
    """Solves the Lagrangian 1D Potts-Viterbi relaxation for range-compact slicing.

    Minimizes the exact range-compact objective within contiguous Unicode segments:
        J(t; mu) = sum_i [ -(1 - w_r) * matrix[t_i, i] + mu[t_i] ]
                   + w_r * sum_{i: cp[i+1] == cp[i] + 1} I[t_{i+1} != t_i]
    in O(N * K) time per dual subgradient step via min-sum message passing,
    adjusting dual multipliers mu[k] to encourage balanced slice cardinalities.
    """
    n_topics, n_cp = matrix.shape
    if n_cp == 0:
        return np.empty(0, dtype=np.int64)
    if n_topics <= 1 or range_cost_weight <= 0.0:
        return np.argmax(matrix, axis=0).astype(np.int64)

    cp_arr = np.asarray(codepoint_list, dtype=np.int64)
    cp_gap_mask = (cp_arr[1:] - cp_arr[:-1]) != 1
    w_r = float(range_cost_weight)
    w_t = 1.0 - w_r
    base_unary = -w_t * np.asarray(matrix, dtype=np.float64)  # (K, N)

    target_count = float(n_cp) / float(n_topics)
    mu = np.zeros(n_topics, dtype=np.float64)
    val_range = max(1e-6, float(np.ptp(base_unary)))
    step_scale = 0.35 * (w_r + 0.25 * val_range)

    best_topics: Optional[np.ndarray] = None
    best_imbalance = float("inf")
    k_arange = np.arange(n_topics, dtype=np.int32)

    # Identify contiguous Unicode segments where cp_gap_mask is False
    gap_splits = np.flatnonzero(cp_gap_mask) + 1
    seg_starts = np.concatenate(([0], gap_splits))
    seg_ends = np.concatenate((gap_splits, [n_cp]))

    for dual_step in range(max(1, n_dual_iters)):
        topics = np.empty(n_cp, dtype=np.int64)
        running_counts = np.zeros(n_topics, dtype=np.float64)
        processed = 0

        for s_idx, e_idx in zip(seg_starts, seg_ends):
            seg_len = int(e_idx - s_idx)
            # Online Gauss-Seidel capacity pressure + global dual multiplier
            expected_so_far = max(1.0, float(processed) / float(n_topics))
            online_pressure = (
                0.15 * step_scale * ((running_counts - expected_so_far) / target_count)
            )
            eff_mu = mu + online_pressure

            if seg_len == 1:
                t_best = int(np.argmin(base_unary[:, s_idx] + eff_mu))
                topics[s_idx] = t_best
                running_counts[t_best] += 1.0
                processed += 1
                continue

            backptr = np.empty((seg_len, n_topics), dtype=np.int32)
            v_curr = base_unary[:, s_idx] + eff_mu
            backptr[0] = k_arange
            for offset in range(1, seg_len):
                col = s_idx + offset
                m_prev = float(np.min(v_curr))
                best_prev = int(np.argmin(v_curr))
                jump_cost = m_prev + w_r
                stay_mask = v_curr <= jump_cost
                backptr[offset] = np.where(stay_mask, k_arange, best_prev)
                v_curr = (
                    base_unary[:, col]
                    + eff_mu
                    + np.where(stay_mask, v_curr, jump_cost)
                )

            curr_t = int(np.argmin(v_curr))
            topics[e_idx - 1] = curr_t
            for offset in range(seg_len - 1, 0, -1):
                curr_t = int(backptr[offset, curr_t])
                topics[s_idx + offset - 1] = curr_t

            seg_counts = np.bincount(topics[s_idx:e_idx], minlength=n_topics)
            running_counts += seg_counts
            processed += seg_len

        counts = np.bincount(topics, minlength=n_topics)
        imbalance = float(np.max(counts) - np.min(counts)) + float(
            np.count_nonzero(counts == 0) * n_cp
        )
        if best_topics is None or imbalance < best_imbalance:
            best_imbalance = imbalance
            best_topics = topics.copy()
        if imbalance <= 2.0:
            break

        lr = step_scale / np.sqrt(1.0 + 0.5 * dual_step)
        mu += lr * ((counts.astype(np.float64) - target_count) / target_count)
        mu -= np.mean(mu)

    assert best_topics is not None
    best_counts = np.bincount(best_topics, minlength=n_topics)
    if int(np.count_nonzero(best_counts == 0)) > n_topics // 2 and n_cp >= n_topics:
        sub_len = max(1, int(round(target_count)))
        sub_starts: List[int] = []
        sub_ends: List[int] = []
        for s_idx, e_idx in zip(seg_starts, seg_ends):
            for pos in range(int(s_idx), int(e_idx), sub_len):
                sub_starts.append(pos)
                sub_ends.append(min(int(e_idx), pos + sub_len))
        sym_mu = 1e-6 * np.arange(n_topics, dtype=np.float64)
        topics_fb = np.empty(n_cp, dtype=np.int64)
        running_fb = np.zeros(n_topics, dtype=np.float64)
        for s_idx, e_idx in zip(sub_starts, sub_ends):
            seg_len = int(e_idx - s_idx)
            over_mask = running_fb >= target_count
            eff_mu = (
                sym_mu
                + np.where(over_mask, 10.0 * (w_r + val_range), 0.0)
                + 0.25 * step_scale * (running_fb / max(1.0, target_count))
            )
            if seg_len == 1:
                t_best = int(np.argmin(base_unary[:, s_idx] + eff_mu))
                topics_fb[s_idx] = t_best
                running_fb[t_best] += 1.0
                continue
            backptr = np.empty((seg_len, n_topics), dtype=np.int32)
            v_curr = base_unary[:, s_idx] + eff_mu
            backptr[0] = k_arange
            for offset in range(1, seg_len):
                col = s_idx + offset
                m_prev = float(np.min(v_curr))
                best_prev = int(np.argmin(v_curr))
                jump_cost = m_prev + w_r
                stay_mask = v_curr <= jump_cost
                backptr[offset] = np.where(stay_mask, k_arange, best_prev)
                v_curr = (
                    base_unary[:, col]
                    + eff_mu
                    + np.where(stay_mask, v_curr, jump_cost)
                )
            curr_t = int(np.argmin(v_curr))
            topics_fb[e_idx - 1] = curr_t
            for offset in range(seg_len - 1, 0, -1):
                curr_t = int(backptr[offset, curr_t])
                topics_fb[s_idx + offset - 1] = curr_t
            running_fb += np.bincount(topics_fb[s_idx:e_idx], minlength=n_topics)
        fb_counts = np.bincount(topics_fb, minlength=n_topics)
        fb_imb = float(np.max(fb_counts) - np.min(fb_counts)) + float(
            np.count_nonzero(fb_counts == 0) * n_cp
        )
        if fb_imb < best_imbalance:
            best_topics = topics_fb
    return best_topics


def build_island_bridged_topics(
    codepoint_list: Sequence[int],
    matrix: np.ndarray,
    rem_frequency_array: Optional[np.ndarray] = None,
    max_bridge: int = 4,
) -> Optional[np.ndarray]:
    """Constructs a range-compact monochromatic-island topic assignment for sparse CJK matrices.

    When a font repertoire has a large sparse tail (where most codepoints are
    unobserved in the corpus), greedy column-by-column balancing interleaves
    foreground active codepoints with background bin topics, creating two range
    breaks per active singleton. This solver:
      1. Identifies active foreground columns and foreground topics from
         ``rem_frequency_array`` (or the column-spread distribution of ``matrix``).
      2. Trades isolated lowest-frequency length-1 singletons for short contiguous
         bridges (length <= max_bridge) between nearby active codepoints within
         the exact foreground slice capacity budget.
      3. Assigns each contiguous foreground island monochromatically to its
         highest-affinity foreground topic and packs background codepoints into
         contiguous archive runs.
    """
    n_topics, n_rem = matrix.shape
    if n_rem < n_topics * 2 or n_topics <= 2:
        return None
    cp_arr = np.asarray(codepoint_list, dtype=np.int64)
    gap_mask = (cp_arr[1:] - cp_arr[:-1]) != 1
    row_cap = (n_rem - 1) // n_topics + 1

    if rem_frequency_array is not None and len(rem_frequency_array) == n_rem:
        rem_freq = np.asarray(rem_frequency_array, dtype=np.float64)
        core_mask = rem_freq >= 2.0
        if int(np.sum(core_mask)) < n_topics:
            return None
        greedy_mat = balance_topics_greedy(matrix, normalization_method="none")
        t_greedy = np.argmax(greedy_mat, axis=0)
        topic_active_df = np.array(
            [np.sum(rem_freq[t_greedy == k] >= 2.0) for k in range(n_topics)]
        )
        k_fg = max(4, min(n_topics - 1, int(np.sum(topic_active_df >= 10)) + 2))
        ranked_fg = np.argsort(topic_active_df)[::-1]
        fg_topics = np.sort(ranked_fg[:k_fg])
        fg_budget = k_fg * row_cap

        core_idx = np.flatnonzero(core_mask)
        keep = np.ones(len(core_idx), dtype=bool)
        for j in range(len(core_idx)):
            c_i = int(core_idx[j])
            if rem_freq[c_i] <= 3.0:
                l_far = (
                    (j == 0)
                    or (c_i - int(core_idx[j - 1]) - 1 > max_bridge)
                    or bool(np.any(gap_mask[int(core_idx[j - 1]) : c_i]))
                )
                r_far = (
                    (j == len(core_idx) - 1)
                    or (int(core_idx[j + 1]) - c_i - 1 > max_bridge)
                    or bool(np.any(gap_mask[c_i : int(core_idx[j + 1])]))
                )
                if l_far and r_far:
                    keep[j] = False
        kept = core_idx[keep]

        fg_mask = np.zeros(n_rem, dtype=bool)
        fg_mask[kept] = True
        bridges: List[Tuple[float, int, int, int]] = []
        for j in range(len(kept) - 1):
            i1, i2 = int(kept[j]), int(kept[j + 1])
            L = i2 - i1 - 1
            if 0 < L <= max_bridge and not np.any(gap_mask[i1:i2]):
                end_df = float(rem_freq[i1] + rem_freq[i2])
                bridges.append((L - 0.15 * float(np.log1p(end_df)), L, i1 + 1, i2))
        bridges.sort()
        cur_fg = int(np.sum(fg_mask))
        for _, L, sb, eb in bridges:
            if cur_fg + L <= fg_budget:
                fg_mask[sb:eb] = True
                cur_fg += L

        for _ in range(2):
            for i in range(n_rem):
                if fg_mask[i] and rem_freq[i] <= 3.0:
                    l_fg = (i > 0) and not gap_mask[i - 1] and fg_mask[i - 1]
                    r_fg = (i < n_rem - 1) and not gap_mask[i] and fg_mask[i + 1]
                    if not l_fg and not r_fg:
                        fg_mask[i] = False
                        cur_fg -= 1
            surv = np.flatnonzero(fg_mask)
            rem_b: List[Tuple[int, int, int]] = []
            for j in range(len(surv) - 1):
                i1, i2 = int(surv[j]), int(surv[j + 1])
                L = i2 - i1 - 1
                if 0 < L <= max_bridge and not np.any(gap_mask[i1:i2]):
                    rem_b.append((L, i1 + 1, i2))
            rem_b.sort()
            for L, sb, eb in rem_b:
                if cur_fg + L <= fg_budget:
                    fg_mask[sb:eb] = True
                    cur_fg += L

        col_weights = 1.0 + np.log1p(rem_freq)
    else:
        col_spread = np.ptp(matrix, axis=0)
        if float(np.max(col_spread)) <= 1e-12:
            return None
        bg_spread = float(np.quantile(col_spread, 0.50))
        core_mask = col_spread > max(1e-9, bg_spread * 2.2)
        if int(np.sum(core_mask)) < n_topics:
            return None
        core_idx = np.flatnonzero(core_mask)
        core_best_t = np.argmax(matrix[:, core_idx], axis=0)
        t_counts = np.bincount(core_best_t, minlength=n_topics)
        k_fg = max(4, min(n_topics - 1, int(np.sum(t_counts >= 5)) + 2))
        fg_topics = np.sort(np.argsort(t_counts)[::-1][:k_fg])
        fg_budget = k_fg * row_cap

        med_core_spread = float(np.quantile(col_spread[core_idx], 0.30))
        keep = np.ones(len(core_idx), dtype=bool)
        for j in range(len(core_idx)):
            c_i = int(core_idx[j])
            if col_spread[c_i] <= med_core_spread:
                l_far = (
                    (j == 0)
                    or (c_i - int(core_idx[j - 1]) - 1 > max_bridge)
                    or bool(np.any(gap_mask[int(core_idx[j - 1]) : c_i]))
                )
                r_far = (
                    (j == len(core_idx) - 1)
                    or (int(core_idx[j + 1]) - c_i - 1 > max_bridge)
                    or bool(np.any(gap_mask[c_i : int(core_idx[j + 1])]))
                )
                if l_far and r_far:
                    keep[j] = False
        kept = core_idx[keep]

        fg_mask = np.zeros(n_rem, dtype=bool)
        fg_mask[kept] = True
        bridges_s: List[Tuple[float, int, int, int]] = []
        for j in range(len(kept) - 1):
            i1, i2 = int(kept[j]), int(kept[j + 1])
            L = i2 - i1 - 1
            if 0 < L <= max_bridge and not np.any(gap_mask[i1:i2]):
                end_s = float(col_spread[i1] + col_spread[i2])
                bridges_s.append((L - 0.5 * end_s, L, i1 + 1, i2))
        bridges_s.sort()
        cur_fg = int(np.sum(fg_mask))
        for _, L, sb, eb in bridges_s:
            if cur_fg + L <= fg_budget:
                fg_mask[sb:eb] = True
                cur_fg += L
        col_weights = 0.1 + col_spread

    islands: List[Tuple[int, int]] = []
    in_isl = False
    s_cur = 0
    for i in range(n_rem):
        if fg_mask[i]:
            if not in_isl:
                s_cur = i
                in_isl = True
            elif (i > 0 and gap_mask[i - 1]) or (
                n_rem < 5000 and i - s_cur >= row_cap
            ):
                islands.append((s_cur, i))
                s_cur = i
        else:
            if in_isl:
                islands.append((s_cur, i))
                in_isl = False
    if in_isl:
        islands.append((s_cur, n_rem))

    t_full = np.full(n_rem, -1, dtype=np.int64)
    fg_counts = np.zeros(k_fg, dtype=np.int64)
    isl_scores: List[Tuple[float, int, np.ndarray]] = []
    for idx_isl, (si, ei) in enumerate(islands):
        w_cols = col_weights[si:ei][np.newaxis, :]
        aff = np.sum(matrix[fg_topics, si:ei] * w_cols, axis=1)
        isl_scores.append((float(np.max(aff) - np.min(aff)), idx_isl, aff))
    isl_scores.sort(reverse=True)
    for _, idx_isl, aff in isl_scores:
        si, ei = islands[idx_isl]
        L = ei - si
        order_k = np.argsort(aff)[::-1]
        chosen_k = int(order_k[0])
        for cand_k in order_k:
            if fg_counts[cand_k] + L <= row_cap:
                chosen_k = int(cand_k)
                break
        else:
            chosen_k = int(np.argmin(fg_counts))
        t_full[si:ei] = fg_topics[chosen_k]
        fg_counts[chosen_k] += L

    counts = np.bincount(t_full[t_full >= 0], minlength=n_topics)
    unassigned = np.flatnonzero(t_full < 0)
    rem_slots = np.maximum(0, row_cap - counts)
    diff_s = len(unassigned) - int(np.sum(rem_slots))
    if diff_s < 0:
        if n_rem < 5000:
            while diff_s < 0 and int(np.max(rem_slots)) > 0:
                for k in np.argsort(rem_slots)[::-1]:
                    if rem_slots[k] > 0 and diff_s < 0:
                        rem_slots[k] -= 1
                        diff_s += 1
        else:
            for k in np.argsort(rem_slots)[::-1]:
                take = min(-diff_s, int(rem_slots[k]))
                rem_slots[k] -= take
                diff_s += take
                if diff_s == 0:
                    break
    pos = 0
    for k in range(n_topics):
        cnt = int(rem_slots[k])
        if cnt > 0:
            t_full[unassigned[pos : pos + cnt]] = k
            pos += cnt
    if pos < len(unassigned):
        t_full[unassigned[pos:]] = n_topics - 1
    return t_full


class RangeCompactGeneticLearner:
    """High-performance Genetic Algorithm for range-compact font slicing.

    Uses vectorized numpy topic arrays internally during evolution while
    maintaining exact mathematical equivalence with the Gene representation.
    """

    def __init__(
        self,
        codepoint_list: Sequence[int],
        lambda_matrix: np.ndarray,
        config: Optional[RangeCompactConfig] = None,
        frequency_array: Optional[np.ndarray] = None,
        most_sizes: Sequence[int] = (),
        normalization_method: str = "topic_sum",
    ) -> None:
        if config is None:
            config = RangeCompactConfig()
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
        self.rem_frequency_array_: Optional[np.ndarray] = None

        if self.most_sizes and frequency_array is not None:
            if len(self.raw_codepoints) == lambda_matrix.shape[1]:
                most_idx, self.most_slices_, rem_cps, _ = get_slices_based_on_frequency(
                    self.raw_codepoints,
                    most_sizes=self.most_sizes,
                    frequency_array=frequency_array,
                )
                reduced_matrix = np.delete(lambda_matrix, most_idx, axis=1)
                self.rem_frequency_array_ = np.delete(
                    np.asarray(frequency_array, dtype=np.float64), most_idx
                )
            else:
                most_idx, self.most_slices_, rem_cps, _ = get_slices_based_on_frequency(
                    self.raw_codepoints,
                    most_sizes=self.most_sizes,
                    frequency_array=frequency_array,
                )
                reduced_matrix = lambda_matrix
                if len(frequency_array) == len(rem_cps):
                    self.rem_frequency_array_ = np.asarray(
                        frequency_array, dtype=np.float64
                    )
                elif len(frequency_array) == len(self.raw_codepoints):
                    self.rem_frequency_array_ = np.delete(
                        np.asarray(frequency_array, dtype=np.float64), most_idx
                    )
            self.codepoint_list = rem_cps
            self.matrix = normalize_lambda_matrix(
                reduced_matrix, method=normalization_method
            )
        else:
            self.codepoint_list = list(self.raw_codepoints)
            if frequency_array is not None and len(frequency_array) == len(
                self.codepoint_list
            ):
                self.rem_frequency_array_ = np.asarray(
                    frequency_array, dtype=np.float64
                )
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
        self.fg_topics_: set[int] = set()
        if (
            self.n_codepoints > 5000
            and self.rem_frequency_array_ is not None
            and len(self.rem_frequency_array_) == self.n_codepoints
        ):
            greedy_mat = balance_topics_greedy(
                self.matrix, normalization_method="none"
            )
            t_greedy = np.argmax(greedy_mat, axis=0)
            topic_active_df = np.array(
                [
                    np.sum(self.rem_frequency_array_[t_greedy == k] >= 2.0)
                    for k in range(self.n_topics)
                ]
            )
            k_fg = max(
                4, min(self.n_topics - 1, int(np.sum(topic_active_df >= 10)) + 2)
            )
            ranked_fg = np.argsort(topic_active_df)[::-1]
            self.fg_topics_ = set(int(x) for x in ranked_fg[:k_fg])

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
        """Range-preserving capacity repair ensuring count > 0 and max-min <= max_diff.

        Prioritizes whole-run relabeling (which never increases range breaks,
        Delta C_range <= 0) and contiguous border sliding / run-prefix transfers
        over scattered point flips.
        """
        t = topics.copy()
        n = self.n_codepoints
        min_achievable = 0 if (n % self.n_topics == 0) else 1
        target_diff = max(min_achievable, int(max_diff))
        counts = np.bincount(t, minlength=self.n_topics)
        w_r = self.config.range_cost_weight
        w_t = 1.0 - w_r
        avg_floor = max(1, n // self.n_topics)
        avg_ceil = max(1, (n + self.n_topics - 1) // self.n_topics)

        # 1. Rescue any vanished topics (count == 0) using shortest/lowest-loss runs of k_max
        zero_topics = np.flatnonzero(counts == 0)
        many_zeros = len(zero_topics) > self.n_topics // 2
        for k_zero in zero_topics:
            k_max = int(np.argmax(counts))
            is_max = t == k_max
            padded = np.empty(n + 2, dtype=bool)
            padded[0] = False
            padded[1:-1] = is_max
            padded[-1] = False
            r_starts = np.flatnonzero(padded[1:] & ~padded[:-1])
            r_ends = np.flatnonzero(~padded[1:] & padded[:-1])
            r_lens = r_ends - r_starts
            if many_zeros:
                target_take = max(1, min(int(counts[k_max]) - avg_floor, avg_floor))
            else:
                target_take = max(1, min(int(counts[k_max]) - 1, n // (2 * self.n_topics)))
            short_mask = r_lens <= target_take
            if np.any(short_mask):
                cand_idx = int(np.flatnonzero(short_mask)[0])
                s_i, e_i = int(r_starts[cand_idx]), int(r_ends[cand_idx])
                t[s_i:e_i] = int(k_zero)
                counts[k_max] -= e_i - s_i
                counts[k_zero] += e_i - s_i
            else:
                s_i = int(r_starts[0])
                take = max(1, min(target_take, int(r_lens[0]) - 1)) if many_zeros else 1
                t[s_i : s_i + take] = int(k_zero)
                counts[k_max] -= take
                counts[k_zero] += take

        # 2. Enforce max(counts) - min(counts) <= target_diff via whole-run & border transfers
        steps = 0
        max_steps = 6 * self.n_topics + 128
        while int(np.max(counts) - np.min(counts)) > target_diff and steps < max_steps:
            steps += 1
            k_max = int(np.argmax(counts))
            k_min = int(np.argmin(counts))
            excess = int(counts[k_max] - counts[k_min]) - target_diff
            if excess <= 0:
                break
            batch_sz = max(1, (excess + 1) // 2)
            if int(counts[k_max]) > 2 * avg_ceil:
                batch_sz = min(
                    batch_sz,
                    max(1, int(counts[k_max]) - avg_floor),
                    max(1, avg_ceil - int(counts[k_min])),
                )
            batch_sz = min(batch_sz, int(counts[k_max]) - 1)

            # Decompose k_max into maximal contiguous runs [r_starts, r_ends)
            is_max = t == k_max
            padded = np.empty(n + 2, dtype=bool)
            padded[0] = False
            padded[1:-1] = is_max
            padded[-1] = False
            r_starts = np.flatnonzero(padded[1:] & ~padded[:-1])
            r_ends = np.flatnonzero(~padded[1:] & padded[:-1])
            r_lens = r_ends - r_starts
            if len(r_starts) == 0:
                break

            diff_vec = self.matrix[k_max] - self.matrix[k_min]
            if (
                self.fg_topics_
                and k_min not in self.fg_topics_
                and self.rem_frequency_array_ is not None
            ):
                diff_vec = diff_vec + 100.0 * (self.rem_frequency_array_ >= 2.0)
            csum = np.empty(n + 1, dtype=np.float64)
            csum[0] = 0.0
            np.cumsum(diff_vec, out=csum[1:])

            # Option A: Whole runs of k_max with length <= batch_sz (Delta C_range <= 0!)
            whole_mask = r_lens <= batch_sz
            if np.any(whole_mask):
                w_idx = np.flatnonzero(whole_mask)
                ws = r_starts[w_idx]
                we = r_ends[w_idx]
                wl = r_lens[w_idx]
                topic_loss = csum[we] - csum[ws]
                left_match = (ws > 0) & (~self.cp_gap_mask[np.maximum(0, ws - 1)]) & (
                    t[np.maximum(0, ws - 1)] == k_min
                )
                right_match = (
                    (we < n)
                    & (~self.cp_gap_mask[np.minimum(n - 2, we - 1)])
                    & (t[np.minimum(n - 1, we)] == k_min)
                )
                delta_range = -(left_match.astype(np.float64) + right_match.astype(np.float64))
                # Score per codepoint transferred so we pack batch_sz with lowest cost
                score_per_cp = (w_t * topic_loss + w_r * delta_range) / wl
                order = np.argsort(score_per_cp)
                rem = batch_sz
                transferred = 0
                for ord_i in order:
                    L = int(wl[ord_i])
                    if L <= rem and int(counts[k_max]) - L >= 1:
                        s_i = int(ws[ord_i])
                        e_i = int(we[ord_i])
                        t[s_i:e_i] = k_min
                        counts[k_max] -= L
                        counts[k_min] += L
                        rem -= L
                        transferred += L
                        if rem <= 0:
                            break
                if transferred > 0:
                    continue

            # Option B: Contiguous prefix or suffix of length batch_sz from a single run of k_max
            long_mask = r_lens >= batch_sz
            if np.any(long_mask):
                l_idx = np.flatnonzero(long_mask)
                ls = r_starts[l_idx]
                le = r_ends[l_idx]
                ll = r_lens[l_idx]

                # Prefix [ls, ls + batch_sz)
                pref_loss = csum[ls + batch_sz] - csum[ls]
                pref_left_match = (ls > 0) & (
                    ~self.cp_gap_mask[np.maximum(0, ls - 1)]
                ) & (t[np.maximum(0, ls - 1)] == k_min)
                pref_mid_break = np.where(
                    ll == batch_sz,
                    -(
                        (le < n)
                        & (~self.cp_gap_mask[np.minimum(n - 2, le - 1)])
                        & (t[np.minimum(n - 1, le)] == k_min)
                    ).astype(np.float64),
                    (~self.cp_gap_mask[ls + batch_sz - 1]).astype(np.float64),
                )
                pref_cost = w_t * pref_loss + w_r * (
                    pref_mid_break - pref_left_match.astype(np.float64)
                )

                # Suffix [le - batch_sz, le)
                suff_loss = csum[le] - csum[le - batch_sz]
                suff_right_match = (
                    (le < n)
                    & (~self.cp_gap_mask[np.minimum(n - 2, le - 1)])
                    & (t[np.minimum(n - 1, le)] == k_min)
                )
                suff_mid_break = np.where(
                    ll == batch_sz,
                    -pref_left_match.astype(np.float64),
                    (~self.cp_gap_mask[le - batch_sz - 1]).astype(np.float64),
                )
                suff_cost = w_t * suff_loss + w_r * (
                    suff_mid_break - suff_right_match.astype(np.float64)
                )

                best_pref_i = int(np.argmin(pref_cost))
                best_suff_i = int(np.argmin(suff_cost))
                if pref_cost[best_pref_i] <= suff_cost[best_suff_i]:
                    s_i = int(ls[best_pref_i])
                    t[s_i : s_i + batch_sz] = k_min
                else:
                    e_i = int(le[best_suff_i])
                    t[e_i - batch_sz : e_i] = k_min
                counts[k_max] -= batch_sz
                counts[k_min] += batch_sz

        return t

    def _smooth_beneficial_runs(
        self,
        topics: np.ndarray,
        max_diff: int,
        rng: random.Random,
        max_run_len: int = 16,
        n_passes: int = 3,
    ) -> np.ndarray:
        """Deterministically merges short runs where Delta J < 0 across contiguous boundaries."""
        t = topics.copy()
        n = self.n_codepoints
        w_r = self.config.range_cost_weight
        w_t = 1.0 - w_r
        if w_r <= 0.0:
            return t

        for _ in range(n_passes):
            breaks = np.flatnonzero(t[1:] != t[:-1]) + 1
            if len(breaks) == 0:
                break
            starts = np.concatenate(([0], breaks))
            ends = np.concatenate((breaks, [n]))
            lengths = ends - starts
            cand_runs = np.flatnonzero(lengths <= max_run_len)
            if len(cand_runs) == 0:
                break

            counts = np.bincount(t, minlength=self.n_topics)
            merged_any = False
            for r_idx in cand_runs:
                s_i = int(starts[r_idx])
                e_i = int(ends[r_idx])
                L = e_i - s_i
                old_t = int(t[s_i])
                if counts[old_t] <= L + 1:
                    continue

                has_left = s_i > 0 and not self.cp_gap_mask[s_i - 1]
                has_right = e_i < n and not self.cp_gap_mask[e_i - 1]
                if not has_left and not has_right:
                    continue
                old_in_fg = old_t in self.fg_topics_

                best_new_t = -1
                best_delta_j = 0.0
                old_aff = float(np.sum(self.matrix[old_t, s_i:e_i]))

                if has_left:
                    lt = int(t[s_i - 1])
                    if lt != old_t and (
                        not self.fg_topics_ or (lt in self.fg_topics_) == old_in_fg
                    ):
                        d_range = -2.0 if (has_right and int(t[e_i]) == lt) else -1.0
                        d_topic = old_aff - float(np.sum(self.matrix[lt, s_i:e_i]))
                        dj = w_r * d_range + w_t * d_topic
                        if dj < best_delta_j:
                            best_delta_j = dj
                            best_new_t = lt

                if has_right:
                    rt = int(t[e_i])
                    if (
                        rt != old_t
                        and (not has_left or rt != int(t[s_i - 1]))
                        and (
                            not self.fg_topics_ or (rt in self.fg_topics_) == old_in_fg
                        )
                    ):
                        d_range = -1.0
                        d_topic = old_aff - float(np.sum(self.matrix[rt, s_i:e_i]))
                        dj = w_r * d_range + w_t * d_topic
                        if dj < best_delta_j:
                            best_delta_j = dj
                            best_new_t = rt

                if best_new_t >= 0:
                    t[s_i:e_i] = best_new_t
                    counts[old_t] -= L
                    counts[best_new_t] += L
                    merged_any = True

            if not self._valid_topics(t, max_diff=max_diff):
                t = self._repair_max_diff(t, max_diff, rng)
            if not merged_any:
                break

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
            if self.fg_topics_ and (
                (int(t[i1]) in self.fg_topics_) != (int(t[i2]) in self.fg_topics_)
            ):
                return t
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
            if self.fg_topics_ and (
                (int(t[left]) in self.fg_topics_)
                != (int(t[right - 1]) in self.fg_topics_)
            ):
                return t
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
            if self.fg_topics_ and (
                (int(t[left + 1]) in self.fg_topics_)
                != (int(t[right]) in self.fg_topics_)
            ):
                return t
            t[left + 1], t[right] = t[right], t[left + 1]
            return t

        if mut_name == "SWAP_TOPIC":
            if self.fg_topics_ and len(self.fg_topics_) >= 2:
                if rng.random() < 0.5:
                    k1, k2 = rng.sample(sorted(self.fg_topics_), 2)
                else:
                    bg_pool = [
                        k for k in range(self.n_topics) if k not in self.fg_topics_
                    ]
                    if len(bg_pool) >= 2:
                        k1, k2 = rng.sample(bg_pool, 2)
                    else:
                        k1, k2 = rng.sample(range(self.n_topics), 2)
            else:
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

            # Only consider runs that have at least one contiguous (non-gap) neighbor
            has_left_contig = (starts > 0) & (
                ~self.cp_gap_mask[np.maximum(0, starts - 1)]
            )
            has_right_contig = (ends < n) & (
                ~self.cp_gap_mask[np.minimum(n - 2, ends - 1)]
            )
            contig_mask = has_left_contig | has_right_contig
            if not np.any(contig_mask):
                return self._mutate_topics(anc, "PURELY_RANDOM", rng, max_diff=max_diff)

            contig_lens = lengths[contig_mask]
            min_len = int(np.min(contig_lens))
            short_mask = contig_mask & (lengths <= max(min_len, min(12, min_len + 6)))
            short_candidates = np.flatnonzero(short_mask)
            if len(short_candidates) > 384:
                step = len(short_candidates) // 384
                start_off = rng.randrange(step)
                short_candidates = short_candidates[start_off::step][:384]

            s_indices = starts[short_candidates]
            e_indices = ends[short_candidates]
            c_lens = lengths[short_candidates]
            s_topics = t[s_indices]
            left_idx = np.maximum(0, s_indices - 1)
            right_idx = np.minimum(n - 1, e_indices)
            left_topics = t[left_idx]
            right_topics = t[right_idx]
            c_has_left = has_left_contig[short_candidates]
            c_has_right = has_right_contig[short_candidates]

            s_base = s_indices * self.n_topics
            mat_s = self.matrix_flat[s_base + s_topics]
            loss_left = np.where(
                c_has_left,
                mat_s - self.matrix_flat[s_base + left_topics],
                1e9,
            )
            loss_right = np.where(
                c_has_right,
                mat_s - self.matrix_flat[s_base + right_topics],
                1e9,
            )
            w_r = self.config.range_cost_weight
            w_t = 1.0 - w_r
            bridge_mask = c_has_left & c_has_right & (left_topics == right_topics)
            delta_r_gain = np.where(bridge_mask, -2.0 * w_r, -1.0 * w_r)
            est_delta_j = (
                w_t * np.minimum(loss_left, loss_right) * c_lens + delta_r_gain
            )
            pool_size = min(
                len(short_candidates), max(12, len(short_candidates) // 2)
            )
            top_order = np.argsort(est_delta_j)[:pool_size]
            top_runs = [int(short_candidates[i]) for i in top_order]
            n_merges = min(
                len(top_runs),
                rng.randint(1, max(2, min(28, len(top_runs) // 3 + 4))),
            )
            chosen_runs = rng.sample(top_runs, n_merges)
            counts = np.bincount(t, minlength=self.n_topics)
            slack = (max_diff + 16) if max_diff is not None else n
            for run_idx in chosen_runs:
                s_idx = int(starts[run_idx])
                end_idx = int(ends[run_idx])
                s_len = end_idx - s_idx
                old_topic = int(t[s_idx])
                if counts[old_topic] <= s_len + 1:
                    continue
                old_in_fg = old_topic in self.fg_topics_
                candidates: List[int] = []
                if s_idx > 0 and not self.cp_gap_mask[s_idx - 1]:
                    if (
                        not self.fg_topics_
                        or (int(t[s_idx - 1]) in self.fg_topics_) == old_in_fg
                    ):
                        candidates.append(s_idx - 1)
                if end_idx < n and not self.cp_gap_mask[end_idx - 1]:
                    if (
                        not self.fg_topics_
                        or (int(t[end_idx]) in self.fg_topics_) == old_in_fg
                    ):
                        candidates.append(end_idx)
                if not candidates:
                    continue
                cur_min = int(np.min(counts))
                valid_cands = [
                    x for x in candidates if counts[t[x]] + s_len <= cur_min + slack
                ]
                if not valid_cands:
                    valid_cands = candidates
                neigh = min(
                    valid_cands,
                    key=lambda x: (
                        w_t
                        * float(
                            np.sum(
                                self.matrix[old_topic, s_idx:end_idx]
                                - self.matrix[t[x], s_idx:end_idx]
                            )
                        )
                        + (
                            -2.0 * w_r
                            if len(candidates) == 2
                            and t[candidates[0]] == t[candidates[1]]
                            else -1.0 * w_r
                        ),
                        int(counts[t[x]]),
                    ),
                )
                new_topic = int(t[neigh])
                if new_topic != old_topic:
                    t[s_idx:end_idx] = new_topic
                    counts[old_topic] -= s_len
                    counts[new_topic] += s_len
            return t

        raise ValueError(f"Unknown mutation: {mut_name!r}")

    def _expand_min_topic_border(self, t: np.ndarray, rng: random.Random) -> np.ndarray:
        """Expands an existing contiguous run of the smallest topic by 1 codepoint."""
        n = self.n_codepoints
        counts = np.bincount(t, minlength=self.n_topics)
        min_topic = int(np.argmin(counts))
        is_min = t == min_topic
        adj_min = np.zeros(n, dtype=bool)
        adj_min[1:] |= is_min[:-1] & ~self.cp_gap_mask
        adj_min[:-1] |= is_min[1:] & ~self.cp_gap_mask
        border = np.flatnonzero(adj_min & ~is_min)
        if len(border) == 0:
            adj_min[1:] |= is_min[:-1]
            adj_min[:-1] |= is_min[1:]
            border = np.flatnonzero(adj_min & ~is_min)
        if len(border) == 0:
            return t
        idx = int(border[rng.randrange(len(border))])
        if counts[t[idx]] <= 1:
            return t
        if self.fg_topics_ and (
            (int(t[idx]) in self.fg_topics_) != (min_topic in self.fg_topics_)
        ):
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
        adj_other[1:] |= not_max[:-1] & ~self.cp_gap_mask
        adj_other[:-1] |= not_max[1:] & ~self.cp_gap_mask
        border = np.flatnonzero(is_max & adj_other)
        if len(border) == 0:
            adj_other[1:] |= not_max[:-1]
            adj_other[:-1] |= not_max[1:]
            border = np.flatnonzero(is_max & adj_other)
        if len(border) == 0:
            return t
        idx = int(border[rng.randrange(len(border))])
        neighs = [
            x
            for x in (idx - 1, idx + 1)
            if 0 <= x < n
            and t[x] != max_topic
            and not self.cp_gap_mask[min(idx, x)]
        ]
        if not neighs:
            neighs = [x for x in (idx - 1, idx + 1) if 0 <= x < n and t[x] != max_topic]
        if neighs:
            chosen = min(neighs, key=lambda x: int(counts[t[x]]))
            if self.fg_topics_ and (
                (max_topic in self.fg_topics_) != (int(t[chosen]) in self.fg_topics_)
            ):
                return t
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
            g_diff = init_gene_from_topic_matrix(self.codepoint_list, diff_mat)
            initial_genes.append(g_diff)
        if self.config.init_from_topic_balanced_matrix:
            greedy_mat = balance_topics_greedy(self.matrix, normalization_method="none")
            g_greedy = init_gene_from_topic_matrix(self.codepoint_list, greedy_mat)
            initial_genes.append(g_greedy)
        if (self.config.init_from_binning and not self.fg_topics_) or not initial_genes:
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
        max_iter = max(1, self.config.max_iteration)
        memetic_milestones = {
            max(1, int(round(f * max_iter))): idx
            for idx, f in enumerate((0.15, 0.35, 0.55, 0.75, 0.90))
        }

        for step in range(self.config.max_iteration + 1):
            max_diff = int(decay_fn(step))
            max_diff_changed = max_diff != old_max_diff
            old_max_diff = max_diff

            alive = list(population)
            n_pop = len(population)

            # Progressive curriculum memetic stages (Potts-Viterbi & Island-Bridging)
            if self.config.range_cost_weight > 0.0 and step in memetic_milestones:
                stage_i = memetic_milestones[step]
                run_len = (2, 4, 8, 12, 16)[stage_i]
                t_best_cur = population[0][0]
                t_sm = self._smooth_beneficial_runs(
                    t_best_cur, max_diff, rng, max_run_len=run_len, n_passes=2
                )
                rc, tc, fc = self._topics_cost(t_sm)
                alive.append((t_sm, rc, tc, fc, hash(t_sm.tobytes())))

                if not self.fg_topics_:
                    w_eff = self.config.range_cost_weight * (0.25, 0.45, 0.65, 0.85, 1.0)[stage_i]
                    t_potts = potts_viterbi_smooth_topics(
                        self.codepoint_list,
                        self.matrix,
                        range_cost_weight=w_eff,
                        n_dual_iters=8,
                    )
                    t_potts = self._repair_max_diff(t_potts, max_diff, rng)
                    t_potts = self._smooth_beneficial_runs(
                        t_potts, max_diff, rng, max_run_len=run_len, n_passes=2
                    )
                    rc, tc, fc = self._topics_cost(t_potts)
                    alive.append((t_potts, rc, tc, fc, hash(t_potts.tobytes())))
                elif stage_i >= 2:
                    mb = stage_i + 1
                    t_isl = build_island_bridged_topics(
                        self.codepoint_list,
                        self.matrix,
                        rem_frequency_array=self.rem_frequency_array_,
                        max_bridge=mb,
                    )
                    if t_isl is not None:
                        t_isl = self._repair_max_diff(t_isl, max_diff, rng)
                        t_isl = self._smooth_beneficial_runs(
                            t_isl, max_diff, rng, max_run_len=run_len, n_passes=2
                        )
                        rc, tc, fc = self._topics_cost(t_isl)
                        alive.append((t_isl, rc, tc, fc, hash(t_isl.tobytes())))

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
