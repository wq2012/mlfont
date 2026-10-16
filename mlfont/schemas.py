"""Data structures and configuration schemas for MLFont."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass
class SlicingStrategy:
    """Represents a partition of font codepoints into ordered subsets (slices).

    Attributes:
        subsets: List of slices, where each slice is a sorted list of integer
            Unicode codepoints.
        name: Optional strategy identifier.
        metadata: Optional dictionary of training/configuration metadata.
    """

    subsets: List[List[int]] = field(default_factory=list)
    name: str = ""
    metadata: Dict[str, object] = field(default_factory=dict)

    @property
    def num_slices(self) -> int:
        return len(self.subsets)

    @property
    def num_codepoints(self) -> int:
        return sum(len(s) for s in self.subsets)

    def all_codepoints(self) -> set[int]:
        result: set[int] = set()
        for subset in self.subsets:
            result.update(subset)
        return result

    def to_dna(self) -> List[Tuple[int, int]]:
        """Converts strategy subsets to sorted (codepoint, topic_id) DNA pairs."""
        dna: List[Tuple[int, int]] = []
        for topic_id, subset in enumerate(self.subsets):
            for cp in subset:
                dna.append((int(cp), topic_id))
        dna.sort(key=lambda item: item[0])
        return dna

    @classmethod
    def from_dna(
        cls,
        dna: Sequence[Tuple[int, int]],
        num_topics: Optional[int] = None,
        name: str = "",
        metadata: Optional[Dict[str, object]] = None,
    ) -> "SlicingStrategy":
        """Constructs a SlicingStrategy from (codepoint, topic_id) pairs."""
        if not dna:
            return cls(subsets=[], name=name, metadata=metadata or {})
        max_topic = max(topic for _, topic in dna)
        total_topics = num_topics if num_topics is not None else (max_topic + 1)
        if max_topic >= total_topics:
            raise ValueError(
                f"DNA contains topic {max_topic} >= num_topics ({total_topics})."
            )
        slices: List[List[int]] = [[] for _ in range(total_topics)]
        for cp, topic in sorted(dna, key=lambda item: item[0]):
            slices[topic].append(int(cp))
        # Filter out any empty slices only if num_topics was inferred
        return cls(subsets=slices, name=name, metadata=metadata or {})

    def count_unicode_ranges(self) -> int:
        """Counts the total number of contiguous CSS unicode-range intervals.

        Each slice is sorted independently and consecutive codepoints
        (cp_next == cp_prev + 1) within the same slice are merged into one range.
        """
        total_ranges = 0
        for subset in self.subsets:
            if not subset:
                continue
            sorted_cps = sorted(set(subset))
            total_ranges += 1
            for idx in range(1, len(sorted_cps)):
                if sorted_cps[idx] != sorted_cps[idx - 1] + 1:
                    total_ranges += 1
        return total_ranges

    @property
    def num_ranges(self) -> int:
        """Total number of contiguous CSS unicode-range intervals across all slices."""
        return self.count_unicode_ranges()

    def total_num_ranges(self) -> int:
        """Returns the total number of contiguous CSS unicode-range intervals."""
        return self.count_unicode_ranges()

    def count_dna_compactness_ranges(self) -> int:
        """Counts range breaks across the sorted global DNA sequence.

        Matches the Range-Compact GA cost ``C_range(s)``: a new range starts
        whenever (cp != prev_cp + 1) or (topic != prev_topic). This equals
        ``count_unicode_ranges()`` whenever every codepoint belongs to exactly
        one slice.
        """
        dna = self.to_dna()
        if not dna:
            return 0
        ranges = 0
        last_cp: Optional[int] = None
        last_topic: Optional[int] = None
        for cp, topic in dna:
            if (last_cp is None) or (cp - 1 != last_cp) or (topic != last_topic):
                ranges += 1
            last_cp = cp
            last_topic = topic
        return ranges

    def save_json(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "name": self.name,
            "num_slices": self.num_slices,
            "num_codepoints": self.num_codepoints,
            "num_unicode_ranges": self.count_unicode_ranges(),
            "metadata": self.metadata,
            "subsets": [sorted(int(c) for c in s) for s in self.subsets],
        }
        p.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> "SlicingStrategy":
        p = Path(path)
        payload = json.loads(p.read_text(encoding="utf-8"))
        return cls(
            subsets=[[int(c) for c in s] for s in payload["subsets"]],
            name=payload.get("name", p.stem),
            metadata=payload.get("metadata", {}),
        )

    def save_nam_dir(self, directory: str | Path, prefix: str = "slice") -> List[Path]:
        """Saves each slice as a .nam codepoint subset file."""
        out_dir = Path(directory)
        out_dir.mkdir(parents=True, exist_ok=True)
        paths: List[Path] = []
        for idx, subset in enumerate(self.subsets):
            p = out_dir / f"{prefix}_{idx:03d}.nam"
            lines = [f"0x{cp:04X}" for cp in sorted(subset)]
            p.write_text("\n".join(lines) + "\n", encoding="utf-8")
            paths.append(p)
        return paths


@dataclass
class EvaluationParams:
    """Network and browsing session parameters for font loading simulation."""

    name: str
    rtt_ms: float
    bandwidth_bytes_per_ms: float
    average_glyph_size_bytes: float = 200.0
    session_length: int = 10


STANDARD_EVALUATION_PARAMS: List[EvaluationParams] = [
    EvaluationParams(
        name="GIN-2g-poor",
        rtt_ms=1600.0,
        bandwidth_bytes_per_ms=2.5,  # 20 Kbps
        average_glyph_size_bytes=200.0,
        session_length=10,
    ),
    EvaluationParams(
        name="GIN-2g",
        rtt_ms=1200.0,
        bandwidth_bytes_per_ms=12.5,  # 100 Kbps
        average_glyph_size_bytes=200.0,
        session_length=10,
    ),
    EvaluationParams(
        name="GIN-3g",
        rtt_ms=400.0,
        bandwidth_bytes_per_ms=50.0,  # 400 Kbps
        average_glyph_size_bytes=200.0,
        session_length=10,
    ),
    EvaluationParams(
        name="residential-DSL",
        rtt_ms=50.0,
        bandwidth_bytes_per_ms=250.0,  # 2 Mbps
        average_glyph_size_bytes=200.0,
        session_length=10,
    ),
    EvaluationParams(
        name="residential-ADSL2+",
        rtt_ms=50.0,
        bandwidth_bytes_per_ms=2500.0,  # 20 Mbps
        average_glyph_size_bytes=200.0,
        session_length=10,
    ),
    EvaluationParams(
        name="GIN-3g_no_session",
        rtt_ms=400.0,
        bandwidth_bytes_per_ms=50.0,
        average_glyph_size_bytes=200.0,
        session_length=1,
    ),
    EvaluationParams(
        name="GIN-3g_short_session",
        rtt_ms=400.0,
        bandwidth_bytes_per_ms=50.0,
        average_glyph_size_bytes=200.0,
        session_length=3,
    ),
]


@dataclass
class DistributionStats:
    """Summary statistics for a latency or utilization distribution."""

    mean: float
    p95: float
    minimal: float
    maximal: float
    stddev: float


@dataclass
class StrategyEvaluationResult:
    """Evaluation metrics for a single strategy under one EvaluationParams."""

    strategy_name: str
    params_name: str
    num_slices: int
    num_codepoints: int
    num_samples: int
    num_missing_codepoints: int
    num_ranges: int
    total_per_glyph_ms: DistributionStats
    rtt_per_glyph_ms: DistributionStats
    download_per_glyph_ms: DistributionStats
    upfront_utilization_pct: float
    session_utilization_pct: float
    num_unassigned_font_codepoints: int = 0


@dataclass
class DecayConfig:
    """Configuration for decaying max slice-size difference in Range-Compact GA."""

    min_val: float = 30.0
    max_val: float = 50.0
    decay_method: str = "POWER"  # "POWER" or "EXPONENTIAL"
    decay_power: float = 3.0
    decay_rate: float = 0.1


@dataclass
class RangeCompactConfig:
    """Configuration for the Range-Compact Genetic Algorithm."""

    range_cost_weight: float = 0.005
    max_iteration: int = 5000
    n_survivals: int = 50
    n_mutations_per_iteration: int = 100
    n_crossovers_per_iteration: int = 50
    init_from_matrix: bool = True
    init_from_topic_balanced_matrix: bool = True
    init_from_binning: bool = True
    decay_max_diff: DecayConfig = field(default_factory=DecayConfig)
    mutation_probabilities: Dict[str, float] = field(
        default_factory=lambda: {
            "PURELY_RANDOM": 0.15,
            "LEFT_SHIFT": 0.25,
            "RIGHT_SHIFT": 0.25,
            "SWAP_TOPIC": 0.15,
            "INCREASE_MIN_TOPIC": 0.05,
            "DECREASE_MAX_TOPIC": 0.05,
            "MERGE_SHORT_RANGE": 0.10,
        }
    )
    history_interval: int = 10
    random_seed: Optional[int] = 42
    population_size: Optional[int] = None
    num_iterations: Optional[int] = None
    decay_power: Optional[float] = None
    random_state: Optional[int] = None

    def __post_init__(self) -> None:
        if self.population_size is not None:
            self.n_survivals = int(self.population_size)
        if self.num_iterations is not None:
            self.max_iteration = int(self.num_iterations)
        if self.decay_power is not None:
            self.decay_max_diff.decay_power = float(self.decay_power)
        if self.random_state is not None:
            self.random_seed = int(self.random_state)


@dataclass
class GeneticHistoryEvent:
    """Snapshot of population costs at a single GA iteration."""

    iteration: int
    elapsed_sec: float
    best_range_cost: float
    best_topic_cost: float
    best_final_cost: float
    avg_range_cost: float
    avg_topic_cost: float
    avg_final_cost: float
    min_range_cost: float
    min_topic_cost: float
    max_diff: int
    num_survivals: int
    best_raw_ranges: int
