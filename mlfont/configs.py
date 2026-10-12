"""Pre-configured paper presets for MLFont training and optimization."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

from mlfont.dnntm import DnnTmConfig
from mlfont.olda import OnlineLdaConfig
from mlfont.schemas import DecayConfig, RangeCompactConfig

DEFAULT_GA_MUTATION_PROBS: Dict[str, float] = {
    "PURELY_RANDOM": 0.10,
    "LEFT_SHIFT": 0.20,
    "RIGHT_SHIFT": 0.20,
    "SWAP_TOPIC": 0.10,
    "INCREASE_MIN_TOPIC": 0.05,
    "DECREASE_MAX_TOPIC": 0.05,
    "MERGE_SHORT_RANGE": 0.30,
}


@dataclass
class MLFontPreset:
    """Complete configuration preset for a font script/language family."""

    name: str
    description: str
    olda_config: OnlineLdaConfig
    dnntm_config: DnnTmConfig
    range_compact_config: RangeCompactConfig
    metadata: Dict[str, object] = field(default_factory=dict)


def chinese_production_config() -> MLFontPreset:
    """Preset matching Google Fonts Early Access 'Noto Sans SC Sliced' (102 slices)."""
    return MLFontPreset(
        name="chinese_production",
        description=(
            "Simplified Chinese (Noto Sans SC): 2 frequency tiers (1000, 500) "
            "+ 100 Online LDA topics + Range-Compact GA"
        ),
        olda_config=OnlineLdaConfig(
            n_topics=100,
            most_sizes=[1000, 500],
            topic_balancing="greedy",
            normalization_method="global_max",
            batch_size=1000,
            n_batches=20,
            random_state=42,
        ),
        dnntm_config=DnnTmConfig(
            n_topics=100,
            most_sizes=[1000, 500],
            topic_balancing="greedy",
            normalization_method="global_max",
            batch_size=256,
            n_layers=2,
            layer_size=256,
            training_steps=400,
            loss_type="ALL_COMBINED",
        ),
        range_compact_config=RangeCompactConfig(
            range_cost_weight=0.005,
            max_iteration=1200,
            n_survivals=40,
            n_mutations_per_iteration=80,
            n_crossovers_per_iteration=40,
            init_from_matrix=True,
            init_from_topic_balanced_matrix=True,
            init_from_binning=False,
            decay_max_diff=DecayConfig(
                min_val=30.0, max_val=50.0, decay_method="POWER", decay_power=3.0
            ),
            mutation_probabilities=dict(DEFAULT_GA_MUTATION_PROBS),
            random_seed=42,
        ),
    )


def traditional_chinese_production_config() -> MLFontPreset:
    """Preset for Traditional Chinese (Noto Sans TC)."""
    return MLFontPreset(
        name="traditional_chinese_production",
        description=(
            "Traditional Chinese (Noto Sans TC): 2 frequency tiers (600, 600) "
            "+ 80 Online LDA topics + Range-Compact GA"
        ),
        olda_config=OnlineLdaConfig(
            n_topics=80,
            most_sizes=[600, 600],
            topic_balancing="greedy",
            normalization_method="global_max",
            batch_size=1000,
            n_batches=20,
            random_state=42,
        ),
        dnntm_config=DnnTmConfig(
            n_topics=80,
            most_sizes=[600, 600],
            topic_balancing="greedy",
            normalization_method="global_max",
            batch_size=256,
            n_layers=2,
            layer_size=256,
            training_steps=400,
            loss_type="ALL_COMBINED",
        ),
        range_compact_config=RangeCompactConfig(
            range_cost_weight=0.005,
            max_iteration=1200,
            n_survivals=40,
            n_mutations_per_iteration=80,
            n_crossovers_per_iteration=40,
            init_from_matrix=True,
            init_from_topic_balanced_matrix=True,
            init_from_binning=False,
            decay_max_diff=DecayConfig(
                min_val=30.0, max_val=50.0, decay_method="POWER", decay_power=3.0
            ),
            mutation_probabilities=dict(DEFAULT_GA_MUTATION_PROBS),
            random_seed=42,
        ),
    )


def japanese_production_config() -> MLFontPreset:
    """Preset for Japanese (Noto Sans JP)."""
    return MLFontPreset(
        name="japanese_production",
        description=(
            "Japanese (Noto Sans JP): 2 frequency tiers (500, 500) "
            "+ 60 Online LDA topics + Range-Compact GA"
        ),
        olda_config=OnlineLdaConfig(
            n_topics=60,
            most_sizes=[500, 500],
            topic_balancing="greedy",
            normalization_method="global_max",
            batch_size=1000,
            n_batches=20,
            random_state=42,
        ),
        dnntm_config=DnnTmConfig(
            n_topics=60,
            most_sizes=[500, 500],
            topic_balancing="greedy",
            normalization_method="global_max",
            batch_size=256,
            n_layers=2,
            layer_size=256,
            training_steps=400,
            loss_type="ALL_COMBINED",
        ),
        range_compact_config=RangeCompactConfig(
            range_cost_weight=0.005,
            max_iteration=1200,
            n_survivals=40,
            n_mutations_per_iteration=80,
            n_crossovers_per_iteration=40,
            init_from_matrix=True,
            init_from_topic_balanced_matrix=True,
            init_from_binning=False,
            decay_max_diff=DecayConfig(
                min_val=30.0, max_val=50.0, decay_method="POWER", decay_power=3.0
            ),
            mutation_probabilities=dict(DEFAULT_GA_MUTATION_PROBS),
            random_seed=42,
        ),
    )


def korean_production_config() -> MLFontPreset:
    """Preset matching Korean (Noto Sans KR, 20170619)."""
    return MLFontPreset(
        name="korean_production",
        description=(
            "Korean (Noto Sans KR): 2 frequency tiers (300, 300) "
            "+ 50 Online LDA topics + Range-Compact GA"
        ),
        olda_config=OnlineLdaConfig(
            n_topics=50,
            most_sizes=[300, 300],
            topic_balancing="greedy",
            normalization_method="global_max",
            batch_size=1000,
            n_batches=20,
            random_state=42,
        ),
        dnntm_config=DnnTmConfig(
            n_topics=50,
            most_sizes=[300, 300],
            topic_balancing="greedy",
            normalization_method="global_max",
            batch_size=256,
            n_layers=2,
            layer_size=256,
            training_steps=400,
            loss_type="ALL_COMBINED",
        ),
        range_compact_config=RangeCompactConfig(
            range_cost_weight=0.005,
            max_iteration=1200,
            n_survivals=40,
            n_mutations_per_iteration=80,
            n_crossovers_per_iteration=40,
            init_from_matrix=True,
            init_from_topic_balanced_matrix=True,
            init_from_binning=False,
            decay_max_diff=DecayConfig(
                min_val=30.0, max_val=50.0, decay_method="POWER", decay_power=3.0
            ),
            mutation_probabilities=dict(DEFAULT_GA_MUTATION_PROBS),
            random_seed=42,
        ),
    )


def lgc_production_config() -> MLFontPreset:
    """Preset matching Pan-European Latin/Greek/Cyrillic (LGC, 20161017 / 20161104)."""
    return MLFontPreset(
        name="lgc_production",
        description="Pan-European LGC (Noto Sans): 20 Online LDA / DNNTM topics + Range-Compact GA",
        olda_config=OnlineLdaConfig(
            n_topics=20,
            most_sizes=[],
            topic_balancing="greedy",
            normalization_method="global_max",
            batch_size=1000,
            n_batches=20,
            random_state=42,
        ),
        dnntm_config=DnnTmConfig(
            n_topics=20,
            most_sizes=[],
            topic_balancing="greedy",
            normalization_method="global_max",
            batch_size=256,
            n_layers=2,
            layer_size=128,
            training_steps=300,
            loss_type="ALL_COMBINED",
        ),
        range_compact_config=RangeCompactConfig(
            range_cost_weight=0.005,
            max_iteration=1200,
            n_survivals=40,
            n_mutations_per_iteration=80,
            n_crossovers_per_iteration=40,
            init_from_matrix=True,
            init_from_topic_balanced_matrix=True,
            init_from_binning=False,
            decay_max_diff=DecayConfig(
                min_val=20.0, max_val=40.0, decay_method="POWER", decay_power=3.0
            ),
            mutation_probabilities=dict(DEFAULT_GA_MUTATION_PROBS),
            random_seed=42,
        ),
    )


PRESET_REGISTRY = {
    "chinese": chinese_production_config,
    "traditional_chinese": traditional_chinese_production_config,
    "japanese": japanese_production_config,
    "korean": korean_production_config,
    "lgc": lgc_production_config,
}


def get_preset_config(name: str) -> MLFontPreset:
    """Returns a named MLFontPreset configuration."""
    key = name.lower().strip()
    if key not in PRESET_REGISTRY:
        raise ValueError(
            f"Unknown preset {name!r}. Available presets: {sorted(PRESET_REGISTRY.keys())}"
        )
    return PRESET_REGISTRY[key]()
