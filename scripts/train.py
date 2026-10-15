#!/usr/bin/env python3
"""Model training CLI for MLFont (Frequency, Online LDA, DNNTM, and Range-Compact GA)."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import List

import numpy as np

from mlfont.configs import get_preset_config
from mlfont.data import DocumentCorpusProcessor, parse_codepoints_file
from mlfont.dnntm import DnnTmConfig, DnnTmLearner, DnnTmLossType
from mlfont.export import save_safetensors_model
from mlfont.frequency import generate_most_and_equal_bin_strategy
from mlfont.olda import OnlineLdaConfig, OnlineLdaLearner
from mlfont.range_compact import RangeCompactGeneticLearner
from mlfont.schemas import DecayConfig, RangeCompactConfig


def _load_documents(jsonl_path: Path) -> List[str]:
    docs: List[str] = []
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            obj = json.loads(line)
            if obj.get("text"):
                docs.append(obj["text"])
    return docs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train MLFont slicing strategies from a document corpus."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing codepoints.nam and train.jsonl.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save learned matrices, strategies, and GA history.",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="",
        help="Optional named preset (chinese, traditional_chinese, japanese, korean, lgc).",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="range_compact",
        choices=["most_equal_bin", "online_lda", "dnntm", "range_compact"],
        help="Training algorithm to execute.",
    )
    parser.add_argument(
        "--topic_model",
        type=str,
        default="online_lda",
        choices=["online_lda", "dnntm"],
        help="Underlying topic model when --algorithm=range_compact.",
    )
    parser.add_argument("--n_topics", type=int, default=20, help="Number of topic slices.")
    parser.add_argument(
        "--most_sizes",
        type=str,
        default="",
        help="Comma-separated high-frequency slice sizes (e.g. '1000,500').",
    )
    parser.add_argument(
        "--normalization_method",
        type=str,
        default="sinkhorn",
        choices=[
            "sinkhorn",
            "bistochastic",
            "topic_sum",
            "row_sum",
            "global_max",
            "spectral_norm",
            "none",
        ],
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default=DnnTmLossType.ALL_COMBINED.value,
        choices=[e.value for e in DnnTmLossType],
        help="Loss objective variant for DNNTM training.",
    )
    parser.add_argument(
        "--range_cost_weight",
        type=float,
        default=0.012,
        help="Weight w_r on CSS unicode-range compactness cost in Range-Compact GA.",
    )
    parser.add_argument(
        "--max_iteration",
        type=int,
        default=1200,
        help="Maximum GA iterations for range_compact.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    codepoints = sorted(parse_codepoints_file(data_dir / "codepoints.nam"))
    train_docs = _load_documents(data_dir / "train.jsonl")
    corpus = DocumentCorpusProcessor(train_docs, random_seed=args.seed)
    freq_arr = corpus.count_frequency_array(codepoints, deduplicate_within_doc=True)
    np.save(out_dir / "frequency_array.npy", freq_arr)

    if args.preset:
        preset = get_preset_config(args.preset)
        olda_cfg = preset.olda_config
        dnntm_cfg = preset.dnntm_config
        ga_cfg = preset.range_compact_config
    else:
        most_list = [int(x) for x in args.most_sizes.split(",") if x.strip()]
        olda_cfg = OnlineLdaConfig(
            n_topics=args.n_topics,
            most_sizes=most_list,
            normalization_method=args.normalization_method,
            random_state=args.seed,
        )
        dnntm_cfg = DnnTmConfig(
            n_topics=args.n_topics,
            most_sizes=most_list,
            normalization_method=args.normalization_method,
            loss_type=args.loss_type,
            random_seed=args.seed,
        )
        ga_cfg = RangeCompactConfig(
            range_cost_weight=args.range_cost_weight,
            max_iteration=args.max_iteration,
            decay_max_diff=DecayConfig(min_val=20.0, max_val=40.0),
            random_seed=args.seed,
        )

    if args.algorithm == "most_equal_bin":
        strategy = generate_most_and_equal_bin_strategy(
            codepoints,
            frequency_array=freq_arr,
            most_sizes=olda_cfg.most_sizes,
            num_bins=olda_cfg.n_topics,
        )
        strategy.save_json(out_dir / "strategy.json")
        print(f"Saved Most-and-Equal-Bin strategy ({strategy.num_slices} slices) to {out_dir}")
        return

    if args.algorithm == "online_lda" or (
        args.algorithm == "range_compact" and args.topic_model == "online_lda"
    ):
        learner = OnlineLdaLearner(corpus, codepoints, olda_cfg)
        topic_strategy = learner.fit()
        lambda_matrix = learner.lambda_matrix_
        rem_cps = learner.remaining_codepoints_ or codepoints
        most_sizes = olda_cfg.most_sizes
        norm_method = olda_cfg.normalization_method
    else:
        learner_dnn = DnnTmLearner(corpus, codepoints, dnntm_cfg)
        topic_strategy = learner_dnn.fit()
        lambda_matrix = learner_dnn.lambda_matrix_
        rem_cps = learner_dnn.remaining_codepoints_ or codepoints
        most_sizes = dnntm_cfg.most_sizes
        norm_method = dnntm_cfg.normalization_method

    assert lambda_matrix is not None
    np.save(out_dir / "lambda_matrix.npy", lambda_matrix)
    np.save(out_dir / "remaining_codepoints.npy", np.asarray(rem_cps, dtype=np.int32))
    save_safetensors_model(
        out_dir / "model.safetensors",
        lambda_matrix=lambda_matrix,
        codepoints=codepoints,
        frequency_array=freq_arr,
        remaining_codepoints=rem_cps,
    )
    topic_strategy.save_json(out_dir / "topic_strategy.json")

    if args.algorithm in ("online_lda", "dnntm"):
        topic_strategy.save_json(out_dir / "strategy.json")
        print(
            f"Saved {args.algorithm} strategy ({topic_strategy.num_slices} slices, "
            f"{topic_strategy.count_unicode_ranges()} ranges) to {out_dir}"
        )
        return

    # Run Range-Compact GA on the learned lambda_matrix
    ga_learner = RangeCompactGeneticLearner(
        codepoint_list=codepoints,
        lambda_matrix=lambda_matrix,
        config=ga_cfg,
        frequency_array=freq_arr,
        most_sizes=most_sizes,
        normalization_method=norm_method,
    )
    ga_strategy = ga_learner.run()
    ga_strategy.save_json(out_dir / "strategy.json")
    (out_dir / "ga_history.json").write_text(
        json.dumps([asdict(e) for e in ga_learner.history_], indent=2),
        encoding="utf-8",
    )
    print(
        f"Saved Range-Compact GA strategy ({ga_strategy.num_slices} slices, "
        f"{ga_strategy.count_unicode_ranges()} ranges) to {out_dir}"
    )


if __name__ == "__main__":
    main()
