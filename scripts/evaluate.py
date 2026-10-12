#!/usr/bin/env python3
"""Benchmark evaluation CLI for MLFont slicing strategies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from mlfont.data import DocumentCorpusProcessor, parse_codepoints_file
from mlfont.evaluator import FontLoadingEvaluator, save_evaluation_results_json
from mlfont.schemas import STANDARD_EVALUATION_PARAMS, SlicingStrategy


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
        description="Evaluate MLFont slicing strategies across standard network profiles."
    )
    parser.add_argument(
        "--codepoints_file",
        type=str,
        required=True,
        help="Path to codepoints.nam file.",
    )
    parser.add_argument(
        "--test_jsonl",
        type=str,
        required=True,
        help="Path to test.jsonl evaluation corpus.",
    )
    parser.add_argument(
        "--strategy_files",
        type=str,
        required=True,
        help="Comma-separated list of strategy .json files to evaluate.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        required=True,
        help="Path to write structured evaluation results JSON.",
    )
    args = parser.parse_args()

    codepoints = sorted(parse_codepoints_file(args.codepoints_file))
    test_docs = _load_documents(Path(args.test_jsonl))
    proc = DocumentCorpusProcessor(test_docs)
    samples = list(proc.generate_codepoint_sets(set(codepoints)))

    strategies = [
        SlicingStrategy.load_json(p.strip())
        for p in args.strategy_files.split(",")
        if p.strip()
    ]

    evaluator = FontLoadingEvaluator(codepoints, samples)
    results = evaluator.evaluate_all(strategies, STANDARD_EVALUATION_PARAMS)
    save_evaluation_results_json(results, args.output_json)

    print(
        f"Evaluated {len(strategies)} strategies across "
        f"{len(STANDARD_EVALUATION_PARAMS)} profiles."
    )
    for r in results:
        if r.params_name == "GIN-3g":
            print(
                f"  [{r.strategy_name}] slices={r.num_slices}, ranges={r.num_ranges}, "
                f"GIN-3g mean={r.total_per_glyph_ms.mean:.2f} ms, "
                f"p95={r.total_per_glyph_ms.p95:.2f} ms, "
                f"session_util={r.session_utilization_pct:.2f}%"
            )


if __name__ == "__main__":
    main()
