#!/usr/bin/env python3
"""Multi-format model & font slice export CLI for MLFont."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from mlfont.export import export_pretrained_bundle
from mlfont.schemas import SlicingStrategy


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a trained MLFont directory into a Hugging Face pretrained bundle."
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help=(
            "Directory containing strategy.json "
            "(and optional lambda_matrix.npy / frequency_array.npy)."
        ),
    )
    parser.add_argument(
        "--export_dir",
        type=str,
        required=True,
        help="Target directory for the exported pretrained bundle.",
    )
    parser.add_argument(
        "--font_path",
        type=str,
        default="",
        help="Optional path to .ttf/.otf font file to slice into WOFF2.",
    )
    parser.add_argument(
        "--font_family",
        type=str,
        default="",
        help="Optional CSS font-family name.",
    )
    args = parser.parse_args()

    ckpt = Path(args.checkpoint_dir)
    strategy = SlicingStrategy.load_json(ckpt / "strategy.json")

    lambda_mat = (
        np.load(ckpt / "lambda_matrix.npy")
        if (ckpt / "lambda_matrix.npy").exists()
        else None
    )
    freq_arr = (
        np.load(ckpt / "frequency_array.npy")
        if (ckpt / "frequency_array.npy").exists()
        else None
    )

    out_dir = export_pretrained_bundle(
        output_dir=args.export_dir,
        strategy=strategy,
        lambda_matrix=lambda_mat,
        frequency_array=freq_arr,
        font_path=args.font_path if args.font_path else None,
        font_family=args.font_family if args.font_family else None,
        slice_font=bool(args.font_path),
    )
    print(f"Exported pretrained MLFont bundle to {out_dir}")


if __name__ == "__main__":
    main()
