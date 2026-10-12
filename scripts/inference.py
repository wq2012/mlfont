#!/usr/bin/env python3
"""Production font slicing and web-page slice analysis CLI for MLFont."""

from __future__ import annotations

import argparse

from mlfont.export import load_pretrained
from mlfont.schemas import SlicingStrategy
from mlfont.subsetter import slice_font_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run MLFont inference: slice a font into WOFF2+CSS or analyze text slice usage."
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="",
        help="Path to a pretrained MLFont bundle directory.",
    )
    parser.add_argument(
        "--strategy_json",
        type=str,
        default="",
        help="Path to a strategy.json file (if --model_dir is not provided).",
    )
    parser.add_argument(
        "--font_path",
        type=str,
        default="",
        help="Optional path to a .ttf/.otf font file to slice.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Directory to write WOFF2 slices and CSS when --font_path is provided.",
    )
    parser.add_argument(
        "--text",
        type=str,
        default="",
        help="Optional sample text to analyze required font slices and glyph utilization.",
    )
    args = parser.parse_args()

    if args.model_dir:
        bundle = load_pretrained(args.model_dir)
        strategy = bundle.strategy
    elif args.strategy_json:
        strategy = SlicingStrategy.load_json(args.strategy_json)
    else:
        raise ValueError("Provide either --model_dir or --strategy_json")

    if args.font_path and args.output_dir:
        artifacts, css_path = slice_font_file(
            font_path=args.font_path,
            strategy=strategy,
            output_dir=args.output_dir,
            flavor="woff2",
        )
        total_bytes = sum(a.byte_size for a in artifacts)
        print(
            f"Sliced {args.font_path} into {len(artifacts)} WOFF2 files "
            f"({total_bytes / 1024.0:.1f} KB total) and {css_path}"
        )

    if args.text:
        all_cps = strategy.all_codepoints()
        text_cps = {ord(ch) for ch in args.text} & all_cps
        req_slices = []
        downloaded_glyphs = 0
        for idx, subset in enumerate(strategy.subsets):
            if text_cps.intersection(subset):
                req_slices.append(idx)
                downloaded_glyphs += len(subset)
        util_pct = (
            100.0 * len(text_cps) / downloaded_glyphs if downloaded_glyphs > 0 else 0.0
        )
        print(
            f"Text analysis: {len(text_cps)} unique target glyphs -> "
            f"{len(req_slices)}/{strategy.num_slices} slices triggered {req_slices}, "
            f"{downloaded_glyphs} glyphs downloaded ({util_pct:.2f}% utilization)"
        )


if __name__ == "__main__":
    main()
