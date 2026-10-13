#!/usr/bin/env python3
"""Data ingestion and corpus preparation CLI for MLFont."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from typing import List

from mlfont.data import (
    extract_codepoints_from_font,
    parse_codepoints_file,
    save_codepoints_file,
)

NOTO_FONT_URLS = {
    "lgc": (
        "NotoSans-Regular.ttf",
        "https://raw.githubusercontent.com/notofonts/noto-fonts/main/"
        "hinted/ttf/NotoSans/NotoSans-Regular.ttf",
    ),
    "kr": (
        "NotoSansKR-Regular.otf",
        "https://raw.githubusercontent.com/notofonts/noto-cjk/main/"
        "Sans/OTF/Korean/NotoSansCJKkr-Regular.otf",
    ),
    "jp": (
        "NotoSansJP-Regular.otf",
        "https://raw.githubusercontent.com/notofonts/noto-cjk/main/"
        "Sans/OTF/Japanese/NotoSansCJKjp-Regular.otf",
    ),
    "tc": (
        "NotoSansTC-Regular.otf",
        "https://raw.githubusercontent.com/notofonts/noto-cjk/main/"
        "Sans/OTF/TraditionalChinese/NotoSansCJKtc-Regular.otf",
    ),
    "sc": (
        "NotoSansSC-Regular.otf",
        "https://raw.githubusercontent.com/notofonts/noto-cjk/main/"
        "Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Regular.otf",
    ),
}


def download_noto_font(font_key: str, dest_dir: Path) -> Path:
    """Downloads an official Google Noto font file if not already cached in dest_dir."""
    import urllib.request

    if font_key not in NOTO_FONT_URLS:
        raise ValueError(f"Unknown Noto font key: {font_key!r}")
    filename, url = NOTO_FONT_URLS[font_key]
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / filename
    if not dest_path.exists():
        urllib.request.urlretrieve(url, str(dest_path))
    return dest_path


def generate_synthetic_documents(
    codepoints: List[int],
    num_docs: int = 500,
    num_clusters: int = 10,
    seed: int = 42,
) -> List[str]:
    """Generates synthetic documents with realistic frequency skew and co-occurrence clusters."""
    rng = random.Random(seed)
    n = len(codepoints)
    if n == 0:
        return []
    head_size = min(max(10, n // 10), 500)
    head_cps = codepoints[:head_size]
    tail_cps = codepoints[head_size:] if head_size < n else codepoints

    clusters: List[List[int]] = [[] for _ in range(max(1, num_clusters))]
    for idx, cp in enumerate(tail_cps):
        clusters[idx % len(clusters)].append(cp)

    docs: List[str] = []
    for _ in range(num_docs):
        chosen_cluster = rng.choice(clusters)
        k_head = min(len(head_cps), rng.randint(15, 60))
        k_tail = min(len(chosen_cluster), rng.randint(10, 50))
        doc_cps = rng.sample(head_cps, k_head) + rng.sample(chosen_cluster, k_tail)
        docs.append("".join(chr(c) for c in doc_cps))
    return docs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare codepoint list and train/test document splits for MLFont."
    )
    parser.add_argument(
        "--font_path",
        type=str,
        default="",
        help="Path to a .ttf/.otf font file to extract supported Unicode codepoints.",
    )
    parser.add_argument(
        "--download_noto_font",
        type=str,
        default="",
        choices=["", "lgc", "kr", "jp", "tc", "sc"],
        help="Optional Noto font key (lgc, kr, jp, tc, sc) to download automatically.",
    )
    parser.add_argument(
        "--codepoints_file",
        type=str,
        default="",
        help="Optional existing .nam / hex codepoints file if --font_path is not used.",
    )
    parser.add_argument(
        "--text_dir",
        type=str,
        default="",
        help="Directory of plain-text (.txt) or .jsonl files to ingest.",
    )
    parser.add_argument(
        "--hf_dataset",
        type=str,
        default="",
        help="Optional Hugging Face dataset ID (e.g., wikimedia/wikipedia).",
    )
    parser.add_argument(
        "--hf_config",
        type=str,
        default="",
        help="Optional Hugging Face dataset config/subset (e.g., 20231101.zh).",
    )
    parser.add_argument(
        "--max_docs",
        type=int,
        default=3500,
        help="Maximum number of documents to load from --hf_dataset.",
    )
    parser.add_argument(
        "--generate_synthetic",
        action="store_true",
        help="Generate a synthetic document corpus for fast testing.",
    )
    parser.add_argument(
        "--num_synthetic",
        type=int,
        default=400,
        help="Number of synthetic documents to generate when --generate_synthetic is set.",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Fraction of documents assigned to the training split.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for codepoints.nam, train.jsonl, and test.jsonl.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.download_noto_font and not args.font_path:
        dl_font = download_noto_font(args.download_noto_font, out_dir / "fonts")
        codepoints = extract_codepoints_from_font(dl_font)
    elif args.font_path:
        codepoints = extract_codepoints_from_font(args.font_path)
    elif args.codepoints_file:
        codepoints = sorted(parse_codepoints_file(args.codepoints_file))
    else:
        # Default synthetic codepoint set (1,000 CJK + Latin codepoints)
        codepoints = list(range(0x0020, 0x007F)) + list(range(0x4E00, 0x4E00 + 905))

    save_codepoints_file(codepoints, out_dir / "codepoints.nam")

    documents: List[str] = []
    if args.generate_synthetic:
        documents = generate_synthetic_documents(
            codepoints, num_docs=args.num_synthetic, seed=args.seed
        )
    elif args.hf_dataset:
        from datasets import load_dataset

        ds = load_dataset(
            args.hf_dataset,
            args.hf_config or None,
            split="train",
            streaming=True,
        )
        for row in ds:
            txt = str(row.get("text", "")).strip()
            if len(txt) >= 80:
                documents.append(txt[:4000])
                if len(documents) >= args.max_docs:
                    break
    elif args.text_dir:
        tdir = Path(args.text_dir)
        for p in sorted(tdir.rglob("*")):
            if p.suffix.lower() == ".txt":
                text = p.read_text(encoding="utf-8", errors="ignore").strip()
                if text:
                    documents.append(text)
            elif p.suffix.lower() == ".jsonl":
                for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
                    if line.strip():
                        obj = json.loads(line)
                        txt = obj.get("text", "")
                        if txt:
                            documents.append(txt)
    else:
        raise ValueError(
            "Provide --text_dir, --hf_dataset, or --generate_synthetic"
        )

    rng = random.Random(args.seed)
    rng.shuffle(documents)
    split_idx = max(1, int(len(documents) * args.train_ratio))
    train_docs = documents[:split_idx]
    test_docs = documents[split_idx:] if split_idx < len(documents) else documents

    with (out_dir / "train.jsonl").open("w", encoding="utf-8") as f:
        for idx, doc in enumerate(train_docs):
            f.write(json.dumps({"id": idx, "text": doc}, ensure_ascii=False) + "\n")

    with (out_dir / "test.jsonl").open("w", encoding="utf-8") as f:
        for idx, doc in enumerate(test_docs):
            f.write(json.dumps({"id": idx, "text": doc}, ensure_ascii=False) + "\n")

    print(
        f"Prepared {len(codepoints)} codepoints, {len(train_docs)} train docs, "
        f"and {len(test_docs)} test docs in {out_dir}"
    )


if __name__ == "__main__":
    main()
