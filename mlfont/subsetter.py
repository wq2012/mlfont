"""Font subsetting (TTF/OTF/WOFF2 slicing) and CSS @font-face generation."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import multiprocessing
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from fontTools import subset as ft_subset
from fontTools.ttLib import TTFont

from mlfont.schemas import SlicingStrategy


def format_unicode_ranges(codepoints: Iterable[int]) -> List[str]:
    """Formats a collection of integer codepoints into CSS unicode-range intervals.

    Contiguous codepoints [a, a+1, ..., b] are merged into 'U+XXXX-YYYY',
    while isolated codepoints become 'U+XXXX'.
    """
    sorted_cps = sorted(set(int(cp) for cp in codepoints))
    if not sorted_cps:
        return []

    ranges: List[str] = []
    start = sorted_cps[0]
    prev = start

    for cp in sorted_cps[1:]:
        if cp == prev + 1:
            prev = cp
        else:
            if start == prev:
                ranges.append(f"U+{start:X}")
            else:
                ranges.append(f"U+{start:X}-{prev:X}")
            start = cp
            prev = cp

    if start == prev:
        ranges.append(f"U+{start:X}")
    else:
        ranges.append(f"U+{start:X}-{prev:X}")

    return ranges


def generate_css(
    strategy: SlicingStrategy,
    font_family: str,
    font_url_prefix: str = "",
    font_basename: str = "font",
    extension: str = "woff2",
    font_style: str = "normal",
    font_weight: int = 400,
) -> str:
    """Generates a complete CSS stylesheet with @font-face rules for a SlicingStrategy.

    Each non-empty slice produces one @font-face declaration with a compact
    unicode-range descriptor.
    """
    ext = extension.lstrip(".")
    fmt_map = {
        "woff2": "woff2",
        "woff": "woff",
        "otf": "opentype",
        "ttf": "truetype",
    }
    css_format = fmt_map.get(ext.lower(), ext.lower())
    prefix = font_url_prefix.rstrip("/")
    if prefix:
        prefix = prefix + "/"

    blocks: List[str] = []
    for idx, subset in enumerate(strategy.subsets):
        if not subset:
            continue
        uranges = format_unicode_ranges(subset)
        urange_str = ", ".join(uranges)
        src_url = f"{prefix}{font_basename}_slice_{idx:03d}.{ext}"
        block = (
            f"/* [{idx}] ({len(subset)} glyphs, {len(uranges)} ranges) */\n"
            f"@font-face {{\n"
            f"  font-family: '{font_family}';\n"
            f"  font-style: {font_style};\n"
            f"  font-weight: {font_weight};\n"
            f"  src: url({src_url}) format('{css_format}');\n"
            f"  unicode-range: {urange_str};\n"
            f"}}"
        )
        blocks.append(block)

    return "\n\n".join(blocks) + "\n"


@dataclass
class FontSliceArtifact:
    """Metadata for a single physical font slice file."""

    slice_index: int
    num_codepoints: int
    num_unicode_ranges: int
    path: Path
    byte_size: int


def _subset_single_slice_worker(
    args: Tuple[str, int, List[int], str, Optional[str], str],
) -> FontSliceArtifact:
    """Worker function for slicing one subset from a font file."""
    src_path_str, idx, subset, out_dir_str, flavor, ext = args
    src_path = Path(src_path_str)
    out_dir = Path(out_dir_str)
    stem = src_path.stem

    options = ft_subset.Options()
    options.flavor = flavor
    options.NameIDs = [0, 1, 2, 3, 4, 5, 6]
    options.layout_features = ["*"]
    options.notdef_outline = True

    font = TTFont(str(src_path), lazy=True)
    subsetter = ft_subset.Subsetter(options=options)
    subsetter.populate(unicodes=subset)
    subsetter.subset(font)

    slice_path = out_dir / f"{stem}_slice_{idx:03d}.{ext}"
    font.save(str(slice_path))
    font.close()

    uranges = format_unicode_ranges(subset)
    return FontSliceArtifact(
        slice_index=idx,
        num_codepoints=len(subset),
        num_unicode_ranges=len(uranges),
        path=slice_path,
        byte_size=slice_path.stat().st_size,
    )


def slice_font_file(
    font_path: str | Path,
    strategy: SlicingStrategy,
    output_dir: str | Path,
    font_family: Optional[str] = None,
    flavor: Optional[str] = "woff2",
    max_slices: Optional[int] = None,
    num_workers: int = 1,
) -> Tuple[List[FontSliceArtifact], Path]:
    """Physically slices a .ttf/.otf font file according to a SlicingStrategy.

    Uses fontTools.subset to retain only the codepoints in each slice while
    preserving layout features and hinting, and writes both the sliced font
    files and a companion CSS stylesheet to output_dir.

    Args:
        font_path: Path to the source .ttf or .otf font.
        strategy: SlicingStrategy defining the codepoint subsets.
        output_dir: Directory where slice files and CSS will be saved.
        font_family: CSS font-family name (defaults to source font stem).
        flavor: Output compression flavor: 'woff2', 'woff', or None (keep
            source .ttf/.otf container format).
        max_slices: Optional cap on number of slices to physically write.
        num_workers: Number of parallel worker processes for font subsetting.

    Returns:
        Tuple of (list of FontSliceArtifact, path to generated .css file).
    """
    src_path = Path(font_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = src_path.stem
    family_name = font_family or f"{stem} Sliced"
    ext = flavor if flavor is not None else src_path.suffix.lstrip(".")

    subsets_to_process = (
        strategy.subsets[:max_slices] if max_slices is not None else strategy.subsets
    )
    tasks: List[Tuple[str, int, List[int], str, Optional[str], str]] = [
        (str(src_path), idx, subset, str(out_dir), flavor, ext)
        for idx, subset in enumerate(subsets_to_process)
        if subset
    ]

    if num_workers > 1 and len(tasks) > 1:
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as pool:
            artifacts = list(pool.map(_subset_single_slice_worker, tasks))
    else:
        artifacts = [_subset_single_slice_worker(t) for t in tasks]

    css_text = generate_css(
        strategy=strategy,
        font_family=family_name,
        font_basename=stem,
        extension=ext,
    )
    css_path = out_dir / f"{stem.lower()}_sliced.css"
    css_path.write_text(css_text, encoding="utf-8")

    return artifacts, css_path
