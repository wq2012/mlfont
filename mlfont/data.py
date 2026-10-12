"""Font codepoint extraction and text corpus processing utilities."""

from __future__ import annotations

from pathlib import Path
import random
from typing import Dict, Iterable, Iterator, List, Sequence, Set

from fontTools.ttLib import TTFont
import numpy as np
from scipy import sparse


def extract_codepoints_from_font(
    font_path: str | Path,
    exclude_control_chars: bool = True,
) -> List[int]:
    """Extracts sorted Unicode codepoints supported by a TTF/OTF/WOFF2 font."""
    with TTFont(str(font_path), lazy=True) as ttfont:
        cmap = ttfont.getBestCmap() or {}
        codepoints = set(int(cp) for cp in cmap.keys())
    if exclude_control_chars:
        codepoints = {cp for cp in codepoints if cp > 0x001F and cp != 0x007F}
    return sorted(codepoints)


def parse_codepoints_file(path: str | Path) -> Set[int]:
    """Reads a set of codepoints from a hex-formatted (.nam or .txt) file.

    Lines starting with '0x' or 'U+' are parsed as hexadecimal integers.
    """
    codepoints: Set[int] = set()
    for raw_line in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        token = line.split()[0]
        if token.lower().startswith("0x"):
            codepoints.add(int(token, 16))
        elif token.upper().startswith("U+"):
            codepoints.add(int(token[2:], 16))
    return codepoints


def save_codepoints_file(codepoints: Iterable[int], path: str | Path) -> None:
    """Writes sorted codepoints in '0xXXXX' hexadecimal format."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"0x{int(cp):04X}" for cp in sorted(set(codepoints))]
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fill_codepoint_gaps(codepoint_set: Set[int], fill_gap: int) -> Set[int]:
    """Fills gaps between contiguous codepoint ranges that are <= fill_gap."""
    if not isinstance(codepoint_set, set):
        raise TypeError("codepoint_set must be a set of integers")
    if not isinstance(fill_gap, int) or isinstance(fill_gap, bool):
        raise TypeError("fill_gap must be an integer")
    if fill_gap < 0:
        raise ValueError("fill_gap must be non-negative")
    if not codepoint_set or fill_gap == 0:
        return set(codepoint_set)

    sorted_cps = sorted(codepoint_set)
    ranges: List[tuple[int, int]] = []
    start = sorted_cps[0]
    prev = sorted_cps[0]
    for cp in sorted_cps[1:]:
        if cp - 1 != prev:
            ranges.append((start, prev))
            start = cp
        prev = cp
    ranges.append((start, prev))

    result = set(codepoint_set)
    for idx in range(len(ranges) - 1):
        gap = ranges[idx + 1][0] - ranges[idx][1] - 1
        if gap <= fill_gap:
            for cp in range(ranges[idx][1] + 1, ranges[idx + 1][0]):
                result.add(cp)
    return result


def _iter_doc_codepoints(
    doc: str | Iterable[int | str],
    deduplicate: bool = True,
) -> List[int]:
    if isinstance(doc, str):
        chars = set(doc) if deduplicate else doc
        return [ord(ch) for ch in chars]
    items = set(doc) if deduplicate else doc
    return [ord(ch) if isinstance(ch, str) else int(ch) for ch in items]


class DocumentCorpusProcessor:
    """Processes a collection of text documents for MLFont training & evaluation."""

    def __init__(
        self,
        documents: Sequence[str | Iterable[int | str]],
        random_seed: int = 42,
    ) -> None:
        if not documents:
            raise ValueError("documents must be non-empty")
        self._documents = list(documents)
        self._rng = random.Random(random_seed)

    @property
    def num_documents(self) -> int:
        return len(self._documents)

    def generate_codepoint_sets(self, valid_codepoints: Set[int]) -> Iterator[Set[int]]:
        """Yields the set of valid target codepoints appearing in each document."""
        for doc in self._documents:
            cp_set = set(_iter_doc_codepoints(doc, deduplicate=True)) & valid_codepoints
            if cp_set:
                yield cp_set

    def count_codepoint_frequency(
        self,
        valid_codepoints: Set[int],
        deduplicate_within_doc: bool = True,
    ) -> Dict[int, int]:
        """Counts how frequently each codepoint occurs across the corpus."""
        counts: Dict[int, int] = {int(cp): 0 for cp in valid_codepoints}
        for doc in self._documents:
            for cp in _iter_doc_codepoints(doc, deduplicate=deduplicate_within_doc):
                if cp in counts:
                    counts[cp] += 1
        return counts

    def count_frequency_array(
        self,
        codepoint_list: Sequence[int],
        deduplicate_within_doc: bool = True,
    ) -> np.ndarray:
        """Returns a 1D numpy array of frequencies aligned with codepoint_list."""
        cp_set = set(codepoint_list)
        freq_dict = self.count_codepoint_frequency(
            cp_set, deduplicate_within_doc=deduplicate_within_doc
        )
        return np.array([freq_dict[cp] for cp in codepoint_list], dtype=np.int64)

    def construct_batch_sparse_matrix(
        self,
        codepoint_list: Sequence[int],
        n_samples: int = 0,
    ) -> sparse.csr_matrix:
        """Constructs a binary document-codepoint sparse matrix (n_samples x n_codepoints).

        Args:
            codepoint_list: Ordered list of target codepoints (columns).
            n_samples: Number of non-empty documents to sample. If 0, uses all
                documents in the corpus that contain at least one valid codepoint.
        """
        if n_samples < 0:
            raise ValueError("n_samples cannot be negative")
        cp_to_col = {int(cp): idx for idx, cp in enumerate(codepoint_list)}
        n_cols = len(codepoint_list)

        rows: List[int] = []
        cols: List[int] = []
        if n_samples == 0:
            row_idx = 0
            for doc in self._documents:
                doc_cols = [
                    cp_to_col[cp]
                    for cp in _iter_doc_codepoints(doc, deduplicate=True)
                    if cp in cp_to_col
                ]
                if doc_cols:
                    rows.extend([row_idx] * len(doc_cols))
                    cols.extend(doc_cols)
                    row_idx += 1
            total_rows = row_idx
        else:
            row_idx = 0
            attempts = 0
            max_attempts = max(n_samples * 20, len(self._documents) * 2)
            while row_idx < n_samples and attempts < max_attempts:
                attempts += 1
                doc = self._rng.choice(self._documents)
                doc_cols = [
                    cp_to_col[cp]
                    for cp in _iter_doc_codepoints(doc, deduplicate=True)
                    if cp in cp_to_col
                ]
                if doc_cols:
                    rows.extend([row_idx] * len(doc_cols))
                    cols.extend(doc_cols)
                    row_idx += 1
            total_rows = row_idx

        data = np.ones(len(rows), dtype=np.uint8)
        return sparse.csr_matrix((data, (rows, cols)), shape=(total_rows, n_cols), dtype=np.uint8)
