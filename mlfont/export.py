"""Export and load pretrained MLFont models, strategies, and font slice bundles."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
from safetensors.numpy import load_file as load_safetensors
from safetensors.numpy import save_file as save_safetensors

from mlfont.schemas import SlicingStrategy, StrategyEvaluationResult
from mlfont.subsetter import FontSliceArtifact, generate_css, slice_font_file


@dataclass
class PretrainedMLFontBundle:
    """Loaded pretrained MLFont model and slicing strategy bundle."""

    model_dir: Path
    config: Dict[str, object]
    strategy: SlicingStrategy
    lambda_matrix: Optional[np.ndarray] = None
    dnntm_lambda_matrix: Optional[np.ndarray] = None
    codepoints: Optional[np.ndarray] = None
    remaining_codepoints: Optional[np.ndarray] = None
    frequency_array: Optional[np.ndarray] = None
    olda_strategy: Optional[SlicingStrategy] = None
    dnntm_strategy: Optional[SlicingStrategy] = None
    css_path: Optional[Path] = None

    def slice_font(
        self,
        font_path: str | Path,
        output_dir: str | Path,
        font_family: Optional[str] = None,
        flavor: Optional[str] = "woff2",
    ) -> tuple[List[FontSliceArtifact], Path]:
        """Slices any target TTF/OTF font using this pretrained strategy."""
        return slice_font_file(
            font_path=font_path,
            strategy=self.strategy,
            output_dir=output_dir,
            font_family=font_family,
            flavor=flavor,
        )

    def predict_required_slices(self, text: str) -> List[int]:
        """Returns the sorted list of slice indices required to render the given text."""
        text_cps = {ord(ch) for ch in text}
        required: List[int] = []
        for idx, subset in enumerate(self.strategy.subsets):
            if text_cps.intersection(subset):
                required.append(idx)
        return required


def save_safetensors_model(
    path: str | Path,
    lambda_matrix: np.ndarray,
    codepoints: Sequence[int],
    frequency_array: Optional[np.ndarray] = None,
    remaining_codepoints: Optional[Sequence[int]] = None,
    dnntm_lambda_matrix: Optional[np.ndarray] = None,
) -> Path:
    """Saves learned MLFont topic matrices and codepoint metadata to a SafeTensors file."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tensors: Dict[str, np.ndarray] = {
        "lambda_matrix": np.ascontiguousarray(lambda_matrix, dtype=np.float32),
        "codepoints": np.ascontiguousarray(codepoints, dtype=np.int32),
    }
    if remaining_codepoints is not None:
        tensors["remaining_codepoints"] = np.ascontiguousarray(
            remaining_codepoints, dtype=np.int32
        )
    if frequency_array is not None:
        tensors["frequency_array"] = np.ascontiguousarray(
            frequency_array, dtype=np.int64
        )
    if dnntm_lambda_matrix is not None:
        tensors["dnntm_lambda_matrix"] = np.ascontiguousarray(
            dnntm_lambda_matrix, dtype=np.float32
        )
    save_safetensors(tensors, str(out_path))
    return out_path


def export_pretrained_bundle(
    output_dir: str | Path,
    strategy: SlicingStrategy,
    lambda_matrix: Optional[np.ndarray] = None,
    codepoints: Optional[Sequence[int]] = None,
    remaining_codepoints: Optional[Sequence[int]] = None,
    frequency_array: Optional[np.ndarray] = None,
    dnntm_lambda_matrix: Optional[np.ndarray] = None,
    olda_strategy: Optional[SlicingStrategy] = None,
    dnntm_strategy: Optional[SlicingStrategy] = None,
    font_path: Optional[str | Path] = None,
    font_family: Optional[str] = None,
    slice_font: bool = True,
    max_physical_slices: Optional[int] = None,
    num_workers: int = 1,
    evaluation_results: Optional[Sequence[StrategyEvaluationResult]] = None,
    extra_config: Optional[Dict[str, object]] = None,
) -> Path:
    """Exports a complete Hugging Face-ready pretrained MLFont bundle.

    Writes:
        - model.safetensors: lambda_matrix, dnntm_lambda_matrix, codepoints,
          remaining_codepoints, frequency_array, and dna_assignment tensors.
        - strategy.json: Range-Compact GA SlicingStrategy definition and metadata.
        - olda_strategy.json: Online LDA + Greedy SlicingStrategy (if provided).
        - dnntm_strategy.json: DNNTM + Greedy SlicingStrategy (if provided).
        - config.json: Model hyperparameters and statistics.
        - nam/slice_XXX.nam: Google Fonts .nam files for each slice.
        - font_sliced.css: Ready-to-serve @font-face CSS with unicode-range.
        - woff2/*.woff2: Physical WOFF2 font slices (if font_path is provided and slice_font=True).
        - evaluation_metrics.json: Benchmark latency and utilization metrics (if provided).
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Save strategy.json, optional olda/dnntm strategies, and .nam files
    strategy.save_json(out / "strategy.json")
    if olda_strategy is not None:
        olda_strategy.save_json(out / "olda_strategy.json")
    if dnntm_strategy is not None:
        dnntm_strategy.save_json(out / "dnntm_strategy.json")
    strategy.save_nam_dir(out / "nam", prefix="slice")

    # 2. Save model.safetensors
    tensors: Dict[str, np.ndarray] = {}
    all_cps = (
        sorted(int(c) for c in codepoints)
        if codepoints is not None
        else sorted(strategy.all_codepoints())
    )
    tensors["codepoints"] = np.ascontiguousarray(all_cps, dtype=np.int32)

    if lambda_matrix is not None:
        tensors["lambda_matrix"] = np.ascontiguousarray(lambda_matrix, dtype=np.float32)
        if remaining_codepoints is not None:
            rem_cps = [int(c) for c in remaining_codepoints]
        elif lambda_matrix.shape[1] == len(all_cps):
            rem_cps = all_cps
        else:
            n_most = strategy.num_slices - lambda_matrix.shape[0]
            most_set: set[int] = set()
            for s in strategy.subsets[: max(0, n_most)]:
                most_set.update(int(c) for c in s)
            rem_cps = [c for c in all_cps if c not in most_set]
        if len(rem_cps) == lambda_matrix.shape[1]:
            tensors["remaining_codepoints"] = np.ascontiguousarray(
                rem_cps, dtype=np.int32
            )

    if dnntm_lambda_matrix is not None:
        tensors["dnntm_lambda_matrix"] = np.ascontiguousarray(
            dnntm_lambda_matrix, dtype=np.float32
        )

    if frequency_array is not None:
        tensors["frequency_array"] = np.ascontiguousarray(
            frequency_array, dtype=np.int64
        )

    dna = strategy.to_dna()
    tensors["dna_assignment"] = np.ascontiguousarray(dna, dtype=np.int32)
    save_safetensors(tensors, str(out / "model.safetensors"))

    # 3. Physically slice font or generate CSS
    family = font_family or strategy.name or "MLFont Sliced"
    css_path = out / "font_sliced.css"
    total_woff2_bytes = 0
    num_written_woff2 = 0
    woff2_slice_byte_sizes: List[int] = []
    if font_path is not None and slice_font:
        woff2_dir = out / "woff2"
        artifacts, gen_css_path = slice_font_file(
            font_path=font_path,
            strategy=strategy,
            output_dir=woff2_dir,
            font_family=family,
            flavor="woff2",
            max_slices=max_physical_slices,
            num_workers=num_workers,
        )
        css_path.write_text(gen_css_path.read_text(encoding="utf-8"), encoding="utf-8")
        woff2_slice_byte_sizes = [a.byte_size for a in artifacts]
        total_woff2_bytes = sum(woff2_slice_byte_sizes)
        num_written_woff2 = len(artifacts)
    else:
        css_text = generate_css(
            strategy=strategy,
            font_family=family,
            font_basename="font",
            extension="woff2",
        )
        css_path.write_text(css_text, encoding="utf-8")

    # 4. Save config.json
    config_payload: Dict[str, object] = {
        "strategy_name": strategy.name,
        "font_family": family,
        "num_slices": strategy.num_slices,
        "num_codepoints": strategy.num_codepoints,
        "num_unicode_ranges": strategy.count_unicode_ranges(),
        "css_byte_size": css_path.stat().st_size,
        "num_woff2_slices_written": num_written_woff2,
        "total_woff2_bytes": total_woff2_bytes,
        "woff2_slice_byte_sizes": woff2_slice_byte_sizes,
        "strategy_metadata": strategy.metadata,
    }
    if extra_config:
        config_payload.update(extra_config)
    (out / "config.json").write_text(
        json.dumps(config_payload, indent=2), encoding="utf-8"
    )

    # 5. Save evaluation_metrics.json if provided
    if evaluation_results is not None:
        eval_payload = [asdict(r) for r in evaluation_results]
        (out / "evaluation_metrics.json").write_text(
            json.dumps(eval_payload, indent=2), encoding="utf-8"
        )

    return out


def load_pretrained(model_dir: str | Path) -> PretrainedMLFontBundle:
    """Loads a pretrained MLFont bundle from a local directory or downloaded HF repo."""
    root = Path(model_dir)
    strategy = SlicingStrategy.load_json(root / "strategy.json")
    olda_strategy: Optional[SlicingStrategy] = None
    if (root / "olda_strategy.json").exists():
        olda_strategy = SlicingStrategy.load_json(root / "olda_strategy.json")
    dnntm_strategy: Optional[SlicingStrategy] = None
    if (root / "dnntm_strategy.json").exists():
        dnntm_strategy = SlicingStrategy.load_json(root / "dnntm_strategy.json")

    config: Dict[str, object] = {}
    if (root / "config.json").exists():
        config = json.loads((root / "config.json").read_text(encoding="utf-8"))

    lambda_mat: Optional[np.ndarray] = None
    dnntm_lambda_mat: Optional[np.ndarray] = None
    cps: Optional[np.ndarray] = None
    rem_cps: Optional[np.ndarray] = None
    freq_arr: Optional[np.ndarray] = None
    st_path = root / "model.safetensors"
    if st_path.exists():
        tensors = load_safetensors(str(st_path))
        lambda_mat = tensors.get("lambda_matrix")
        dnntm_lambda_mat = tensors.get("dnntm_lambda_matrix")
        cps = tensors.get("codepoints")
        rem_cps = tensors.get("remaining_codepoints")
        freq_arr = tensors.get("frequency_array")

    css_path = root / "font_sliced.css"
    return PretrainedMLFontBundle(
        model_dir=root,
        config=config,
        strategy=strategy,
        lambda_matrix=lambda_mat,
        dnntm_lambda_matrix=dnntm_lambda_mat,
        codepoints=cps,
        remaining_codepoints=rem_cps,
        frequency_array=freq_arr,
        olda_strategy=olda_strategy,
        dnntm_strategy=dnntm_strategy,
        css_path=css_path if css_path.exists() else None,
    )
