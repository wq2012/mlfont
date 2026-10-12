#!/usr/bin/env python3
"""End-to-end Noto font benchmark runner and Hugging Face bundle exporter."""

from __future__ import annotations

from dataclasses import asdict
import glob
import json
import os
from pathlib import Path
import time
from typing import Dict, List, Sequence, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from datasets import load_dataset  # noqa: E402
import numpy as np  # noqa: E402

from mlfont import (
    DocumentCorpusProcessor,
    FontLoadingEvaluator,
    OnlineLdaConfig,
    OnlineLdaLearner,
    RangeCompactGeneticLearner,
    STANDARD_EVALUATION_PARAMS,
    SlicingStrategy,
    StrategyEvaluationResult,
    DnnTmLearner,
    export_pretrained_bundle,
    extract_codepoints_from_font,
    generate_equal_bin_strategy,
    generate_greedy_topic_strategy,
    generate_language_slice_strategy,
    generate_most_and_equal_bin_strategy,
    generate_random_slice_strategy,
    generate_single_slice_strategy,
    get_preset_config,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = Path(os.environ.get("MLFONT_ASSETS_DIR", "/tmp/mlfont_noto_assets"))
NAM_DIR = ASSETS_DIR / "nam"
FONTS_DIR = ASSETS_DIR / "fonts"

FONT_SPECS = [
    {
        "key": "notosans_lgc",
        "hf_name": "mlfont-notosans-lgc",
        "title": "Noto Sans (Pan-European LGC)",
        "preset": "lgc",
        "font_file": "NotoSans-Regular.ttf",
        "wiki_langs": ["en", "de", "fr", "pl", "ru", "el"],
        "n_train": 1500,
        "n_test": 300,
    },
    {
        "key": "notosanskr",
        "hf_name": "mlfont-notosanskr",
        "title": "Noto Sans KR (Korean)",
        "preset": "korean",
        "font_file": "NotoSansKR-Regular.otf",
        "wiki_langs": ["ko"],
        "n_train": 1600,
        "n_test": 400,
    },
    {
        "key": "notosansjp",
        "hf_name": "mlfont-notosansjp",
        "title": "Noto Sans JP (Japanese)",
        "preset": "japanese",
        "font_file": "NotoSansJP-Regular.otf",
        "wiki_langs": ["ja"],
        "n_train": 1600,
        "n_test": 400,
    },
    {
        "key": "notosanstc",
        "hf_name": "mlfont-notosanstc",
        "title": "Noto Sans TC (Traditional Chinese)",
        "preset": "traditional_chinese",
        "font_file": "NotoSansTC-Regular.otf",
        "wiki_langs": ["zh"],
        "script_filter": "traditional",
        "n_train": 1600,
        "n_test": 400,
    },
    {
        "key": "notosanssc",
        "hf_name": "mlfont-notosanssc",
        "title": "Noto Sans SC (Simplified Chinese)",
        "preset": "chinese",
        "font_file": "NotoSansSC-Regular.otf",
        "wiki_langs": ["zh"],
        "script_filter": "simplified",
        "n_train": 1600,
        "n_test": 400,
    },
]

TRAD_MARKERS = set("國學體會語發電書車長門開關東風馬鳥魚區與專業時間應當後實經")
SIMP_MARKERS = set("国学体会语发电书车长门开关东风马鸟鱼区与专业时间应当后实经")


def fetch_wikipedia_corpus(
    langs: Sequence[str],
    n_train: int,
    n_test: int,
    valid_cps: set[int],
    script_filter: str = "",
) -> Tuple[List[str], List[str]]:
    """Streams real Wikipedia articles across one or more language editions."""
    total_needed = n_train + n_test
    per_lang_train = max(1, n_train // len(langs))
    per_lang_test = max(1, n_test // len(langs))

    train_docs: List[str] = []
    test_docs: List[str] = []

    for lang in langs:
        need = per_lang_train + per_lang_test
        if lang == langs[-1]:
            need = total_needed - len(train_docs) - len(test_docs)
        ds = load_dataset(
            "wikimedia/wikipedia",
            f"20231101.{lang}",
            split="train",
            streaming=True,
        )
        lang_docs: List[str] = []
        for item in ds:
            text = (item.get("text") or "").strip()
            if len(text) < 100:
                continue
            snippet = text[:2500]
            if script_filter == "traditional":
                n_t = sum(1 for ch in snippet if ch in TRAD_MARKERS)
                n_s = sum(1 for ch in snippet if ch in SIMP_MARKERS)
                if n_t <= n_s:
                    continue
            elif script_filter == "simplified":
                n_t = sum(1 for ch in snippet if ch in TRAD_MARKERS)
                n_s = sum(1 for ch in snippet if ch in SIMP_MARKERS)
                if n_s < n_t:
                    continue
            matched = {ord(ch) for ch in snippet} & valid_cps
            if len(matched) >= 15:
                lang_docs.append(snippet)
                if len(lang_docs) >= need:
                    break
        split_pt = min(len(lang_docs), per_lang_train)
        train_docs.extend(lang_docs[:split_pt])
        test_docs.extend(lang_docs[split_pt:])

    return train_docs[:n_train], test_docs[:n_test]


def write_hf_model_card(
    bundle_dir: Path,
    spec: Dict[str, object],
    strategy: SlicingStrategy,
    eval_results: Sequence[StrategyEvaluationResult],
    all_strat_results: Sequence[StrategyEvaluationResult],
    total_woff2_bytes: int,
    css_bytes: int,
    wall_time_sec: float,
) -> None:
    """Writes a complete Hugging Face Model Card README.md without raw $ math."""
    hf_name = str(spec["hf_name"])
    title = str(spec["title"])
    n_train = int(spec["n_train"])
    n_test = int(spec["n_test"])
    langs_str = ", ".join(spec["wiki_langs"])

    # Extract GIN-3g and DSL rows for all strategies
    by_strat: Dict[str, Dict[str, StrategyEvaluationResult]] = {}
    for r in all_strat_results:
        by_strat.setdefault(r.strategy_name, {})[r.params_name] = r

    table_lines = [
        "| Strategy | Slices | CSS Ranges | Missing Glyphs | GIN-2g Mean / P95 (ms) | GIN-3g Mean / P95 (ms) | DSL Mean / P95 (ms) | Session Util (%) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for s_name, prof_map in by_strat.items():
        r3g = prof_map.get("GIN-3g")
        r2g = prof_map.get("GIN-2g")
        rdsl = prof_map.get("residential-DSL")
        if not (r3g and r2g and rdsl):
            continue
        is_main = s_name == strategy.name
        label = f"**{s_name} (This Model)**" if is_main else f"`{s_name}`"
        table_lines.append(
            f"| {label} | {r3g.num_slices} | {r3g.num_ranges} | {r3g.num_missing_codepoints} | "
            f"{r2g.total_per_glyph_ms.mean:.2f} / {r2g.total_per_glyph_ms.p95:.2f} | "
            f"{r3g.total_per_glyph_ms.mean:.2f} / {r3g.total_per_glyph_ms.p95:.2f} | "
            f"{rdsl.total_per_glyph_ms.mean:.2f} / {rdsl.total_per_glyph_ms.p95:.2f} | "
            f"{r3g.session_utilization_pct:.2f}% |"
        )
    metrics_table = "\n".join(table_lines)

    card = f"""---
license: apache-2.0
library_name: mlfont
tags:
- fonts
- web-fonts
- topic-modeling
- genetic-algorithm
- unicode-range
- woff2
datasets:
- wikimedia/wikipedia
---

# {hf_name}: Range-Compact MLFont Slicing for {title}

**This is not an officially supported Google product.**

## Overview

Pretrained **MLFont** machine-learning-guided font slicing model and WOFF2 slice bundle for **{title}** (`{spec["font_file"]}`, {strategy.num_codepoints:,} Unicode codepoints partitioned into {strategy.num_slices} range-compact slices).

* **Paper**: *Learning Better Font Slicing Strategies from Data* (Quan Wang and Yiran Mao)
* **Open-Source Repository**: https://github.com/wq2012/mlfont
* **PyPI Package**: https://pypi.org/project/mlfont/

## Training Configuration

* **Target Font**: `{spec["font_file"]}` ({strategy.num_codepoints:,} Unicode codepoints)
* **Training & Evaluation Corpus**: `wikimedia/wikipedia` (`20231101`, languages: `{langs_str}`), {n_train:,} training articles and {n_test:,} testing articles
* **Slicing Architecture**: Tiered Document-Frequency Slicing + Online Variational LDA Topic Modeling + Range-Compact Genetic Algorithm (`mlfont` preset: `{spec["preset"]}`)
* **Total Slices**: {strategy.num_slices} slices ({strategy.count_unicode_ranges():,} total CSS `unicode-range` intervals)
* **Physical WOFF2 Bundle Size**: {total_woff2_bytes / 1024.0:.1f} KB across {strategy.num_slices} `.woff2` files (companion CSS: {css_bytes / 1024.0:.1f} KB)
* **Training Wall-Time**: {wall_time_sec:.1f} seconds on CPU/GPU

## Model Files Included

* `model.safetensors`: Learned topic-codepoint weight matrix (`lambda_matrix`), `codepoints`, `frequency_array`, and `dna_assignment` in SafeTensors format
* `strategy.json`: Complete `SlicingStrategy` definition and hyperparameters
* `config.json`: Model and font bundle summary metadata
* `font_sliced.css`: Production-ready `@font-face` stylesheet with compact `unicode-range` descriptors
* `woff2/*.woff2`: {strategy.num_slices} physical WOFF2 font slices generated via `fontTools.subset`
* `nam/*.nam`: {strategy.num_slices} Google Fonts `.nam` codepoint subset files
* `evaluation_metrics.json`: Full benchmark metrics across all 7 network/session profiles

## Benchmark Performance ({n_test:,} Wikipedia Test Pages, Session Length L = 10)

{metrics_table}

## Usage

Install `mlfont`:

```bash
pip install mlfont huggingface_hub
```

Load the pretrained bundle and analyze or slice a font in Python:

```python
from huggingface_hub import snapshot_download
import mlfont

model_dir = snapshot_download(repo_id="wq2012/{hf_name}")
bundle = mlfont.load_pretrained(model_dir)
print("Loaded strategy:", bundle.strategy.name, "slices:", bundle.strategy.num_slices)
print("CSS unicode-range count:", bundle.strategy.count_unicode_ranges())
```

## Citation

```bibtex
@article{{wang2017learning,
  title={{Learning Better Font Slicing Strategies from Data}},
  author={{Wang, Quan and Mao, Yiran}},
  journal={{Technical Disclosure Commons, Defensive Publications Series}},
  volume={{906}},
  year={{2017}}
}}
```
"""
    (bundle_dir / "README.md").write_text(card, encoding="utf-8")


def main() -> None:
    nam_paths = sorted(glob.glob(str(NAM_DIR / "*.nam")))
    experiments_dir = REPO_ROOT / "experiments"
    pretrained_root = REPO_ROOT / "pretrained_models"
    experiments_dir.mkdir(parents=True, exist_ok=True)
    pretrained_root.mkdir(parents=True, exist_ok=True)

    summary_report: Dict[str, object] = {"fonts": {}, "ablations": {}}

    for spec in FONT_SPECS:
        key = str(spec["key"])
        hf_name = str(spec["hf_name"])
        font_path = FONTS_DIR / str(spec["font_file"])
        preset = get_preset_config(str(spec["preset"]))
        t_start = time.time()

        print(f"\n=== Processing {spec['title']} ({font_path.name}) ===")
        codepoints = extract_codepoints_from_font(font_path)
        cp_set = set(codepoints)
        print(f"  Extracted {len(codepoints)} codepoints from {font_path.name}")

        train_docs, test_docs = fetch_wikipedia_corpus(
            spec["wiki_langs"],
            n_train=int(spec["n_train"]),
            n_test=int(spec["n_test"]),
            valid_cps=cp_set,
            script_filter=str(spec.get("script_filter", "")),
        )
        print(f"  Loaded {len(train_docs)} train docs and {len(test_docs)} test docs")

        train_proc = DocumentCorpusProcessor(train_docs, random_seed=42)
        test_proc = DocumentCorpusProcessor(test_docs, random_seed=42)
        test_samples = list(test_proc.generate_codepoint_sets(cp_set))
        freq_arr = train_proc.count_frequency_array(codepoints, deduplicate_within_doc=True)

        total_slices = len(preset.olda_config.most_sizes) + preset.olda_config.n_topics

        # 1. Baselines
        s_single = generate_single_slice_strategy(codepoints, name="single_slice")
        s_eq = generate_equal_bin_strategy(
            codepoints, num_slices=total_slices, name=f"equal_bin_{total_slices}"
        )
        s_rand = generate_random_slice_strategy(
            codepoints, num_slices=total_slices, seed=42, name=f"random_slice_{total_slices}"
        )
        s_lang = generate_language_slice_strategy(
            codepoints, nam_paths, include_unassigned=False, name="language_slice"
        )
        strategies: List[SlicingStrategy] = [s_single, s_eq, s_rand, s_lang]

        if preset.olda_config.most_sizes:
            s_most_eq = generate_most_and_equal_bin_strategy(
                codepoints,
                frequency_array=freq_arr,
                most_sizes=preset.olda_config.most_sizes,
                num_bins=preset.olda_config.n_topics,
            )
            strategies.append(s_most_eq)

        # 2. Online LDA
        print("  Training Online LDA...")
        olda_learner = OnlineLdaLearner(train_proc, codepoints, preset.olda_config)
        s_olda = olda_learner.fit()
        s_olda.name = f"online_lda_topics{preset.olda_config.n_topics}_greedy"
        strategies.append(s_olda)

        # 3. DNNTM
        print("  Training DNNTM (Entropy Difference Loss)...")
        dnntm_learner = DnnTmLearner(train_proc, codepoints, preset.dnntm_config)
        s_dnntm = dnntm_learner.fit()
        s_dnntm.name = f"dnntm_topics{preset.dnntm_config.n_topics}_greedy"
        strategies.append(s_dnntm)

        # 4. Range-Compact GA on Online LDA lambda_matrix
        print("  Running Range-Compact Genetic Algorithm...")
        assert olda_learner.lambda_matrix_ is not None
        ga_learner = RangeCompactGeneticLearner(
            codepoint_list=codepoints,
            lambda_matrix=olda_learner.lambda_matrix_,
            config=preset.range_compact_config,
            frequency_array=freq_arr,
            most_sizes=preset.olda_config.most_sizes,
            normalization_method=preset.olda_config.normalization_method,
        )
        s_ga = ga_learner.run()
        strategies.append(s_ga)
        wall_time = time.time() - t_start

        # 5. Evaluate all strategies across all 7 network profiles
        print("  Evaluating all strategies across 7 network/session profiles...")
        evaluator = FontLoadingEvaluator(codepoints, test_samples)
        all_results = evaluator.evaluate_all(strategies, STANDARD_EVALUATION_PARAMS)
        ga_results = [r for r in all_results if r.strategy_name == s_ga.name]

        exp_out = experiments_dir / key
        exp_out.mkdir(parents=True, exist_ok=True)
        (exp_out / "evaluation_results.json").write_text(
            json.dumps([asdict(r) for r in all_results], indent=2), encoding="utf-8"
        )
        (exp_out / "ga_history.json").write_text(
            json.dumps([asdict(e) for e in ga_learner.history_], indent=2),
            encoding="utf-8",
        )

        # 6. Export Hugging Face Pretrained Bundle with physical WOFF2 font slices!
        print(f"  Exporting Hugging Face bundle & slicing {font_path.name} into WOFF2...")
        bundle_dir = pretrained_root / hf_name
        export_pretrained_bundle(
            output_dir=bundle_dir,
            strategy=s_ga,
            lambda_matrix=olda_learner.lambda_matrix_,
            codepoints=olda_learner.remaining_codepoints_ or codepoints,
            frequency_array=freq_arr,
            font_path=font_path,
            font_family=f"{font_path.stem} Sliced",
            slice_font=True,
            num_workers=32,
            evaluation_results=all_results,
            extra_config={
                "preset": spec["preset"],
                "wiki_langs": spec["wiki_langs"],
                "n_train_docs": len(train_docs),
                "n_test_docs": len(test_docs),
                "training_wall_time_sec": round(wall_time, 2),
            },
        )
        cfg_json = json.loads((bundle_dir / "config.json").read_text(encoding="utf-8"))
        write_hf_model_card(
            bundle_dir=bundle_dir,
            spec=spec,
            strategy=s_ga,
            eval_results=ga_results,
            all_strat_results=all_results,
            total_woff2_bytes=int(cfg_json["total_woff2_bytes"]),
            css_bytes=int(cfg_json["css_byte_size"]),
            wall_time_sec=wall_time,
        )

        # Print concise summary for GIN-3g
        for r in all_results:
            if r.params_name == "GIN-3g":
                print(
                    f"    {r.strategy_name:38s} | slices={r.num_slices:3d} | "
                    f"ranges={r.num_ranges:5d} | GIN-3g={r.total_per_glyph_ms.mean:6.2f} ms "
                    f"(P95={r.total_per_glyph_ms.p95:6.2f} ms) | util={r.session_utilization_pct:5.2f}%"
                )

        summary_report["fonts"][key] = {
            "title": spec["title"],
            "hf_name": hf_name,
            "num_codepoints": len(codepoints),
            "num_train_docs": len(train_docs),
            "num_test_samples": len(test_samples),
            "total_woff2_bytes": cfg_json["total_woff2_bytes"],
            "css_byte_size": cfg_json["css_byte_size"],
            "results": [asdict(r) for r in all_results],
        }

        # 7. Run Normalization & Slice Count Ablations on LGC
        if key == "notosans_lgc":
            print("  Running Lambda-Matrix Normalization & Slice Count K Ablations on LGC...")
            norm_ablation = []
            for norm_method in ("topic_sum", "row_sum", "global_max", "spectral_norm"):
                s_norm = generate_greedy_topic_strategy(
                    codepoints,
                    olda_learner.lambda_matrix_,
                    normalization_method=norm_method,
                    name=f"olda_norm_{norm_method}",
                )
                res_3g = evaluator.evaluate_strategy(s_norm, STANDARD_EVALUATION_PARAMS[2])
                res_2g = evaluator.evaluate_strategy(s_norm, STANDARD_EVALUATION_PARAMS[1])
                norm_ablation.append(
                    {
                        "normalization_method": norm_method,
                        "num_ranges": res_3g.num_ranges,
                        "gin_2g_mean_ms": res_2g.total_per_glyph_ms.mean,
                        "gin_2g_p95_ms": res_2g.total_per_glyph_ms.p95,
                        "gin_3g_mean_ms": res_3g.total_per_glyph_ms.mean,
                        "gin_3g_p95_ms": res_3g.total_per_glyph_ms.p95,
                        "session_util_pct": res_3g.session_utilization_pct,
                    }
                )
            summary_report["ablations"]["normalization_lgc"] = norm_ablation

            k_ablation = []
            for k_val in (5, 10, 20, 40, 60):
                k_learner = OnlineLdaLearner(
                    train_proc,
                    codepoints,
                    OnlineLdaConfig(
                        n_topics=k_val,
                        batch_size=1000,
                        n_batches=15,
                        random_state=42,
                    ),
                )
                s_k = k_learner.fit()
                r3g = evaluator.evaluate_strategy(s_k, STANDARD_EVALUATION_PARAMS[2])
                r2g = evaluator.evaluate_strategy(s_k, STANDARD_EVALUATION_PARAMS[1])
                rdsl = evaluator.evaluate_strategy(s_k, STANDARD_EVALUATION_PARAMS[3])
                k_ablation.append(
                    {
                        "n_topics": k_val,
                        "num_ranges": r3g.num_ranges,
                        "gin_2g_mean_ms": r2g.total_per_glyph_ms.mean,
                        "gin_3g_mean_ms": r3g.total_per_glyph_ms.mean,
                        "dsl_mean_ms": rdsl.total_per_glyph_ms.mean,
                        "session_util_pct": r3g.session_utilization_pct,
                    }
                )
            summary_report["ablations"]["slice_count_lgc"] = k_ablation

    (experiments_dir / "summary_report.json").write_text(
        json.dumps(summary_report, indent=2), encoding="utf-8"
    )
    print("\nAll 5 Noto font benchmarks and Hugging Face bundles completed successfully!")
    os._exit(0)


if __name__ == "__main__":
    main()
