---
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

# mlfont-notosanskr: Range-Compact MLFont Slicing for Noto Sans KR (Korean)

**This is not an officially supported Google product.**

## Overview

Pretrained **MLFont** machine-learning-guided font slicing model and WOFF2 slice bundle for **Noto Sans KR (Korean)** (`NotoSansKR-Regular.otf`, 12,254 Unicode codepoints partitioned into 52 range-compact slices).

* **Paper**: *Learning Better Font Slicing Strategies from Data* (Quan Wang and Yiran Mao)
* **Open-Source Repository**: https://github.com/wq2012/mlfont
* **PyPI Package**: https://pypi.org/project/mlfont/

## Training Configuration

* **Target Font**: `NotoSansKR-Regular.otf` (12,254 Unicode codepoints)
* **Training & Evaluation Corpus**: `wikimedia/wikipedia` (`20231101`, languages: `ko`), 1,600 training articles and 400 testing articles
* **Slicing Architecture**: Tiered Document-Frequency Slicing + Online Variational LDA Topic Modeling + Range-Compact Genetic Algorithm (`mlfont` preset: `korean`)
* **Total Slices**: 52 slices (3,022 total CSS `unicode-range` intervals)
* **Physical WOFF2 Bundle Size**: 2094.0 KB across 52 `.woff2` files (companion CSS: 40.5 KB)
* **Training Wall-Time**: 47.2 seconds on CPU/GPU

## Model Files Included

* `model.safetensors`: Learned topic-codepoint weight matrix (`lambda_matrix`), `codepoints`, `frequency_array`, and `dna_assignment` in SafeTensors format
* `strategy.json`: Complete `SlicingStrategy` definition and hyperparameters
* `config.json`: Model and font bundle summary metadata
* `font_sliced.css`: Production-ready `@font-face` stylesheet with compact `unicode-range` descriptors
* `woff2/*.woff2`: 52 physical WOFF2 font slices generated via `fontTools.subset`
* `nam/*.nam`: 52 Google Fonts `.nam` codepoint subset files
* `evaluation_metrics.json`: Full benchmark metrics across all 7 network/session profiles

## Benchmark Performance (400 Wikipedia Test Pages, Session Length L = 10)

| Strategy | Slices | CSS Ranges | Missing Glyphs | GIN-2g Mean / P95 (ms) | GIN-3g Mean / P95 (ms) | DSL Mean / P95 (ms) | Session Util (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `single_slice` | 1 | 90 | 0 | 271.84 / 2535.86 | 68.10 / 635.25 | 13.58 / 126.66 | 2.55% |
| `equal_bin_52` | 52 | 141 | 0 | 252.76 / 2014.68 | 68.28 / 544.20 | 12.13 / 96.68 | 2.88% |
| `random_slice_52` | 52 | 12039 | 0 | 311.02 / 2677.23 | 84.01 / 723.16 | 14.93 / 128.48 | 2.58% |
| `language_slice` | 9 | 1659 | 3 | 61.54 / 565.22 | 15.72 / 143.87 | 3.04 / 28.00 | 11.81% |
| `most_300_300_equal_bin_50` | 52 | 1114 | 0 | 72.80 / 319.47 | 19.62 / 85.96 | 3.50 / 15.39 | 7.52% |
| `online_lda_topics50_greedy` | 52 | 3092 | 0 | 38.76 / 248.79 | 10.42 / 66.67 | 1.87 / 11.99 | 16.14% |
| `dnntm_topics50_greedy` | 52 | 3108 | 0 | 31.58 / 229.96 | 8.47 / 61.57 | 1.52 / 11.09 | 19.68% |
| **range_compact_most300_300_topics50 (This Model)** | 52 | 3022 | 0 | 38.84 / 248.79 | 10.44 / 66.67 | 1.87 / 11.99 | 16.02% |

## Usage

Install `mlfont`:

```bash
pip install mlfont huggingface_hub
```

Load the pretrained bundle and analyze or slice a font in Python:

```python
from huggingface_hub import snapshot_download
import mlfont

model_dir = snapshot_download(repo_id="wq2012/mlfont-notosanskr")
bundle = mlfont.load_pretrained(model_dir)
print("Loaded strategy:", bundle.strategy.name, "slices:", bundle.strategy.num_slices)
print("CSS unicode-range count:", bundle.strategy.count_unicode_ranges())
```

## Citation

```bibtex
@article{wang2017learning,
  title={Learning Better Font Slicing Strategies from Data},
  author={Wang, Quan and Mao, Yiran},
  journal={Technical Disclosure Commons, Defensive Publications Series},
  volume={906},
  year={2017}
}
```
