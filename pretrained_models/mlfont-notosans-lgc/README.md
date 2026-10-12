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

# mlfont-notosans-lgc: Range-Compact MLFont Slicing for Noto Sans (Pan-European LGC)

**This is not an officially supported Google product.**

## Overview

Pretrained **MLFont** machine-learning-guided font slicing model and WOFF2 slice bundle for **Noto Sans (Pan-European LGC)** (`NotoSans-Regular.ttf`, 2,297 Unicode codepoints partitioned into 20 range-compact slices).

* **Paper**: *Learning Better Font Slicing Strategies from Data* (Quan Wang and Yiran Mao)
* **Open-Source Repository**: https://github.com/wq2012/mlfont
* **PyPI Package**: https://pypi.org/project/mlfont/

## Training Configuration

* **Target Font**: `NotoSans-Regular.ttf` (2,297 Unicode codepoints)
* **Training & Evaluation Corpus**: `wikimedia/wikipedia` (`20231101`, languages: `en, de, fr, pl, ru, el`), 1,500 training articles and 300 testing articles
* **Slicing Architecture**: Tiered Document-Frequency Slicing + Online Variational LDA Topic Modeling + Range-Compact Genetic Algorithm (`mlfont` preset: `lgc`)
* **Total Slices**: 20 slices (411 total CSS `unicode-range` intervals)
* **Physical WOFF2 Bundle Size**: 437.6 KB across 20 `.woff2` files (companion CSS: 8.1 KB)
* **Training Wall-Time**: 37.0 seconds on CPU/GPU

## Model Files Included

* `model.safetensors`: Learned topic-codepoint weight matrix (`lambda_matrix`), `codepoints`, `frequency_array`, and `dna_assignment` in SafeTensors format
* `strategy.json`: Complete `SlicingStrategy` definition and hyperparameters
* `config.json`: Model and font bundle summary metadata
* `font_sliced.css`: Production-ready `@font-face` stylesheet with compact `unicode-range` descriptors
* `woff2/*.woff2`: 20 physical WOFF2 font slices generated via `fontTools.subset`
* `nam/*.nam`: 20 Google Fonts `.nam` codepoint subset files
* `evaluation_metrics.json`: Full benchmark metrics across all 7 network/session profiles

## Benchmark Performance (300 Wikipedia Test Pages, Session Length L = 10)

| Strategy | Slices | CSS Ranges | Missing Glyphs | GIN-2g Mean / P95 (ms) | GIN-3g Mean / P95 (ms) | DSL Mean / P95 (ms) | Session Util (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `single_slice` | 1 | 123 | 0 | 58.52 / 542.17 | 14.78 / 136.97 | 2.91 / 26.97 | 4.63% |
| `equal_bin_20` | 20 | 140 | 0 | 22.55 / 157.92 | 6.38 / 44.68 | 1.05 / 7.38 | 20.11% |
| `random_slice_20` | 20 | 2180 | 0 | 92.96 / 849.22 | 26.30 / 240.26 | 4.34 / 39.67 | 4.63% |
| `language_slice` | 12 | 100 | 8 | 22.99 / 140.49 | 6.25 / 36.78 | 1.10 / 6.88 | 18.43% |
| `online_lda_topics20_greedy` | 20 | 516 | 0 | 20.15 / 98.53 | 5.70 / 27.87 | 0.94 / 4.60 | 21.26% |
| `dnntm_topics20_greedy` | 20 | 584 | 0 | 18.00 / 118.44 | 5.09 / 33.51 | 0.84 / 5.53 | 24.30% |
| **range_compact_topics20 (This Model)** | 20 | 411 | 0 | 19.95 / 101.93 | 5.62 / 28.72 | 0.93 / 4.77 | 21.17% |

## Usage

Install `mlfont`:

```bash
pip install mlfont huggingface_hub
```

Load the pretrained bundle and analyze or slice a font in Python:

```python
from huggingface_hub import snapshot_download
import mlfont

model_dir = snapshot_download(repo_id="wq2012/mlfont-notosans-lgc")
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
