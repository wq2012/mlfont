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

# mlfont-notosansjp: Range-Compact MLFont Slicing for Noto Sans JP (Japanese)

**This is not an officially supported Google product.**

## Overview

Pretrained **MLFont** machine-learning-guided font slicing model and WOFF2 slice bundle for **Noto Sans JP (Japanese)** (`NotoSansJP-Regular.otf`, 16,665 Unicode codepoints partitioned into 62 range-compact slices).

* **Paper**: *Learning Better Font Slicing Strategies from Data* (Quan Wang and Yiran Mao)
* **Open-Source Repository**: https://github.com/wq2012/mlfont
* **PyPI Package**: https://pypi.org/project/mlfont/

## Training Configuration

* **Target Font**: `NotoSansJP-Regular.otf` (16,665 Unicode codepoints)
* **Training & Evaluation Corpus**: `wikimedia/wikipedia` (`20231101`, languages: `ja`), 1,600 training articles and 400 testing articles
* **Slicing Architecture**: Tiered Document-Frequency Slicing + Online Variational LDA Topic Modeling + Range-Compact Genetic Algorithm (`mlfont` preset: `japanese`)
* **Total Slices**: 62 slices (9,380 total CSS `unicode-range` intervals)
* **Physical WOFF2 Bundle Size**: 4732.7 KB across 62 `.woff2` files (companion CSS: 99.6 KB)
* **Training Wall-Time**: 73.2 seconds on CPU/GPU

## Model Files Included

* `model.safetensors`: Learned topic-codepoint weight matrix (`lambda_matrix`), `codepoints`, `frequency_array`, and `dna_assignment` in SafeTensors format
* `strategy.json`: Complete `SlicingStrategy` definition and hyperparameters
* `config.json`: Model and font bundle summary metadata
* `font_sliced.css`: Production-ready `@font-face` stylesheet with compact `unicode-range` descriptors
* `woff2/*.woff2`: 62 physical WOFF2 font slices generated via `fontTools.subset`
* `nam/*.nam`: 62 Google Fonts `.nam` codepoint subset files
* `evaluation_metrics.json`: Full benchmark metrics across all 7 network/session profiles

## Benchmark Performance (400 Wikipedia Test Pages, Session Length L = 10)

| Strategy | Slices | CSS Ranges | Missing Glyphs | GIN-2g Mean / P95 (ms) | GIN-3g Mean / P95 (ms) | DSL Mean / P95 (ms) | Session Util (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `single_slice` | 1 | 4831 | 0 | 82.38 / 642.61 | 20.63 / 160.89 | 4.12 / 32.11 | 7.05% |
| `equal_bin_62` | 62 | 4876 | 0 | 86.91 / 646.00 | 23.31 / 173.24 | 4.19 / 31.13 | 7.98% |
| `random_slice_62` | 62 | 16489 | 0 | 103.47 / 818.24 | 27.75 / 219.43 | 4.99 / 39.42 | 7.05% |
| `language_slice` | 12 | 5966 | 38 | 46.07 / 354.24 | 11.68 / 89.52 | 2.29 / 17.62 | 12.97% |
| `most_500_500_equal_bin_60` | 62 | 5982 | 0 | 73.54 / 382.26 | 19.71 / 102.34 | 3.54 / 18.43 | 8.52% |
| `online_lda_topics60_greedy` | 62 | 9521 | 0 | 32.23 / 188.16 | 8.61 / 50.09 | 1.56 / 9.09 | 19.92% |
| `dnntm_topics60_greedy` | 62 | 9524 | 0 | 30.14 / 147.36 | 8.05 / 39.33 | 1.46 / 7.12 | 21.68% |
| **range_compact_most500_500_topics60 (This Model)** | 62 | 9380 | 0 | 32.62 / 190.08 | 8.71 / 50.64 | 1.58 / 9.21 | 19.61% |

## Usage

Install `mlfont`:

```bash
pip install mlfont huggingface_hub
```

Load the pretrained bundle and analyze or slice a font in Python:

```python
from huggingface_hub import snapshot_download
import mlfont

model_dir = snapshot_download(repo_id="wq2012/mlfont-notosansjp")
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
