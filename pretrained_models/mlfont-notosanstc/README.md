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

# mlfont-notosanstc: Range-Compact MLFont Slicing for Noto Sans TC (Traditional Chinese)

**This is not an officially supported Google product.**

## Overview

Pretrained **MLFont** machine-learning-guided font slicing model and WOFF2 slice bundle for **Noto Sans TC (Traditional Chinese)** (`NotoSansTC-Regular.otf`, 20,637 Unicode codepoints partitioned into 82 range-compact slices).

* **Paper**: *Learning Better Font Slicing Strategies from Data* (Quan Wang and Yiran Mao)
* **Open-Source Repository**: https://github.com/wq2012/mlfont
* **PyPI Package**: https://pypi.org/project/mlfont/

## Training Configuration

* **Target Font**: `NotoSansTC-Regular.otf` (20,637 Unicode codepoints)
* **Training & Evaluation Corpus**: `wikimedia/wikipedia` (`20231101`, languages: `zh`), 1,600 training articles and 400 testing articles
* **Slicing Architecture**: Tiered Document-Frequency Slicing + Online Variational LDA Topic Modeling + Range-Compact Genetic Algorithm (`mlfont` preset: `traditional_chinese`)
* **Total Slices**: 82 slices (12,787 total CSS `unicode-range` intervals)
* **Physical WOFF2 Bundle Size**: 6182.1 KB across 82 `.woff2` files (companion CSS: 133.4 KB)
* **Training Wall-Time**: 102.5 seconds on CPU/GPU

## Model Files Included

* `model.safetensors`: Learned topic-codepoint weight matrix (`lambda_matrix`), `codepoints`, `frequency_array`, and `dna_assignment` in SafeTensors format
* `strategy.json`: Complete `SlicingStrategy` definition and hyperparameters
* `config.json`: Model and font bundle summary metadata
* `font_sliced.css`: Production-ready `@font-face` stylesheet with compact `unicode-range` descriptors
* `woff2/*.woff2`: 82 physical WOFF2 font slices generated via `fontTools.subset`
* `nam/*.nam`: 82 Google Fonts `.nam` codepoint subset files
* `evaluation_metrics.json`: Full benchmark metrics across all 7 network/session profiles

## Benchmark Performance (400 Wikipedia Test Pages, Session Length L = 10)

| Strategy | Slices | CSS Ranges | Missing Glyphs | GIN-2g Mean / P95 (ms) | GIN-3g Mean / P95 (ms) | DSL Mean / P95 (ms) | Session Util (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `single_slice` | 1 | 5036 | 0 | 237.44 / 2367.09 | 59.43 / 592.49 | 11.86 / 118.28 | 3.84% |
| `equal_bin_82` | 82 | 5098 | 0 | 237.60 / 1807.68 | 63.95 / 486.50 | 11.43 / 86.93 | 4.81% |
| `random_slice_82` | 82 | 20436 | 0 | 301.24 / 2497.94 | 81.07 / 672.28 | 14.49 / 120.12 | 3.85% |
| `language_slice` | 12 | 5970 | 114 | 107.79 / 1015.13 | 27.26 / 256.58 | 5.36 / 50.53 | 8.59% |
| `most_600_600_equal_bin_80` | 82 | 6659 | 0 | 183.21 / 719.66 | 49.24 / 192.33 | 8.82 / 34.74 | 5.50% |
| `online_lda_topics80_greedy` | 82 | 13041 | 0 | 107.34 / 543.65 | 28.78 / 144.97 | 5.17 / 26.28 | 9.46% |
| `dnntm_topics80_greedy` | 82 | 13116 | 0 | 89.56 / 510.59 | 23.99 / 136.35 | 4.32 / 24.66 | 11.34% |
| **range_compact_most600_600_topics80 (This Model)** | 82 | 12787 | 0 | 109.34 / 552.75 | 29.29 / 147.25 | 5.27 / 26.73 | 9.27% |

## Usage

Install `mlfont`:

```bash
pip install mlfont huggingface_hub
```

Load the pretrained bundle and analyze or slice a font in Python:

```python
from huggingface_hub import snapshot_download
import mlfont

model_dir = snapshot_download(repo_id="wq2012/mlfont-notosanstc")
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
