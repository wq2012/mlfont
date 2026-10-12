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

# mlfont-notosanssc: Range-Compact MLFont Slicing for Noto Sans SC (Simplified Chinese)

**This is not an officially supported Google product.**

## Overview

Pretrained **MLFont** machine-learning-guided font slicing model and WOFF2 slice bundle for **Noto Sans SC (Simplified Chinese)** (`NotoSansSC-Regular.otf`, 30,759 Unicode codepoints partitioned into 102 range-compact slices).

* **Paper**: *Learning Better Font Slicing Strategies from Data* (Quan Wang and Yiran Mao)
* **Open-Source Repository**: https://github.com/wq2012/mlfont
* **PyPI Package**: https://pypi.org/project/mlfont/

## Training Configuration

* **Target Font**: `NotoSansSC-Regular.otf` (30,759 Unicode codepoints)
* **Training & Evaluation Corpus**: `wikimedia/wikipedia` (`20231101`, languages: `zh`), 1,600 training articles and 400 testing articles
* **Slicing Architecture**: Tiered Document-Frequency Slicing + Online Variational LDA Topic Modeling + Range-Compact Genetic Algorithm (`mlfont` preset: `chinese`)
* **Total Slices**: 102 slices (11,741 total CSS `unicode-range` intervals)
* **Physical WOFF2 Bundle Size**: 9183.8 KB across 102 `.woff2` files (companion CSS: 129.6 KB)
* **Training Wall-Time**: 118.6 seconds on CPU/GPU

## Model Files Included

* `model.safetensors`: Learned topic-codepoint weight matrix (`lambda_matrix`), `codepoints`, `frequency_array`, and `dna_assignment` in SafeTensors format
* `strategy.json`: Complete `SlicingStrategy` definition and hyperparameters
* `config.json`: Model and font bundle summary metadata
* `font_sliced.css`: Production-ready `@font-face` stylesheet with compact `unicode-range` descriptors
* `woff2/*.woff2`: 102 physical WOFF2 font slices generated via `fontTools.subset`
* `nam/*.nam`: 102 Google Fonts `.nam` codepoint subset files
* `evaluation_metrics.json`: Full benchmark metrics across all 7 network/session profiles

## Benchmark Performance (400 Wikipedia Test Pages, Session Length L = 10)

| Strategy | Slices | CSS Ranges | Missing Glyphs | GIN-2g Mean / P95 (ms) | GIN-3g Mean / P95 (ms) | DSL Mean / P95 (ms) | Session Util (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `single_slice` | 1 | 418 | 0 | 190.95 / 1046.58 | 47.78 / 261.86 | 9.54 / 52.31 | 5.00% |
| `equal_bin_102` | 102 | 519 | 0 | 140.76 / 851.22 | 37.53 / 226.94 | 6.80 / 41.15 | 6.90% |
| `random_slice_102` | 102 | 30457 | 0 | 202.63 / 1325.94 | 54.02 / 353.49 | 9.80 / 64.10 | 5.00% |
| `language_slice` | 12 | 7087 | 121 | 68.74 / 377.15 | 17.36 / 95.16 | 3.42 / 18.77 | 13.95% |
| `most_1000_500_equal_bin_100` | 102 | 2978 | 0 | 111.15 / 493.00 | 29.58 / 131.64 | 5.38 / 23.81 | 7.17% |
| `online_lda_topics100_greedy` | 102 | 11848 | 0 | 57.88 / 305.81 | 15.36 / 81.13 | 2.81 / 14.82 | 14.99% |
| `dnntm_topics100_greedy` | 102 | 11928 | 0 | 48.51 / 210.03 | 12.86 / 55.64 | 2.35 / 10.19 | 17.46% |
| **range_compact_most1000_500_topics100 (This Model)** | 102 | 11741 | 0 | 58.10 / 307.01 | 15.41 / 81.43 | 2.82 / 14.88 | 14.93% |

## Usage

Install `mlfont`:

```bash
pip install mlfont huggingface_hub
```

Load the pretrained bundle and analyze or slice a font in Python:

```python
from huggingface_hub import snapshot_download
import mlfont

model_dir = snapshot_download(repo_id="wq2012/mlfont-notosanssc")
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
