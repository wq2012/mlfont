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
* **Training & Evaluation Corpus**: `wikimedia/wikipedia` (`20231101`, languages: `zh`), 1,600 training articles and 400 uniformly sampled testing articles
* **Slicing Architecture**: Tiered Document-Frequency Slicing + Online Variational LDA Topic Modeling + Range-Compact Genetic Algorithm (`mlfont` preset: `chinese`)
* **Total Slices**: 102 slices (9,220 total CSS `unicode-range` intervals)
* **Physical WOFF2 Bundle Size**: 9145.2 KB across 102 `.woff2` files (companion CSS: 109.8 KB)
* **Training Wall-Time**: 74.4 seconds on CPU/GPU

## Model Files Included

* `model.safetensors`: Learned topic-codepoint weight matrices (`lambda_matrix`, `dnntm_lambda_matrix`), `codepoints`, `remaining_codepoints`, `frequency_array`, and `dna_assignment` in SafeTensors format
* `strategy.json`: Range-Compact GA `SlicingStrategy` definition and hyperparameters
* `olda_strategy.json`: Online LDA + Greedy `SlicingStrategy` definition
* `dnntm_strategy.json`: DNNTM (Entropy Difference Loss) + Greedy `SlicingStrategy` definition
* `test_samples.json`: Exact held-out evaluation codepoint sets (400 test pages)
* `config.json`: Model and font bundle summary metadata (including per-slice WOFF2 byte sizes)
* `font_sliced.css`: Production-ready `@font-face` stylesheet with compact `unicode-range` descriptors
* `woff2/*.woff2`: 102 physical WOFF2 font slices generated via `fontTools.subset`
* `nam/*.nam`: 102 Google Fonts `.nam` codepoint subset files
* `evaluation_metrics.json`: Full benchmark metrics across all 7 network/session profiles

## Benchmark Performance (400 Wikipedia Test Pages, Session Length L = 10)

| Strategy | Slices | CSS Ranges | Missing (Eval / Font) | GIN-2g Mean / P95 (ms) | GIN-3g Mean / P95 (ms) | DSL Mean / P95 (ms) | Session Util (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `single_slice` | 1 | 418 | 0 / 0 | 253.67 / 1629.56 | 63.47 / 407.72 | 12.68 / 81.45 | 5.55% |
| `equal_bin_102` | 102 | 519 | 0 / 0 | 169.75 / 1165.36 | 45.26 / 310.69 | 8.21 / 56.33 | 7.63% |
| `random_slice_102` | 102 | 30457 | 0 / 0 | 242.74 / 1930.38 | 64.71 / 514.64 | 11.73 / 93.32 | 5.55% |
| `language_slice` | 10 | 6646 | 126 / 19659 | 86.54 / 554.31 | 21.82 / 139.59 | 4.31 / 27.61 | 16.05% |
| `most_1000_500_equal_bin_100` | 102 | 2965 | 0 / 0 | 110.52 / 502.64 | 29.38 / 131.07 | 5.35 / 24.59 | 7.82% |
| `google_fonts_ea_2017` | 102 | 6063 | 0 / 0 | 115.96 / 609.01 | 30.29 / 160.05 | 5.67 / 29.67 | 7.74% |
| `online_lda_topics100_greedy` | 102 | 10177 | 0 / 0 | 60.16 / 314.64 | 15.93 / 83.23 | 2.92 / 15.27 | 16.90% |
| `dnntm_topics100_greedy` | 102 | 9640 | 0 / 0 | 59.30 / 311.97 | 15.71 / 82.49 | 2.88 / 15.14 | 16.57% |
| **range_compact_most1000_500_topics100 (This Model)** | 102 | 9220 | 0 / 0 | 66.51 / 343.80 | 17.61 / 90.79 | 3.23 / 16.71 | 14.77% |

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
