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

* **Paper**: *MLFont: Machine Learning-Driven Font Subsetting and Optimal Unicode-Range Slicing for Low-Latency Web Typography* (Quan Wang and Yiran Mao)
* **Open-Source Repository**: https://github.com/wq2012/mlfont
* **PyPI Package**: https://pypi.org/project/mlfont/

## Training Configuration

* **Target Font**: `NotoSans-Regular.ttf` (2,297 Unicode codepoints)
* **Training & Evaluation Corpus**: `wikimedia/wikipedia` (`20231101`, languages: `en, de, fr, pl, ru, el`), 1,500 training articles and 300 uniformly sampled testing articles
* **Slicing Architecture**: Tiered Document-Frequency Slicing + Online Variational LDA & Deep Neural Network Topic Model (DNNTM) + Range-Compact Genetic Algorithm (`mlfont` preset: `lgc`)
* **Total Slices**: 20 slices (384 total CSS `unicode-range` intervals)
* **Physical WOFF2 Bundle Size**: 434.1 KB (444,484 bytes) across 20 `.woff2` files (companion CSS: 7.8 KB)
* **Training Wall-Time**: 24.0 seconds on CPU/GPU

## Model Files Included

* `model.safetensors`: Learned topic-codepoint weight matrices (`lambda_matrix`, `dnntm_lambda_matrix`), `codepoints`, `remaining_codepoints`, `frequency_array`, and `dna_assignment` in SafeTensors format
* `strategy.json`: Range-Compact GA `SlicingStrategy` definition and hyperparameters
* `olda_strategy.json`: Online LDA + Greedy `SlicingStrategy` definition
* `dnntm_strategy.json`: DNNTM (Entropy Difference Loss) + Greedy `SlicingStrategy` definition
* `test_samples.json`: Exact held-out evaluation codepoint sets (300 test pages)
* `config.json`: Model and font bundle summary metadata (including per-slice WOFF2 byte sizes)
* `font_sliced.css`: Production-ready `@font-face` stylesheet with compact `unicode-range` descriptors
* `woff2/*.woff2`: 20 physical WOFF2 font slices generated via `fontTools.subset`
* `nam/*.nam`: 20 Google Fonts `.nam` codepoint subset files
* `evaluation_metrics.json`: Full benchmark metrics across all 7 network/session profiles

## Benchmark Performance (300 Wikipedia Test Pages, Session Length L = 10)

| Strategy | Slices | CSS Ranges | Missing (Eval / Font) | GIN-2g Mean / P95 (ms) | GIN-3g Mean / P95 (ms) | DSL Mean / P95 (ms) | Session Util (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `single_slice` | 1 | 123 | 0 / 0 | 60.26 / 566.45 | 15.22 / 143.10 | 3.00 / 28.17 | 8.66% |
| `equal_bin_20` | 20 | 140 | 0 / 0 | 33.37 / 153.43 | 9.44 / 43.41 | 1.56 / 7.17 | 23.17% |
| `random_slice_20` | 20 | 2180 | 0 / 0 | 95.82 / 906.75 | 27.11 / 256.54 | 4.48 / 42.35 | 8.66% |
| `language_slice` | 8 | 92 | 6 / 300 | 40.17 / 258.54 | 10.62 / 66.28 | 1.95 / 12.76 | 15.08% |
| `online_lda_topics20_greedy` | 20 | 537 | 0 / 0 | 25.15 / 144.76 | 7.11 / 40.95 | 1.17 / 6.76 | 31.26% |
| `dnntm_topics20_greedy` | 20 | 549 | 0 / 0 | 23.29 / 112.70 | 6.59 / 31.88 | 1.09 / 5.26 | 32.56% |
| **range_compact_topics20 (This Model)** | 20 | 384 | 0 / 0 | 25.05 / 142.01 | 7.06 / 40.06 | 1.17 / 6.64 | 30.31% |

## Usage

Install `mlfont`:

```bash
pip install mlfont huggingface_hub
```

Load the pretrained bundle (Range-Compact GA, Online LDA, and DNNTM) in Python:

```python
import mlfont

bundle = mlfont.load_pretrained("wq2012/mlfont-notosans-lgc")
print("GA:", bundle.strategy.name, "ranges:", bundle.strategy.count_unicode_ranges())
print("LDA:", bundle.olda_strategy.name, "ranges:", bundle.olda_strategy.count_unicode_ranges())
print("DNNTM:", bundle.dnntm_strategy.name, "ranges:", bundle.dnntm_strategy.count_unicode_ranges())
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
