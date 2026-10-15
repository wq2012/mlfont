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

Pretrained **MLFont** machine-learning-guided font slicing model and WOFF2 slice bundle for **Noto Sans JP (Japanese)** (`NotoSansJP-Regular.otf`, 16,663 Unicode codepoints partitioned into 62 range-compact slices).

* **Paper**: *MLFont: Machine Learning-Driven Font Subsetting and Optimal Unicode-Range Slicing for Low-Latency Web Typography* (Quan Wang and Yiran Mao)
* **Open-Source Repository**: https://github.com/wq2012/mlfont
* **PyPI Package**: https://pypi.org/project/mlfont/

## Training Configuration

* **Target Font**: `NotoSansJP-Regular.otf` (16,663 Unicode codepoints)
* **Training & Evaluation Corpus**: `wikimedia/wikipedia` (`20231101`, languages: `ja`), 1,600 training articles and 400 uniformly sampled testing articles
* **Slicing Architecture**: Tiered Document-Frequency Slicing + Online Variational LDA & Deep Neural Network Topic Model (DNNTM) + Range-Compact Genetic Algorithm (`mlfont` preset: `japanese`)
* **Total Slices**: 62 slices (7,298 total CSS `unicode-range` intervals)
* **Physical WOFF2 Bundle Size**: 4.67 MB / 4779.7 KB (4,894,460 bytes) across 62 `.woff2` files (companion CSS: 85.3 KB)
* **Training Wall-Time**: 46.0 seconds on CPU/GPU

## Model Files Included

* `model.safetensors`: Learned topic-codepoint weight matrices (`lambda_matrix`, `dnntm_lambda_matrix`), `codepoints`, `remaining_codepoints`, `frequency_array`, and `dna_assignment` in SafeTensors format
* `strategy.json`: Range-Compact GA `SlicingStrategy` definition and hyperparameters
* `olda_strategy.json`: Online LDA + Greedy `SlicingStrategy` definition
* `dnntm_strategy.json`: DNNTM (Entropy Difference Loss) + Greedy `SlicingStrategy` definition
* `test_samples.json`: Exact held-out evaluation codepoint sets (400 test pages)
* `config.json`: Model and font bundle summary metadata (including per-slice WOFF2 byte sizes)
* `font_sliced.css`: Production-ready `@font-face` stylesheet with compact `unicode-range` descriptors
* `woff2/*.woff2`: 62 physical WOFF2 font slices generated via `fontTools.subset`
* `nam/*.nam`: 62 Google Fonts `.nam` codepoint subset files
* `evaluation_metrics.json`: Full benchmark metrics across all 7 network/session profiles

## Benchmark Performance (400 Wikipedia Test Pages, Session Length L = 10)

| Strategy | Slices | CSS Ranges | Missing (Eval / Font) | GIN-2g Mean / P95 (ms) | GIN-3g Mean / P95 (ms) | DSL Mean / P95 (ms) | Session Util (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `single_slice` | 1 | 4830 | 0 / 0 | 94.75 / 701.82 | 23.72 / 175.72 | 4.73 / 35.06 | 7.62% |
| `equal_bin_62` | 62 | 4876 | 0 / 0 | 95.38 / 698.56 | 25.58 / 187.34 | 4.60 / 33.66 | 8.56% |
| `random_slice_62` | 62 | 16450 | 0 / 0 | 114.88 / 893.64 | 30.81 / 239.66 | 5.53 / 43.06 | 7.62% |
| `language_slice` | 10 | 5617 | 43 / 7632 | 45.20 / 305.76 | 11.42 / 77.00 | 2.25 / 15.22 | 15.84% |
| `most_500_500_equal_bin_60` | 62 | 5975 | 0 / 0 | 76.50 / 387.35 | 20.50 / 103.64 | 3.69 / 18.69 | 8.92% |
| `online_lda_topics60_greedy` | 62 | 8888 | 0 / 0 | 34.87 / 201.00 | 9.31 / 53.61 | 1.68 / 9.71 | 21.31% |
| `dnntm_topics60_greedy` | 62 | 8710 | 0 / 0 | 34.80 / 188.72 | 9.30 / 50.47 | 1.68 / 9.12 | 20.88% |
| **range_compact_most500_500_topics60 (This Model)** | 62 | 7298 | 0 / 0 | 42.53 / 223.47 | 11.37 / 59.70 | 2.05 / 10.79 | 16.91% |

## Usage

Install `mlfont`:

```bash
pip install mlfont huggingface_hub
```

Load the pretrained bundle (Range-Compact GA, Online LDA, and DNNTM) in Python:

```python
import mlfont

bundle = mlfont.load_pretrained("wq2012/mlfont-notosansjp")
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
