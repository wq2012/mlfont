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

Pretrained **MLFont** machine-learning-guided font slicing model and WOFF2 slice bundle for **Noto Sans KR (Korean)** (`NotoSansKR-Regular.otf`, 22,853 Unicode codepoints partitioned into 52 range-compact slices).

* **Paper**: *Learning Better Font Slicing Strategies from Data* (Quan Wang and Yiran Mao)
* **Open-Source Repository**: https://github.com/wq2012/mlfont
* **PyPI Package**: https://pypi.org/project/mlfont/

## Training Configuration

* **Target Font**: `NotoSansKR-Regular.otf` (22,853 Unicode codepoints)
* **Training & Evaluation Corpus**: `wikimedia/wikipedia` (`20231101`, languages: `ko`), 1,600 training articles and 400 uniformly sampled testing articles
* **Slicing Architecture**: Tiered Document-Frequency Slicing + Online Variational LDA Topic Modeling + Range-Compact Genetic Algorithm (`mlfont` preset: `korean`)
* **Total Slices**: 52 slices (7,859 total CSS `unicode-range` intervals)
* **Physical WOFF2 Bundle Size**: 5343.5 KB across 52 `.woff2` files (companion CSS: 86.7 KB)
* **Training Wall-Time**: 46.6 seconds on CPU/GPU

## Model Files Included

* `model.safetensors`: Learned topic-codepoint weight matrices (`lambda_matrix`, `dnntm_lambda_matrix`), `codepoints`, `remaining_codepoints`, `frequency_array`, and `dna_assignment` in SafeTensors format
* `strategy.json`: Range-Compact GA `SlicingStrategy` definition and hyperparameters
* `olda_strategy.json`: Online LDA + Greedy `SlicingStrategy` definition
* `dnntm_strategy.json`: DNNTM (Entropy Difference Loss) + Greedy `SlicingStrategy` definition
* `test_samples.json`: Exact held-out evaluation codepoint sets (400 test pages)
* `config.json`: Model and font bundle summary metadata (including per-slice WOFF2 byte sizes)
* `font_sliced.css`: Production-ready `@font-face` stylesheet with compact `unicode-range` descriptors
* `woff2/*.woff2`: 52 physical WOFF2 font slices generated via `fontTools.subset`
* `nam/*.nam`: 52 Google Fonts `.nam` codepoint subset files
* `evaluation_metrics.json`: Full benchmark metrics across all 7 network/session profiles

## Benchmark Performance (400 Wikipedia Test Pages, Session Length L = 10)

| Strategy | Slices | CSS Ranges | Missing (Eval / Font) | GIN-2g Mean / P95 (ms) | GIN-3g Mean / P95 (ms) | DSL Mean / P95 (ms) | Session Util (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `single_slice` | 1 | 4598 | 0 / 0 | 250.67 / 1739.45 | 62.73 / 435.34 | 12.53 / 86.92 | 3.31% |
| `equal_bin_52` | 52 | 4635 | 0 / 0 | 259.74 / 1338.41 | 68.09 / 350.87 | 12.67 / 65.29 | 3.76% |
| `random_slice_52` | 52 | 22493 | 0 / 0 | 270.86 / 2029.63 | 71.01 / 532.06 | 13.21 / 99.02 | 3.31% |
| `language_slice` | 10 | 4579 | 34 / 4322 | 203.56 / 1514.81 | 51.20 / 380.39 | 10.15 / 75.57 | 4.22% |
| `most_300_300_equal_bin_50` | 52 | 5608 | 0 / 0 | 197.85 / 715.15 | 51.88 / 187.72 | 9.65 / 34.86 | 3.81% |
| `google_fonts_2017_kr` | 102 | 7229 | 47 / 5467 | 83.54 / 354.53 | 22.38 / 94.50 | 4.03 / 17.20 | 13.64% |
| `online_lda_topics50_greedy` | 52 | 8327 | 0 / 0 | 66.59 / 348.80 | 17.48 / 91.73 | 3.25 / 16.99 | 12.03% |
| `dnntm_topics50_greedy` | 52 | 8354 | 0 / 0 | 66.97 / 331.00 | 17.59 / 87.21 | 3.26 / 16.10 | 11.51% |
| **range_compact_most300_300_topics50 (This Model)** | 52 | 7859 | 0 / 0 | 70.36 / 360.67 | 18.46 / 94.44 | 3.43 / 17.61 | 11.25% |

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
