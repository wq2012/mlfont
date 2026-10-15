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

* **Paper**: *MLFont: Machine Learning-Driven Font Subsetting and Optimal Unicode-Range Slicing for Low-Latency Web Typography* (Quan Wang and Yiran Mao)
* **Open-Source Repository**: https://github.com/wq2012/mlfont
* **PyPI Package**: https://pypi.org/project/mlfont/

## Using This Sliced Font in Your Website (Frontend Developer Guide)

While basic Latin webfonts are typically 20–100 KB, Pan-Unicode and East Asian fonts like **Noto Sans SC (Simplified Chinese)** (`NotoSansSC-Regular.otf`) are much larger (8.89 MB / 9107.9 KB (9,326,524 bytes) in WOFF2) because they include **30,759 characters**. In any large corpus of documents, there are strong co-occurrence patterns in character usage. By scanning webpages in `zh`, **MLFont** models which characters are most likely to appear together, informing a static subset slicing strategy that sorts all 30,759 glyphs into **102 slices** (5,935 compact CSS `unicode-range` intervals).

Using this method, your entire web page renders in **Noto Sans SC (Simplified Chinese)** while the browser only downloads the small subset of `.woff2` font slices required for the characters present on that page—which is dramatically faster than loading the whole font. Cross-page and cross-site browser caching further increases the latency benefits over time. *(Note: The slicing optimization works automatically in all modern [browsers that support CSS `unicode-range`](https://caniuse.com/font-unicode-range).)*

### 1. Link to Sliced Font CSS (HTML `<link>` or CSS `@import`)

Add a `<link>` element to the `<head>` section of your HTML:

```html
<link rel="stylesheet"
      href="https://cdn.jsdelivr.net/gh/wq2012/mlfont@main/pretrained_models/mlfont-notosanssc/font_sliced_cdn.css">
```

Or in your CSS / SCSS stylesheet:

```css
@import url('https://cdn.jsdelivr.net/gh/wq2012/mlfont@main/pretrained_models/mlfont-notosanssc/font_sliced_cdn.css');
```

### 2. Designate Font Family in CSS

Style your text with the `'Noto Sans SC Sliced'` family in your CSS:

```css
body {
  font-family: 'Noto Sans SC Sliced', 'NotoSansSC-Regular Sliced', Roboto, 'Noto Sans', sans-serif;
}
```

*(Note: `font_sliced_cdn.css` streams the `.woff2` slices on-demand directly from Hugging Face's global CDN `https://huggingface.co/wq2012/mlfont-notosanssc/resolve/main/woff2/...` with `font-display: swap` and CORS `Access-Control-Allow-Origin: *`, so you do not need to host the `.woff2` files yourself. To **self-host** on your own server/CDN, download this repository and link `font_sliced.css` alongside the `woff2/` folder.)*

### 3. Alternative: One-Line Drop-In JavaScript Loader (`mlfont.js`)

You can also load the sliced stylesheet directly from Hugging Face CDN using a single `<script>` tag with `data-mlfont`:

```html
<script src="https://cdn.jsdelivr.net/gh/wq2012/mlfont@main/web/mlfont.js"
        data-mlfont="notosanssc"
        async></script>
```

Or programmatically in JavaScript:

```javascript
MLFont.load('notosanssc');
```

### 4. Latency Metrics & Browser Telemetry (Recommended — `mlfont.js`)

Included in this bundle (`mlfont.js`, an open-source reproduction and extension of Google Fonts' `//fonts.gstatic.com/ea/timing/v1/mlfont.js`) is a small, zero-dependency JS library that accesses the HTML5 [Performance API](https://developer.mozilla.org/en-US/docs/Web/API/Performance) to measure webfont slice performance. **No user-specific data is collected.** The JS library collects:

1. **Download time metrics of individual font slices** (slice index, TTFB, download duration, transfer bytes, cache hit status)
2. **Download time of the sliced `@font-face` CSS file**
3. **Page `onload` and First Contentful Paint (`FCP`) time**

To include the library, add the following code to the bottom of your `<head>` section:

```html
<script src="https://cdn.jsdelivr.net/gh/wq2012/mlfont@main/web/mlfont.js" async></script>
```

You can inspect downloaded slice metrics in your browser DevTools console via `MLFont.printReport()` or `MLFont.getMetrics()`, or forward anonymized telemetry beacons to your own analytics endpoint via `data-endpoint="https://your-analytics.example.com/mlfont-timing"`.

## Training Configuration

* **Target Font**: `NotoSansSC-Regular.otf` (30,759 Unicode codepoints)
* **Training & Evaluation Corpus**: `wikimedia/wikipedia` (`20231101`, languages: `zh`), 1,600 training articles and 400 uniformly sampled testing articles
* **Slicing Architecture**: Tiered Document-Frequency Slicing + Online Variational LDA & Deep Neural Network Topic Model (DNNTM) + Range-Compact Genetic Algorithm (`mlfont` preset: `chinese`)
* **Total Slices**: 102 slices (5,935 total CSS `unicode-range` intervals)
* **Physical WOFF2 Bundle Size**: 8.89 MB / 9107.9 KB (9,326,524 bytes) across 102 `.woff2` files (companion CSS: 83.9 KB)
* **Training Wall-Time**: 87.6 seconds on CPU/GPU

## Model Files Included

* `font_sliced_cdn.css`: Zero-hosting CDN `@font-face` stylesheet (`font-family: 'Noto Sans SC Sliced'`) streaming `.woff2` slices directly from Hugging Face CDN
* `font_sliced.css`: Self-hosted `@font-face` stylesheet (`font-family: 'NotoSansSC-Regular Sliced'`) with compact `unicode-range` descriptors
* `mlfont.js`: Browser JS loader (`MLFont.load`) & HTML5 Performance API slice latency telemetry library (`MLFont.getMetrics` / `window.GoogleFonts`)
* `woff2/*.woff2`: 102 physical WOFF2 font slices generated via `fontTools.subset`
* `nam/*.nam`: 102 Google Fonts `.nam` codepoint subset files
* `model.safetensors`: Learned topic-codepoint weight matrices (`lambda_matrix`, `dnntm_lambda_matrix`), `codepoints`, `remaining_codepoints`, `frequency_array`, and `dna_assignment` in SafeTensors format
* `strategy.json`: Range-Compact GA `SlicingStrategy` definition and hyperparameters
* `olda_strategy.json`: Online LDA + Greedy `SlicingStrategy` definition
* `dnntm_strategy.json`: DNNTM (Entropy Difference Loss) + Greedy `SlicingStrategy` definition
* `test_samples.json`: Exact held-out evaluation codepoint sets (400 test pages)
* `config.json`: Model and font bundle summary metadata (including per-slice WOFF2 byte sizes)
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
| **range_compact_most1000_500_topics100 (This Model)** | 102 | 5935 | 0 / 0 | 66.67 / 356.73 | 17.67 / 94.33 | 3.23 / 17.32 | 14.64% |

## Python API Usage (ML Researchers & Custom Font Slicing)

Install `mlfont`:

```bash
pip install mlfont huggingface_hub
```

Load the pretrained bundle (Range-Compact GA, Online LDA, and DNNTM) in Python:

```python
import mlfont

bundle = mlfont.load_pretrained("wq2012/mlfont-notosanssc")
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
