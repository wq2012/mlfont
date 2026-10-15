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

## Overview

Pretrained **MLFont** machine-learning-guided font slicing model and WOFF2 slice bundle for **Noto Sans JP (Japanese)** (`NotoSansJP-Regular.otf`, 16,663 Unicode codepoints partitioned into 62 range-compact slices).

* **Paper**: *MLFont: Machine Learning-Driven Font Subsetting and Optimal Unicode-Range Slicing for Low-Latency Web Typography* (Quan Wang and Yiran Mao)
* **Background & References**: Open-source reproduction of Google's machine learning font slicing technology ([Official Post](https://x.com/googlefonts/status/900019727732531200), [Defensive Publication](https://www.tdcommons.org/dpubs_series/906/))
* **Open-Source Repository**: https://github.com/wq2012/mlfont
* **PyPI Package**: https://pypi.org/project/mlfont/

## Using This Sliced Font in Your Website (Frontend Developer Guide)

While basic Latin webfonts are typically 20–100 KB, Pan-Unicode and East Asian fonts like **Noto Sans JP (Japanese)** (`NotoSansJP-Regular.otf`) are much larger (4.67 MB / 4779.7 KB (4,894,460 bytes) in WOFF2) because they include **16,663 characters**. In any large corpus of documents, there are strong co-occurrence patterns in character usage. By scanning webpages in `ja`, **MLFont** models which characters are most likely to appear together, informing a static subset slicing strategy that sorts all 16,663 glyphs into **62 slices** (7,298 compact CSS `unicode-range` intervals).

Using this method, your entire web page renders in **Noto Sans JP (Japanese)** while the browser only downloads the small subset of `.woff2` font slices required for the characters present on that page—which is dramatically faster than loading the whole font. Cross-page and cross-site browser caching further increases the latency benefits over time. *(Note: The slicing optimization works automatically in all modern [browsers that support CSS `unicode-range`](https://caniuse.com/font-unicode-range).)*

### 1. Link to Sliced Font CSS (HTML `<link>` or CSS `@import`)

Add a `<link>` element to the `<head>` section of your HTML:

```html
<link rel="stylesheet"
      href="https://cdn.jsdelivr.net/gh/wq2012/mlfont@main/pretrained_models/mlfont-notosansjp/font_sliced_cdn.css">
```

Or in your CSS / SCSS stylesheet:

```css
@import url('https://cdn.jsdelivr.net/gh/wq2012/mlfont@main/pretrained_models/mlfont-notosansjp/font_sliced_cdn.css');
```

### 2. Designate Font Family in CSS

Style your text with the `'Noto Sans JP Sliced'` family in your CSS:

```css
body {
  font-family: 'Noto Sans JP Sliced', 'NotoSansJP-Regular Sliced', Roboto, 'Noto Sans', sans-serif;
}
```

*(Note: `font_sliced_cdn.css` streams the `.woff2` slices on-demand directly from Hugging Face's global CDN `https://huggingface.co/wq2012/mlfont-notosansjp/resolve/main/woff2/...` with `font-display: swap` and CORS `Access-Control-Allow-Origin: *`, so you do not need to host the `.woff2` files yourself. To **self-host** on your own server/CDN, download this repository and link `font_sliced.css` alongside the `woff2/` folder.)*

### 3. Alternative: One-Line Drop-In JavaScript Loader (`mlfont.js`)

You can also load the sliced stylesheet directly from Hugging Face CDN using a single `<script>` tag with `data-mlfont`:

```html
<script src="https://cdn.jsdelivr.net/gh/wq2012/mlfont@main/web/mlfont.js"
        data-mlfont="notosansjp"
        async></script>
```

Or programmatically in JavaScript:

```javascript
MLFont.load('notosansjp');
```

### 4. Latency Metrics & Browser Telemetry (Recommended — `mlfont.js`)

Included in this bundle (`mlfont.js`) is a small, zero-dependency JS library that accesses the HTML5 [Performance API](https://developer.mozilla.org/en-US/docs/Web/API/Performance) to measure webfont slice performance. **No user-specific data is collected.** The JS library collects:

1. **Download time metrics of individual font slices** (slice index, TTFB, download duration, transfer bytes, cache hit status)
2. **Download time of the sliced `@font-face` CSS file**
3. **Page `onload` and First Contentful Paint (`FCP`) time**

To include the library, add the following code to the bottom of your `<head>` section:

```html
<script src="https://cdn.jsdelivr.net/gh/wq2012/mlfont@main/web/mlfont.js" async></script>
```

You can inspect downloaded slice metrics in your browser DevTools console via `MLFont.printReport()` or `MLFont.getMetrics()`, or forward anonymized telemetry beacons to your own analytics endpoint via `data-endpoint="https://your-analytics.example.com/mlfont-timing"`.

### 5. Simulating Real-Website Font Loading Latency & Headless Chrome Verification

Across **3 real-world sample websites** for **Noto Sans JP (Japanese)** (full unsliced WOFF2: `3,260.2 KB`), **`range_compact_most500_500_topics60`** downloads an average of only **`10.33 / 62`** slices (`1,062.9 KB` vs. `3,279.1 KB` unsliced, saving **`67.6%`** bandwidth). On `Cable-5Mbps` (`5 Mbps`, `28 ms` RTT), this achieves a **`2.94x`** cold-cache load speedup (**`1,851 ms`** vs. `5,437 ms` for `single_slice`), with **First Text Paint at `689 ms`** and **Warm-Session Load at `568 ms`** (and **`2.94x`** speedup on `GIN-3g`: **`23,117 ms`** vs. `67,965 ms`):

| Strategy | Slices Dl / Total | Transfer (KB) | Bytes Saved | Cable-5Mbps 1st Paint (ms) | Cable-5Mbps Cold Load (ms) | Cable-5Mbps Warm Load (ms) | GIN-3g Cold Load (ms) | Cold Speedup (Cable / 3G) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `single_slice` | 1.00 / 1 | 3,279.1 KB | 0.0% | 5,437 ms | **5,437 ms** | 0 ms | 67,965 ms | **1.00x / 1.00x** |
| `equal_bin_62` | 37.00 / 62 | 2,873.3 KB | 12.4% | 859 ms | **4,924 ms** | 989 ms | 61,789 ms | **1.10x / 1.10x** |
| `most_500_500_equal_bin_60` | 14.00 / 62 | 1,270.6 KB | 61.2% | 678 ms | **2,207 ms** | 1,727 ms | 27,636 ms | **2.46x / 2.46x** |
| `language_slice` | 2.00 / 10 | 1,949.9 KB | 40.5% | 3,261 ms | **3,261 ms** | 0 ms | 40,745 ms | **1.67x / 1.67x** |
| `online_lda_topics60_greedy` | 9.33 / 62 | 930.8 KB | 71.6% | 696 ms | **1,628 ms** | 521 ms | 20,281 ms | **3.34x / 3.35x** |
| `dnntm_topics60_greedy` | 7.67 / 62 | 805.1 KB | 75.5% | 695 ms | **1,412 ms** | 644 ms | 17,573 ms | **3.85x / 3.87x** |
| **range_compact_most500_500_topics60 (This Model)** | 10.33 / 62 | 1,062.9 KB | 67.6% | 689 ms | **1,851 ms** | 568 ms | 23,117 ms | **2.94x / 2.94x** |

#### Real Headless Chrome (`v155.0.8059.79`) Verification (`Cable-5Mbps`)

Verified against physical `.woff2` files in Headless Chrome (`--headless=new --dump-dom`) via the W3C `document.fonts` (`FontFaceSet`) API (`9/9 = 100%` exact slice-set match between Chromium Blink's `unicode-range` engine and `WebsiteFontSimulator`):

| Website Page ID | Strategy | Chrome Loaded Slices | Sim Exact Match | Transfer (KB) | Chrome Local Decode (ms) | E2E Font Ready (ms) | Decode Speedup | E2E Speedup (Bytes Saved) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `jp_web_dev_docs` | `single_slice` | 1 | **Yes** | 3,279.1 KB | 170.2 ms | **5,599 ms** | 1.00x | **1.00x** (0.0%) |
| `jp_web_dev_docs` | `equal_bin_62` | 34 | **Yes** | 2,642.4 KB | 91.0 ms | **4,616 ms** | 1.87x | **1.21x** (19.4%) |
| `jp_web_dev_docs` | **range_compact_most500_500_topics60 (This Model)** | 7 | **Yes** | 773.6 KB | 54.3 ms | **1,406 ms** | 3.13x | **3.98x** (76.4%) |
| `jp_sports_baseball` | `single_slice` | 1 | **Yes** | 3,279.1 KB | 148.9 ms | **5,577 ms** | 1.00x | **1.00x** (0.0%) |
| `jp_sports_baseball` | `equal_bin_62` | 35 | **Yes** | 2,718.8 KB | 104.3 ms | **4,755 ms** | 1.43x | **1.17x** (17.1%) |
| `jp_sports_baseball` | **range_compact_most500_500_topics60 (This Model)** | 10 | **Yes** | 1,054.6 KB | 60.0 ms | **1,872 ms** | 2.48x | **2.98x** (67.8%) |
| `jp_literature_essay` | `single_slice` | 1 | **Yes** | 3,279.1 KB | 147.5 ms | **5,576 ms** | 1.00x | **1.00x** (0.0%) |
| `jp_literature_essay` | `equal_bin_62` | 42 | **Yes** | 3,258.6 KB | 141.2 ms | **5,704 ms** | 1.04x | **0.98x** (0.6%) |
| `jp_literature_essay` | **range_compact_most500_500_topics60 (This Model)** | 14 | **Yes** | 1,360.4 KB | 61.5 ms | **2,402 ms** | 2.40x | **2.32x** (58.5%) |

You can test your own HTML files, live URLs, or text snippets against `mlfont-notosansjp` before deploying using the `mlfont-simulate` CLI, the Python `WebsiteFontSimulator` API, or the interactive side-by-side **Website Font Loading Simulator** in [`web/demo.html`](https://github.com/wq2012/mlfont/blob/main/web/demo.html):

```bash
# Simulate a local HTML file with mlfont-notosansjp across all 7 network profiles:
mlfont-simulate --font_key notosansjp --html_file path/to/index.html --network_profile all

# Simulate a live URL and verify in real Headless Chrome:
mlfont-simulate --font_key notosansjp --url https://example.com --network_profile Cable-5Mbps --headless_chrome
```

## Training Configuration

* **Target Font**: `NotoSansJP-Regular.otf` (16,663 Unicode codepoints)
* **Training & Evaluation Corpus**: `wikimedia/wikipedia` (`20231101`, languages: `ja`), 1,600 training articles and 400 uniformly sampled testing articles
* **Slicing Architecture**: Tiered Document-Frequency Slicing + Online Variational LDA & Deep Neural Network Topic Model (DNNTM) + Range-Compact Genetic Algorithm (`mlfont` preset: `japanese`)
* **Total Slices**: 62 slices (7,298 total CSS `unicode-range` intervals)
* **Physical WOFF2 Bundle Size**: 4.67 MB / 4779.7 KB (4,894,460 bytes) across 62 `.woff2` files (companion CSS: 85.3 KB)
* **Training Wall-Time**: 25.0 seconds on CPU/GPU

## Model Files Included

* `font_sliced_cdn.css`: Zero-hosting CDN `@font-face` stylesheet (`font-family: 'Noto Sans JP Sliced'`) streaming `.woff2` slices directly from Hugging Face CDN
* `font_sliced.css`: Self-hosted `@font-face` stylesheet (`font-family: 'NotoSansJP-Regular Sliced'`) with compact `unicode-range` descriptors
* `mlfont.js`: Browser JS loader (`MLFont.load`) & HTML5 Performance API slice latency telemetry library (`MLFont.getMetrics` / `MLFont.printReport`)
* `woff2/*.woff2`: 62 physical WOFF2 font slices generated via `fontTools.subset`
* `nam/*.nam`: 62 `.nam` codepoint subset files
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
| `single_slice` | 1 | 4830 | 0 / 0 | 94.75 / 701.82 | 23.72 / 175.72 | 4.73 / 35.06 | 7.62% |
| `equal_bin_62` | 62 | 4876 | 0 / 0 | 95.38 / 698.56 | 25.58 / 187.34 | 4.60 / 33.66 | 8.56% |
| `random_slice_62` | 62 | 16450 | 0 / 0 | 114.88 / 893.64 | 30.81 / 239.66 | 5.53 / 43.06 | 7.62% |
| `language_slice` | 10 | 5617 | 43 / 7632 | 45.20 / 305.76 | 11.42 / 77.00 | 2.25 / 15.22 | 15.84% |
| `most_500_500_equal_bin_60` | 62 | 5975 | 0 / 0 | 76.50 / 387.35 | 20.50 / 103.64 | 3.69 / 18.69 | 8.92% |
| `online_lda_topics60_greedy` | 62 | 8888 | 0 / 0 | 34.87 / 201.00 | 9.31 / 53.61 | 1.68 / 9.71 | 21.31% |
| `dnntm_topics60_greedy` | 62 | 8710 | 0 / 0 | 34.80 / 188.72 | 9.30 / 50.47 | 1.68 / 9.12 | 20.88% |
| **range_compact_most500_500_topics60 (This Model)** | 62 | 7298 | 0 / 0 | 42.53 / 223.47 | 11.37 / 59.70 | 2.05 / 10.79 | 16.91% |

## Python API Usage (ML Researchers & Custom Font Slicing)

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
