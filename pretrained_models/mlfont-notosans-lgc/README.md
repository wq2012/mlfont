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

## Using This Sliced Font in Your Website (Frontend Developer Guide)

While basic Latin webfonts are typically 20–100 KB, Pan-Unicode and East Asian fonts like **Noto Sans (Pan-European LGC)** (`NotoSans-Regular.ttf`) are much larger (434.1 KB (444,484 bytes) in WOFF2) because they include **2,297 characters**. In any large corpus of documents, there are strong co-occurrence patterns in character usage. By scanning webpages in `en, de, fr, pl, ru, el`, **MLFont** models which characters are most likely to appear together, informing a static subset slicing strategy that sorts all 2,297 glyphs into **20 slices** (384 compact CSS `unicode-range` intervals).

Using this method, your entire web page renders in **Noto Sans (Pan-European LGC)** while the browser only downloads the small subset of `.woff2` font slices required for the characters present on that page—which is dramatically faster than loading the whole font. Cross-page and cross-site browser caching further increases the latency benefits over time. *(Note: The slicing optimization works automatically in all modern [browsers that support CSS `unicode-range`](https://caniuse.com/font-unicode-range).)*

### 1. Link to Sliced Font CSS (HTML `<link>` or CSS `@import`)

Add a `<link>` element to the `<head>` section of your HTML:

```html
<link rel="stylesheet"
      href="https://cdn.jsdelivr.net/gh/wq2012/mlfont@main/pretrained_models/mlfont-notosans-lgc/font_sliced_cdn.css">
```

Or in your CSS / SCSS stylesheet:

```css
@import url('https://cdn.jsdelivr.net/gh/wq2012/mlfont@main/pretrained_models/mlfont-notosans-lgc/font_sliced_cdn.css');
```

### 2. Designate Font Family in CSS

Style your text with the `'Noto Sans Sliced'` family in your CSS:

```css
body {
  font-family: 'Noto Sans Sliced', 'NotoSans-Regular Sliced', Roboto, 'Noto Sans', sans-serif;
}
```

*(Note: `font_sliced_cdn.css` streams the `.woff2` slices on-demand directly from Hugging Face's global CDN `https://huggingface.co/wq2012/mlfont-notosans-lgc/resolve/main/woff2/...` with `font-display: swap` and CORS `Access-Control-Allow-Origin: *`, so you do not need to host the `.woff2` files yourself. To **self-host** on your own server/CDN, download this repository and link `font_sliced.css` alongside the `woff2/` folder.)*

### 3. Alternative: One-Line Drop-In JavaScript Loader (`mlfont.js`)

You can also load the sliced stylesheet directly from Hugging Face CDN using a single `<script>` tag with `data-mlfont`:

```html
<script src="https://cdn.jsdelivr.net/gh/wq2012/mlfont@main/web/mlfont.js"
        data-mlfont="notosans-lgc"
        async></script>
```

Or programmatically in JavaScript:

```javascript
MLFont.load('notosans-lgc');
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

### 5. Simulating Real-Website Font Loading Latency & Headless Chrome Verification

Across **3 real-world sample websites** for **Noto Sans (Pan-European LGC)** (full unsliced WOFF2: `157.7 KB`), **`range_compact_topics20`** downloads an average of only **`4.00 / 20`** slices (`87.6 KB` vs. `158.4 KB` unsliced, saving **`44.7%`** bandwidth). On `Cable-5Mbps` (`5 Mbps`, `28 ms` RTT), this achieves a **`1.57x`** cold-cache load speedup (**`201 ms`** vs. `316 ms` for `single_slice`), with **First Text Paint at `94 ms`** and **Warm-Session Load at `0 ms`** (and **`1.56x`** speedup on `GIN-3g`: **`2,596 ms`** vs. `4,043 ms`):

| Strategy | Slices Dl / Total | Transfer (KB) | Bytes Saved | Cable-5Mbps 1st Paint (ms) | Cable-5Mbps Cold Load (ms) | Cable-5Mbps Warm Load (ms) | GIN-3g Cold Load (ms) | Cold Speedup (Cable / 3G) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `single_slice` | 1.00 / 1 | 158.4 KB | 0.0% | 316 ms | **316 ms** | 0 ms | 4,043 ms | **1.00x / 1.00x** |
| `equal_bin_20` | 4.00 / 20 | 87.8 KB | 44.6% | 201 ms | **201 ms** | 32 ms | 2,599 ms | **1.57x / 1.56x** |
| `language_slice` | 3.00 / 8 | 138.8 KB | 12.4% | 284 ms | **284 ms** | 49 ms | 3,642 ms | **1.11x / 1.11x** |
| `online_lda_topics20_greedy` | 3.67 / 20 | 82.0 KB | 48.2% | 97 ms | **192 ms** | 0 ms | 2,480 ms | **1.64x / 1.63x** |
| `dnntm_topics20_greedy` | 3.33 / 20 | 74.7 KB | 52.8% | 97 ms | **180 ms** | 32 ms | 2,333 ms | **1.75x / 1.73x** |
| **range_compact_topics20 (This Model)** | 4.00 / 20 | 87.6 KB | 44.7% | 94 ms | **201 ms** | 0 ms | 2,596 ms | **1.57x / 1.56x** |

#### Real Headless Google Chrome (`v155.0.8059.79`) Verification (`Cable-5Mbps`)

Verified against physical `.woff2` files in Headless Google Chrome (`--headless=new --dump-dom`) via the W3C `document.fonts` (`FontFaceSet`) API (`9/9 = 100%` exact slice-set match between Chromium Blink's `unicode-range` engine and `WebsiteFontSimulator`):

| Website Page ID | Strategy | Chrome Loaded Slices | Sim Exact Match | Transfer (KB) | Chrome Local Decode (ms) | E2E Font Ready (ms) | Decode Speedup | E2E Speedup (Bytes Saved) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `lgc_multilingual_eu_news` | `single_slice` | 1 | **Yes** | 158.4 KB | 38.8 ms | **354 ms** | 1.00x | **1.00x** (0.0%) |
| `lgc_multilingual_eu_news` | `equal_bin_20` | 6 | **Yes** | 131.2 KB | 41.7 ms | **313 ms** | 0.93x | **1.13x** (17.2%) |
| `lgc_multilingual_eu_news` | **range_compact_topics20 (This Model)** | 5 | **Yes** | 110.5 KB | 24.0 ms | **261 ms** | 1.62x | **1.36x** (30.2%) |
| `lgc_slavic_hellenic_encyclopedia` | `single_slice` | 1 | **Yes** | 158.4 KB | 60.6 ms | **376 ms** | 1.00x | **1.00x** (0.0%) |
| `lgc_slavic_hellenic_encyclopedia` | `equal_bin_20` | 5 | **Yes** | 109.4 KB | 28.7 ms | **264 ms** | 2.11x | **1.42x** (30.9%) |
| `lgc_slavic_hellenic_encyclopedia` | **range_compact_topics20 (This Model)** | 5 | **Yes** | 110.5 KB | 30.5 ms | **268 ms** | 1.99x | **1.41x** (30.2%) |
| `lgc_scientific_typography` | `single_slice` | 1 | **Yes** | 158.4 KB | 23.9 ms | **339 ms** | 1.00x | **1.00x** (0.0%) |
| `lgc_scientific_typography` | `equal_bin_20` | 1 | **Yes** | 22.7 KB | 23.0 ms | **116 ms** | 1.04x | **2.92x** (85.7%) |
| `lgc_scientific_typography` | **range_compact_topics20 (This Model)** | 2 | **Yes** | 41.7 KB | 17.0 ms | **141 ms** | 1.41x | **2.40x** (73.7%) |

You can test your own HTML files, live URLs, or text snippets against `mlfont-notosans-lgc` before deploying using the `mlfont-simulate` CLI, the Python `WebsiteFontSimulator` API, or the interactive side-by-side **Website Font Loading Simulator** in [`web/demo.html`](https://github.com/wq2012/mlfont/blob/main/web/demo.html):

```bash
# Simulate a local HTML file with mlfont-notosans-lgc across all 7 network profiles:
mlfont-simulate --font_key notosans_lgc --html_file path/to/index.html --network_profile all

# Simulate a live URL and verify in real Headless Google Chrome:
mlfont-simulate --font_key notosans_lgc --url https://example.com --network_profile Cable-5Mbps --headless_chrome
```

## Training Configuration

* **Target Font**: `NotoSans-Regular.ttf` (2,297 Unicode codepoints)
* **Training & Evaluation Corpus**: `wikimedia/wikipedia` (`20231101`, languages: `en, de, fr, pl, ru, el`), 1,500 training articles and 300 uniformly sampled testing articles
* **Slicing Architecture**: Tiered Document-Frequency Slicing + Online Variational LDA & Deep Neural Network Topic Model (DNNTM) + Range-Compact Genetic Algorithm (`mlfont` preset: `lgc`)
* **Total Slices**: 20 slices (384 total CSS `unicode-range` intervals)
* **Physical WOFF2 Bundle Size**: 434.1 KB (444,484 bytes) across 20 `.woff2` files (companion CSS: 7.8 KB)
* **Training Wall-Time**: 24.0 seconds on CPU/GPU

## Model Files Included

* `font_sliced_cdn.css`: Zero-hosting CDN `@font-face` stylesheet (`font-family: 'Noto Sans Sliced'`) streaming `.woff2` slices directly from Hugging Face CDN
* `font_sliced.css`: Self-hosted `@font-face` stylesheet (`font-family: 'NotoSans-Regular Sliced'`) with compact `unicode-range` descriptors
* `mlfont.js`: Browser JS loader (`MLFont.load`) & HTML5 Performance API slice latency telemetry library (`MLFont.getMetrics` / `window.GoogleFonts`)
* `woff2/*.woff2`: 20 physical WOFF2 font slices generated via `fontTools.subset`
* `nam/*.nam`: 20 Google Fonts `.nam` codepoint subset files
* `model.safetensors`: Learned topic-codepoint weight matrices (`lambda_matrix`, `dnntm_lambda_matrix`), `codepoints`, `remaining_codepoints`, `frequency_array`, and `dna_assignment` in SafeTensors format
* `strategy.json`: Range-Compact GA `SlicingStrategy` definition and hyperparameters
* `olda_strategy.json`: Online LDA + Greedy `SlicingStrategy` definition
* `dnntm_strategy.json`: DNNTM (Entropy Difference Loss) + Greedy `SlicingStrategy` definition
* `test_samples.json`: Exact held-out evaluation codepoint sets (300 test pages)
* `config.json`: Model and font bundle summary metadata (including per-slice WOFF2 byte sizes)
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

## Python API Usage (ML Researchers & Custom Font Slicing)

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
