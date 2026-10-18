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

## Overview

Pretrained **MLFont** machine-learning-guided font slicing model and WOFF2 slice bundle for **Noto Sans KR (Korean)** (`NotoSansKR-Regular.otf`, 22,853 Unicode codepoints partitioned into 52 range-compact slices).

* **Paper**: *MLFont: Machine Learning-Driven Font Subsetting and Optimal Unicode-Range Slicing for Low-Latency Web Typography* (Quan Wang and Yiran Mao)
* **Background & References**: Open-source reproduction of Google's machine learning font slicing technology ([Official Post](https://x.com/googlefonts/status/900019727732531200), [Defensive Publication](https://www.tdcommons.org/dpubs_series/906/))
* **Open-Source Repository**: https://github.com/wq2012/mlfont
* **PyPI Package**: https://pypi.org/project/mlfont/

## Using This Sliced Font in Your Website (Frontend Developer Guide)

While basic Latin webfonts are typically 20–100 KB, Pan-Unicode and East Asian fonts like **Noto Sans KR (Korean)** (`NotoSansKR-Regular.otf`) are much larger (5.29 MB / 5419.9 KB (5,549,952 bytes) in WOFF2) because they include **22,853 characters**. In any large corpus of documents, there are strong co-occurrence patterns in character usage. By scanning webpages in `ko`, **MLFont** models which characters are most likely to appear together, informing a static subset slicing strategy that sorts all 22,853 glyphs into **52 slices** (6,769 compact CSS `unicode-range` intervals).

Using this method, your entire web page renders in **Noto Sans KR (Korean)** while the browser only downloads the small subset of `.woff2` font slices required for the characters present on that page—which is dramatically faster than loading the whole font. Cross-page and cross-site browser caching further increases the latency benefits over time. *(Note: The slicing optimization works automatically in all modern [browsers that support CSS `unicode-range`](https://caniuse.com/font-unicode-range).)*

### 1. Link to Sliced Font CSS (HTML `<link>` or CSS `@import`)

Add a `<link>` element to the `<head>` section of your HTML:

```html
<link rel="stylesheet"
      href="https://cdn.jsdelivr.net/gh/wq2012/mlfont@main/pretrained_models/mlfont-notosanskr/font_sliced_cdn.css">
```

Or in your CSS / SCSS stylesheet:

```css
@import url('https://cdn.jsdelivr.net/gh/wq2012/mlfont@main/pretrained_models/mlfont-notosanskr/font_sliced_cdn.css');
```

### 2. Designate Font Family in CSS

Style your text with the `'Noto Sans KR Sliced'` family in your CSS:

```css
body {
  font-family: 'Noto Sans KR Sliced', 'NotoSansKR-Regular Sliced', Roboto, 'Noto Sans', sans-serif;
}
```

*(Note: `font_sliced_cdn.css` streams the `.woff2` slices on-demand directly from Hugging Face's global CDN `https://huggingface.co/wq2012/mlfont-notosanskr/resolve/main/woff2/...` with `font-display: swap` and CORS `Access-Control-Allow-Origin: *`, so you do not need to host the `.woff2` files yourself. To **self-host** on your own server/CDN, download this repository and link `font_sliced.css` alongside the `woff2/` folder.)*

### 3. Alternative: One-Line Drop-In JavaScript Loader (`mlfont.js`)

You can also load the sliced stylesheet directly from Hugging Face CDN using a single `<script>` tag with `data-mlfont`:

```html
<script src="https://cdn.jsdelivr.net/gh/wq2012/mlfont@main/web/mlfont.js"
        data-mlfont="notosanskr"
        async></script>
```

Or programmatically in JavaScript:

```javascript
MLFont.load('notosanskr');
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

Across **3 real-world sample websites** for **Noto Sans KR (Korean)** (full unsliced WOFF2: `3,032.4 KB`), **`range_compact_most300_300_topics50`** downloads an average of only **`5.67 / 52`** slices (`461.2 KB` vs. `3,048.7 KB` unsliced, saving **`84.9%`** bandwidth). On `Cable-5Mbps` (`5 Mbps`, `28 ms` RTT), this achieves a **`6.06x`** cold-cache load speedup (**`836 ms`** vs. `5,059 ms` for `single_slice`), with **First Text Paint at `287 ms`** and **Warm-Session Load at `97 ms`** (and **`6.09x`** speedup on `GIN-3g`: **`10,394 ms`** vs. `63,246 ms`):

| Strategy | Slices Dl / Total | Transfer (KB) | Bytes Saved | Cable-5Mbps 1st Paint (ms) | Cable-5Mbps Cold Load (ms) | Cable-5Mbps Warm Load (ms) | GIN-3g Cold Load (ms) | Cold Speedup (Cable / 3G) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `single_slice` | 1.00 / 1 | 3,048.7 KB | 0.0% | 5,059 ms | **5,059 ms** | 0 ms | 63,246 ms | **1.00x / 1.00x** |
| `equal_bin_52` | 24.00 / 52 | 2,516.7 KB | 17.4% | 1,119 ms | **4,274 ms** | 99 ms | 53,552 ms | **1.18x / 1.18x** |
| `most_300_300_equal_bin_50` | 8.33 / 52 | 795.5 KB | 73.9% | 277 ms | **1,400 ms** | 273 ms | 17,504 ms | **3.61x / 3.61x** |
| `language_slice` | 3.00 / 10 | 4,046.6 KB | -32.7% | 6,695 ms | **6,695 ms** | 0 ms | 83,682 ms | **0.76x / 0.76x** |
| `online_lda_topics50_greedy` | 5.33 / 52 | 488.3 KB | 84.0% | 296 ms | **873 ms** | 101 ms | 10,818 ms | **5.79x / 5.85x** |
| `dnntm_topics50_greedy` | 4.00 / 52 | 347.3 KB | 88.6% | 296 ms | **642 ms** | 0 ms | 7,929 ms | **7.88x / 7.98x** |
| **range_compact_most300_300_topics50 (This Model)** | 5.67 / 52 | 461.2 KB | 84.9% | 287 ms | **836 ms** | 97 ms | 10,394 ms | **6.06x / 6.09x** |

#### Real Headless Chrome (`v155.0.8059.79`) Verification (`Cable-5Mbps`)

Verified against physical `.woff2` files in Headless Chrome (`--headless=new --dump-dom`) via the W3C `document.fonts` (`FontFaceSet`) API (`9/9 = 100.0%` exact slice-set match between Chromium Blink's `unicode-range` engine and `WebsiteFontSimulator`).

| Website Page ID | Strategy | Chrome Loaded Slices | Sim Exact Match | Transfer (KB) | Chrome Local Decode (ms) | E2E Font Ready (ms) | Decode Speedup | E2E Speedup (Bytes Saved) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `kr_cloud_engineering` | `single_slice` | 1 | **Yes** | 3,048.7 KB | 147.4 ms | **5,198 ms** | 1.00x | **1.00x** (0.0%) |
| `kr_cloud_engineering` | `equal_bin_52` | 24 | **Yes** | 2,516.7 KB | 74.4 ms | **4,338 ms** | 1.98x | **1.20x** (17.4%) |
| `kr_cloud_engineering` | **range_compact_most300_300_topics50 (This Model)** | 6 | **Yes** | 481.9 KB | 43.5 ms | **889 ms** | 3.39x | **5.85x** (84.2%) |
| `kr_kpop_entertainment` | `single_slice` | 1 | **Yes** | 3,048.7 KB | 140.5 ms | **5,191 ms** | 1.00x | **1.00x** (0.0%) |
| `kr_kpop_entertainment` | `equal_bin_52` | 24 | **Yes** | 2,516.7 KB | 95.6 ms | **4,359 ms** | 1.47x | **1.19x** (17.4%) |
| `kr_kpop_entertainment` | **range_compact_most300_300_topics50 (This Model)** | 7 | **Yes** | 582.9 KB | 43.3 ms | **1,082 ms** | 3.24x | **4.80x** (80.9%) |
| `kr_economy_finance` | `single_slice` | 1 | **Yes** | 3,048.7 KB | 138.3 ms | **5,189 ms** | 1.00x | **1.00x** (0.0%) |
| `kr_economy_finance` | `equal_bin_52` | 24 | **Yes** | 2,516.7 KB | 75.7 ms | **4,339 ms** | 1.83x | **1.20x** (17.4%) |
| `kr_economy_finance` | **range_compact_most300_300_topics50 (This Model)** | 4 | **Yes** | 318.9 KB | 37.9 ms | **616 ms** | 3.65x | **8.42x** (89.5%) |

You can test your own HTML files, live URLs, or text snippets against `mlfont-notosanskr` before deploying using the `mlfont-simulate` CLI, the Python `WebsiteFontSimulator` API, or the interactive side-by-side **Website Font Loading Simulator** in [`web/demo.html`](https://github.com/wq2012/mlfont/blob/main/web/demo.html):

```bash
# Simulate a local HTML file with mlfont-notosanskr across all 7 network profiles:
mlfont-simulate --font_key notosanskr --html_file path/to/index.html --network_profile all

# Simulate a live URL and verify in real Headless Chrome:
mlfont-simulate --font_key notosanskr --url https://example.com --network_profile Cable-5Mbps --headless_chrome
```

## Training Configuration

* **Target Font**: `NotoSansKR-Regular.otf` (22,853 Unicode codepoints)
* **Training & Evaluation Corpus**: `wikimedia/wikipedia` (`20231101`, languages: `ko`), 1,600 training articles and 400 uniformly sampled testing articles
* **Slicing Architecture**: Tiered Document-Frequency Slicing + Online Variational LDA & Deep Neural Network Topic Model (DNNTM) + Range-Compact Genetic Algorithm (`mlfont` preset: `korean`)
* **Total Slices**: 52 slices (6,769 total CSS `unicode-range` intervals)
* **Physical WOFF2 Bundle Size**: 5.29 MB / 5419.9 KB (5,549,952 bytes) across 52 `.woff2` files (companion CSS: 78.3 KB). Note: physical `.woff2` slicing replicates shared OpenType tables (`cmap`, `hmtx`, `loca`, `GSUB`, `GPOS`) into each standalone subset file (increasing static CDN disk footprint by +46.6% to +175.3% vs. a monolithic WOFF2), whereas individual webpages only fetch the small subset of triggered slices (saving 65.4%–84.9% per-page bandwidth on CJK and 44.7% on LGC).
* **Training Wall-Time**: 84.5 seconds on CPU/GPU

## Model Files Included

* `font_sliced_cdn.css`: Zero-hosting CDN `@font-face` stylesheet (`font-family: 'Noto Sans KR Sliced'`) streaming `.woff2` slices directly from Hugging Face CDN
* `font_sliced.css`: Self-hosted `@font-face` stylesheet (`font-family: 'NotoSansKR-Regular Sliced'`, `url(woff2/NotoSansKR-Regular_slice_XXX.woff2)`) with compact `unicode-range` descriptors
* `mlfont.js`: Browser JS loader (`MLFont.load`) & HTML5 Performance API slice latency telemetry library (`MLFont.getMetrics` / `MLFont.printReport`)
* `woff2/NotoSansKR-Regular_slice_XXX.woff2`: 52 physical WOFF2 font slices generated via `fontTools.subset` (plus `woff2/NotoSansKR-Regular_sliced.css`)
* `nam/slice_XXX.nam`: 52 `.nam` codepoint subset files
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
| `single_slice` | 1 | 4598 | 0 / 0 | 250.67 / 1739.45 | 62.73 / 435.34 | 12.53 / 86.92 | 3.31% |
| `equal_bin_52` | 52 | 4635 | 0 / 0 | 259.74 / 1338.41 | 68.09 / 350.87 | 12.67 / 65.29 | 3.76% |
| `random_slice_52` | 52 | 22493 | 0 / 0 | 270.86 / 2029.63 | 71.01 / 532.06 | 13.21 / 99.02 | 3.31% |
| `language_slice` | 10 | 4579 | 34 / 4322 | 203.56 / 1514.81 | 51.20 / 380.39 | 10.15 / 75.57 | 4.22% |
| `most_300_300_equal_bin_50` | 52 | 5608 | 0 / 0 | 197.85 / 715.15 | 51.88 / 187.72 | 9.65 / 34.86 | 3.81% |
| `online_lda_topics50_greedy` | 52 | 8327 | 0 / 0 | 66.59 / 348.80 | 17.48 / 91.73 | 3.25 / 16.99 | 12.03% |
| `dnntm_topics50_greedy` | 52 | 8354 | 0 / 0 | 66.97 / 331.00 | 17.59 / 87.21 | 3.26 / 16.10 | 11.51% |
| **range_compact_most300_300_topics50 (This Model)** | 52 | 6769 | 0 / 0 | 80.22 / 354.26 | 21.06 / 93.00 | 3.91 / 17.27 | 9.25% |

## Python API Usage (ML Researchers & Custom Font Slicing)

Install `mlfont`:

```bash
pip install mlfont huggingface_hub
```

Load the pretrained bundle (Range-Compact GA, Online LDA, and DNNTM) in Python:

```python
import mlfont

bundle = mlfont.load_pretrained("wq2012/mlfont-notosanskr")
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
