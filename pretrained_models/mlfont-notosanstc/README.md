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

## Overview

Pretrained **MLFont** machine-learning-guided font slicing model and WOFF2 slice bundle for **Noto Sans TC (Traditional Chinese)** (`NotoSansTC-Regular.otf`, 20,637 Unicode codepoints partitioned into 82 range-compact slices).

* **Paper**: *MLFont: Machine Learning-Driven Font Subsetting and Optimal Unicode-Range Slicing for Low-Latency Web Typography* (Quan Wang and Yiran Mao)
* **Background & References**: Open-source reproduction of Google's machine learning font slicing technology ([Official Post](https://x.com/googlefonts/status/900019727732531200), [Defensive Publication](https://www.tdcommons.org/dpubs_series/906/))
* **Open-Source Repository**: https://github.com/wq2012/mlfont
* **PyPI Package**: https://pypi.org/project/mlfont/

## Using This Sliced Font in Your Website (Frontend Developer Guide)

While basic Latin webfonts are typically 20–100 KB, Pan-Unicode and East Asian fonts like **Noto Sans TC (Traditional Chinese)** (`NotoSansTC-Regular.otf`) are much larger (5.94 MB / 6085.5 KB (6,231,532 bytes) in WOFF2) because they include **20,637 characters**. In any large corpus of documents, there are strong co-occurrence patterns in character usage. By scanning webpages in `zh`, **MLFont** models which characters are most likely to appear together, informing a static subset slicing strategy that sorts all 20,637 glyphs into **82 slices** (8,743 compact CSS `unicode-range` intervals).

Using this method, your entire web page renders in **Noto Sans TC (Traditional Chinese)** while the browser only downloads the small subset of `.woff2` font slices required for the characters present on that page—which is dramatically faster than loading the whole font. Cross-page and cross-site browser caching further increases the latency benefits over time. *(Note: The slicing optimization works automatically in all modern [browsers that support CSS `unicode-range`](https://caniuse.com/font-unicode-range).)*

### 1. Link to Sliced Font CSS (HTML `<link>` or CSS `@import`)

Add a `<link>` element to the `<head>` section of your HTML:

```html
<link rel="stylesheet"
      href="https://cdn.jsdelivr.net/gh/wq2012/mlfont@main/pretrained_models/mlfont-notosanstc/font_sliced_cdn.css">
```

Or in your CSS / SCSS stylesheet:

```css
@import url('https://cdn.jsdelivr.net/gh/wq2012/mlfont@main/pretrained_models/mlfont-notosanstc/font_sliced_cdn.css');
```

### 2. Designate Font Family in CSS

Style your text with the `'Noto Sans TC Sliced'` family in your CSS:

```css
body {
  font-family: 'Noto Sans TC Sliced', 'NotoSansTC-Regular Sliced', Roboto, 'Noto Sans', sans-serif;
}
```

*(Note: `font_sliced_cdn.css` streams the `.woff2` slices on-demand directly from Hugging Face's global CDN `https://huggingface.co/wq2012/mlfont-notosanstc/resolve/main/woff2/...` with `font-display: swap` and CORS `Access-Control-Allow-Origin: *`, so you do not need to host the `.woff2` files yourself. To **self-host** on your own server/CDN, download this repository and link `font_sliced.css` alongside the `woff2/` folder.)*

### 3. Alternative: One-Line Drop-In JavaScript Loader (`mlfont.js`)

You can also load the sliced stylesheet directly from Hugging Face CDN using a single `<script>` tag with `data-mlfont`:

```html
<script src="https://cdn.jsdelivr.net/gh/wq2012/mlfont@main/web/mlfont.js"
        data-mlfont="notosanstc"
        async></script>
```

Or programmatically in JavaScript:

```javascript
MLFont.load('notosanstc');
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

Across **3 real-world sample websites** for **Noto Sans TC (Traditional Chinese)** (full unsliced WOFF2: `4,047.4 KB`), **`range_compact_most600_600_topics80`** downloads an average of only **`16.00 / 82`** slices (`1,406.8 KB` vs. `4,067.0 KB` unsliced, saving **`65.4%`** bandwidth). On `Cable-5Mbps` (`5 Mbps`, `28 ms` RTT), this achieves a **`2.76x`** cold-cache load speedup (**`2,436 ms`** vs. `6,728 ms` for `single_slice`), with **First Text Paint at `665 ms`** and **Warm-Session Load at `737 ms`** (and **`2.76x`** speedup on `GIN-3g`: **`30,431 ms`** vs. `84,100 ms`):

| Strategy | Slices Dl / Total | Transfer (KB) | Bytes Saved | Cable-5Mbps 1st Paint (ms) | Cable-5Mbps Cold Load (ms) | Cable-5Mbps Warm Load (ms) | GIN-3g Cold Load (ms) | Cold Speedup (Cable / 3G) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `single_slice` | 1.00 / 1 | 4,067.0 KB | 0.0% | 6,728 ms | **6,728 ms** | 0 ms | 84,100 ms | **1.00x / 1.00x** |
| `equal_bin_82` | 52.00 / 82 | 3,881.7 KB | 4.6% | 834 ms | **6,662 ms** | 589 ms | 83,644 ms | **1.01x / 1.01x** |
| `most_600_600_equal_bin_80` | 22.67 / 82 | 1,833.9 KB | 54.9% | 648 ms | **3,170 ms** | 1,978 ms | 39,707 ms | **2.12x / 2.12x** |
| `language_slice` | 3.00 / 10 | 2,287.2 KB | 43.8% | 3,814 ms | **3,814 ms** | 0 ms | 47,652 ms | **1.76x / 1.76x** |
| `online_lda_topics80_greedy` | 16.33 / 82 | 1,394.0 KB | 65.7% | 681 ms | **2,430 ms** | 616 ms | 30,307 ms | **2.77x / 2.77x** |
| `dnntm_topics80_greedy` | 16.00 / 82 | 1,369.1 KB | 66.3% | 678 ms | **2,389 ms** | 616 ms | 29,796 ms | **2.82x / 2.82x** |
| **range_compact_most600_600_topics80 (This Model)** | 16.00 / 82 | 1,406.8 KB | 65.4% | 665 ms | **2,436 ms** | 737 ms | 30,431 ms | **2.76x / 2.76x** |

#### Real Headless Chrome (`v155.0.8059.79`) Verification (`Cable-5Mbps`)

Verified against physical `.woff2` files in Headless Chrome (`--headless=new --dump-dom`) via the W3C `document.fonts` (`FontFaceSet`) API (`9/9 = 100%` exact slice-set match between Chromium Blink's `unicode-range` engine and `WebsiteFontSimulator`):

| Website Page ID | Strategy | Chrome Loaded Slices | Sim Exact Match | Transfer (KB) | Chrome Local Decode (ms) | E2E Font Ready (ms) | Decode Speedup | E2E Speedup (Bytes Saved) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `tc_taipei_tech_portal` | `single_slice` | 1 | **Yes** | 4,067.0 KB | 172.4 ms | **6,892 ms** | 1.00x | **1.00x** (0.0%) |
| `tc_taipei_tech_portal` | `equal_bin_82` | 51 | **Yes** | 3,807.7 KB | 104.3 ms | **6,623 ms** | 1.65x | **1.04x** (6.4%) |
| `tc_taipei_tech_portal` | **range_compact_most600_600_topics80 (This Model)** | 12 | **Yes** | 1,103.5 KB | 54.2 ms | **1,946 ms** | 3.18x | **3.54x** (72.9%) |
| `tc_culture_literature` | `single_slice` | 1 | **Yes** | 4,067.0 KB | 165.4 ms | **6,885 ms** | 1.00x | **1.00x** (0.0%) |
| `tc_culture_literature` | `equal_bin_82` | 55 | **Yes** | 4,104.1 KB | 105.2 ms | **7,137 ms** | 1.57x | **0.96x** (-0.9%) |
| `tc_culture_literature` | **range_compact_most600_600_topics80 (This Model)** | 16 | **Yes** | 1,389.4 KB | 54.7 ms | **2,443 ms** | 3.02x | **2.82x** (65.8%) |
| `tc_gastronomy_guide` | `single_slice` | 1 | **Yes** | 4,067.0 KB | 175.8 ms | **6,895 ms** | 1.00x | **1.00x** (0.0%) |
| `tc_gastronomy_guide` | `equal_bin_82` | 50 | **Yes** | 3,733.4 KB | 125.8 ms | **6,523 ms** | 1.40x | **1.06x** (8.2%) |
| `tc_gastronomy_guide` | **range_compact_most600_600_topics80 (This Model)** | 20 | **Yes** | 1,727.6 KB | 62.3 ms | **3,033 ms** | 2.82x | **2.27x** (57.5%) |

You can test your own HTML files, live URLs, or text snippets against `mlfont-notosanstc` before deploying using the `mlfont-simulate` CLI, the Python `WebsiteFontSimulator` API, or the interactive side-by-side **Website Font Loading Simulator** in [`web/demo.html`](https://github.com/wq2012/mlfont/blob/main/web/demo.html):

```bash
# Simulate a local HTML file with mlfont-notosanstc across all 7 network profiles:
mlfont-simulate --font_key notosanstc --html_file path/to/index.html --network_profile all

# Simulate a live URL and verify in real Headless Chrome:
mlfont-simulate --font_key notosanstc --url https://example.com --network_profile Cable-5Mbps --headless_chrome
```

## Training Configuration

* **Target Font**: `NotoSansTC-Regular.otf` (20,637 Unicode codepoints)
* **Training & Evaluation Corpus**: `wikimedia/wikipedia` (`20231101`, languages: `zh`), 1,600 training articles and 400 uniformly sampled testing articles
* **Slicing Architecture**: Tiered Document-Frequency Slicing + Online Variational LDA & Deep Neural Network Topic Model (DNNTM) + Range-Compact Genetic Algorithm (`mlfont` preset: `traditional_chinese`)
* **Total Slices**: 82 slices (8,743 total CSS `unicode-range` intervals)
* **Physical WOFF2 Bundle Size**: 5.94 MB / 6085.5 KB (6,231,532 bytes) across 82 `.woff2` files (companion CSS: 104.1 KB)
* **Training Wall-Time**: 25.0 seconds on CPU/GPU

## Model Files Included

* `font_sliced_cdn.css`: Zero-hosting CDN `@font-face` stylesheet (`font-family: 'Noto Sans TC Sliced'`) streaming `.woff2` slices directly from Hugging Face CDN
* `font_sliced.css`: Self-hosted `@font-face` stylesheet (`font-family: 'NotoSansTC-Regular Sliced'`) with compact `unicode-range` descriptors
* `mlfont.js`: Browser JS loader (`MLFont.load`) & HTML5 Performance API slice latency telemetry library (`MLFont.getMetrics` / `MLFont.printReport`)
* `woff2/*.woff2`: 82 physical WOFF2 font slices generated via `fontTools.subset`
* `nam/*.nam`: 82 `.nam` codepoint subset files
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
| `single_slice` | 1 | 5036 | 0 / 0 | 125.58 / 704.12 | 31.43 / 176.24 | 6.28 / 35.18 | 7.88% |
| `equal_bin_82` | 82 | 5098 | 0 / 0 | 115.29 / 689.78 | 31.03 / 185.64 | 5.54 / 33.17 | 9.67% |
| `random_slice_82` | 82 | 20436 | 0 / 0 | 144.57 / 922.42 | 38.91 / 248.25 | 6.95 / 44.36 | 7.88% |
| `language_slice` | 10 | 5767 | 126 / 11699 | 52.25 / 282.95 | 13.23 / 71.40 | 2.60 / 14.08 | 18.89% |
| `most_600_600_equal_bin_80` | 82 | 6672 | 0 / 0 | 92.66 / 446.08 | 24.90 / 119.91 | 4.46 / 21.47 | 9.83% |
| `online_lda_topics80_greedy` | 82 | 11867 | 0 / 0 | 50.11 / 250.61 | 13.43 / 67.58 | 2.42 / 12.04 | 19.59% |
| `dnntm_topics80_greedy` | 82 | 11410 | 0 / 0 | 51.65 / 226.65 | 13.84 / 60.72 | 2.49 / 10.93 | 19.30% |
| **range_compact_most600_600_topics80 (This Model)** | 82 | 8743 | 0 / 0 | 56.19 / 265.70 | 15.07 / 71.62 | 2.71 / 12.79 | 17.37% |

## Python API Usage (ML Researchers & Custom Font Slicing)

Install `mlfont`:

```bash
pip install mlfont huggingface_hub
```

Load the pretrained bundle (Range-Compact GA, Online LDA, and DNNTM) in Python:

```python
import mlfont

bundle = mlfont.load_pretrained("wq2012/mlfont-notosanstc")
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
