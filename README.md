# MLFont: Machine Learning-Driven Font Subsetting and Optimal Unicode-Range Slicing for Low-Latency Web Typography

[![Build Status](https://github.com/wq2012/mlfont/workflows/Python%20application/badge.svg)](https://github.com/wq2012/mlfont/actions)
[![PyPI Version](https://img.shields.io/pypi/v/mlfont.svg)](https://pypi.org/project/mlfont/)
[![Python Versions](https://img.shields.io/pypi/pyversions/mlfont.svg)](https://pypi.org/project/mlfont/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Hugging Face Models](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Pretrained%20Models%20%26%20WOFF2%20Slices-yellow)](https://huggingface.co/collections/wq2012/mlfont)

**MLFont** (`mlfont`) is a standalone, pure-Python and PyTorch library for machine learning-driven web font subsetting (`@font-face` `unicode-range` slicing). It solves the **Heavy Web Font Loading Problem** for large Pan-Unicode and CJK (Chinese, Japanese, Korean) fonts—which contain 12,000 to over 44,000 glyphs and weigh 4 MB to 20 MB—by learning codepoint co-occurrence structure across real-world web documents and jointly optimizing network download latency, session cache utilization, and CSS `unicode-range` compactness.

> **Disclaimer:** This is not an officially supported Google product.

---

## Table of Contents

1. [Overview and Motivation](#overview-and-motivation)
2. [Open-Source Reproduction Notice](#open-source-reproduction-notice)
3. [System Architecture and Visual Overview](#system-architecture-and-visual-overview)
4. [Algorithmic Pipeline](#algorithmic-pipeline)
5. [Installation](#installation)
6. [Quickstart: Python API](#quickstart-python-api)
7. [Command-Line Interface (CLI) and Reproducible Pipelines](#command-line-interface-cli-and-reproducible-pipelines)
8. [Pretrained Models and Sliced Noto WOFF2 Bundles](#pretrained-models-and-sliced-noto-woff2-bundles)
9. [Original Production Study and Google Fonts Deployment Results](#original-production-study-and-google-fonts-deployment-results)
10. [Open-Source Reproduction Benchmarks](#open-source-reproduction-benchmarks)
11. [Ablation Studies](#ablation-studies)
12. [Testing, Code Quality, and PyPI Packaging](#testing-code-quality-and-pypi-packaging)
13. [Citations](#citations)

---

## Overview and Motivation

Modern web typography relies on CSS3 `@font-face` declarations with `unicode-range` descriptors. When a browser renders a web page, it inspects the text characters present in the DOM and only downloads the `.woff2` font slice files whose `unicode-range` intervals intersect the page's character set.

While Latin fonts contain a few hundred glyphs (~20 KB in WOFF2), Pan-European (Latin-Greek-Cyrillic, **LGC**) and East Asian (**CJK**: Chinese, Japanese, Korean) fonts such as **Noto Sans CJK** contain **12,254 to 44,683 codepoints** and weigh **4 MB to 16 MB**. Naive loading strategies fail in two opposite directions:

- **Single-File Delivery (`single_slice`)**: Transfers megabytes of unused glyphs over the network, causing multi-second render-blocking delays (Flash of Invisible Text / Flash of Unstyled Text) and less than 5% glyph utilization.
- **Naive Equal-Bin or Random Slicing (`equal_bin`, `random_slice`)**: Either triggers dozens of concurrent HTTP Round-Trip Time (RTT) requests per page or explodes the CSS stylesheet with tens of thousands of fragmented `U+XXXX` hex intervals.
- **Static Language/Script Slicing (`language_slice`)**: Fails on CJK scripts (where a single script block contains over 20,000 Han ideographs) and suffers from missing-glyph fallback whenever multilingual pages mix scripts or symbols.

**MLFont** formulates font slicing as a two-tier machine learning and combinatorial optimization problem:
1. **Tier 1 — High-Frequency Deterministic Slices (`most_1 .. most_M`)**: Exploits the steep Zipfian decay of character document frequencies by isolating the top most frequent characters into 1–2 small, universally cached slices (`most=[1000, 500]` for CJK SC).
2. **Tier 2 — Latent Co-Occurrence Topic Modeling (`Online LDA` & `DNNTM`)**: Models document-character co-occurrence across the remaining tail codepoints using either Variational Bayes Online Latent Dirichlet Allocation (`OnlineLDASolver`) or a Deep Neural Network Topic Model (`DNNTMSolver` — a nonnegative autoencoder with L1 sparsity and Frobenius reconstruction loss) followed by column/row affinity normalization.
3. **Tier 3 — Range-Compact Genetic Algorithm (`RangeCompactGASolver`)**: Refines the topic-to-slice assignment matrix via a multi-objective evolutionary algorithm with a power-law curriculum constraint `max_diff(t)` that merges adjacent Unicode codepoints into contiguous `[U+start-end]` intervals, slashing CSS rule bloat while preserving low per-glyph network latency.
4. **Tier 4 — Physical WOFF2 Subsetting & Network Session Simulator (`FontSubsetter` & `FontLoadingEvaluator`)**:Slices real `.ttf` / `.otf` fonts via `fontTools` + Brotli into production-ready `.woff2` files and `@font-face` CSS stylesheets, and evaluates them across realistic multi-page browsing sessions (`GIN-2g`, `GIN-3g`, `DSL`, `GIN-3g_no_session`, `GIN-3g_short_session`).

---

## Open-Source Reproduction Notice

This repository provides a clean-room, open-source Python/PyTorch reproduction of the **MLFont (`mlfontloading`)** system originally developed at Google (2016–2017) by **Quan Wang and Yiran Mao**, which powered the `Noto Sans SC Sliced` Early Access deployment on Google Fonts (`fonts.googleapis.com/earlyaccess/notosansscsliced.css`, peaking at over **4.05 million daily CSS requests**).

To ensure 100% reproducibility without proprietary dependencies, this implementation adapts the original internal pipeline as follows:

| Component | Original Internal Implementation (2016–2017) | Open-Source Reproduction (`mlfont`) |
| :--- | :--- | :--- |
| **Execution Engine** | Internal MapReduce / Flume C++ & Python 2.7 pipelines | Standalone Python 3.10+ library with NumPy, SciPy, Scikit-Learn, and PyTorch |
| **Training Corpus** | Internal Google web index crawls (`DocJoins`) | Public **Wikimedia / Wikipedia** dumps (`wikimedia/wikipedia`) across `en, de, fr, pl, ru, el, ko, ja, zh` |
| **Topic Models** | Online LDA (`gensim` / custom EM) & TensorFlow 1.x `DNNTM` | Clean-room `OnlineLDASolver` (`sklearn` + custom normalization) & PyTorch `DNNTMSolver` |
| **Combinatorial Optimizer** | Range-Compact Genetic Algorithm (`RangeCompactModel`) | Vectorized `RangeCompactGASolver` with power-law curriculum `max_diff(t)` and contiguous block crossover/mutation |
| **Font Subsetting** | Internal `fontTools` + Google Fonts serving definitions | Open-source `fontTools.subset` (`woff2` Brotli compression) + standard `@font-face` CSS & `.nam` exports |
| **Model Serialization** | Internal `cPickle` / `sstable` checkpoints | Safe, zero-code-execution **Hugging Face `safetensors`** (`model.safetensors`) + JSON strategies |

---

## System Architecture and Visual Overview

```
+-----------------------------------------------------------------------------------+
|                        1. INPUT FONT & MULTILINGUAL CORPUS                        |
|  Font Binary (.ttf / .otf)                 Open Web / Wikipedia Corpus (JSONL)    |
|  Extract cmap -> V valid codepoints        Extract per-page unique codepoint sets |
+-------------------+-------------------------------------------+-------------------+
                    |                                           |
                    v                                           v
+---------------------------------------+   +---------------------------------------+
|   2. TIER-1 FREQUENCY ISOLATION       |   |   3. TIER-2 TOPIC CO-OCCURRENCE       |
|  Compute Document Frequency DF(c)     |   |  Filter out Tier-1 codepoints         |
|  Partition top codepoints into        |   |  Learn K x V_rem affinity matrix Phi: |
|  deterministic slices:                |   |   - OnlineLDASolver (Variational EM)  |
|   most_1 (e.g. top 1,000 cps)         |   |   - DNNTMSolver (Nonnegative AE + L1) |
|   most_2 (e.g. next 500 cps)          |   |  Normalize via global_max / spectral  |
+-------------------+-------------------+   +-------------------+-------------------+
                    |                                           |
                    +---------------------+---------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|              4. TIER-3 RANGE-COMPACT GENETIC ALGORITHM (RangeCompactGA)           |
|  Jointly minimize Topic Discrepancy Cost C_topic(S) & CSS Range Cost C_range(S)   |
|  under power-law slice-capacity curriculum:                                       |
|     max_diff(t) = N_rem * (1 - (t / T)^(1 / p))                                   |
|  Output: Hard partition of all V codepoints into M + K disjoint slices            |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|         5. PHYSICAL WOFF2 SUBSETTING, CSS GENERATION & SESSION EVALUATION         |
|  - Slice .ttf/.otf -> .woff2 via fontTools + Brotli & emit @font-face CSS         |
|  - Simulate multi-page browser sessions (L=10 pages) across GIN-2g, GIN-3g, DSL   |
+-----------------------------------------------------------------------------------+
```

### Zipfian Decay and the Latency vs. CSS Compactness Pareto Frontier

![Zipfian Codepoint Decay and Pareto Frontier](resources/fig1_architecture.png)

*Figure 1: (a) Empirical Zipfian decay of codepoint document frequencies across five Google Noto fonts on Wikipedia corpora, motivating the Tier-1 high-frequency slices (`most_1..M`). (b) Pareto trade-off between mean per-glyph download latency (`GIN-3g`) and total CSS `unicode-range` intervals on `Noto Sans SC` (30,759 codepoints).*

---

## Algorithmic Pipeline

### 1. Problem Formulation and Two-Tier Decomposition

Let a target font file (e.g., `NotoSansSC-Regular.otf`) support a vocabulary of `V` valid Unicode codepoints `C = {c_1, c_2, ..., c_V}` extracted from its OpenType `cmap` table. Given a training corpus of `D` web documents, each document `d` is represented as a binary indicator vector `x_d in {0, 1}^V`, where `x_{d,v} = 1` if codepoint `c_v` appears at least once in document `d`.

Because character frequencies follow a steep Zipfian power law, we first compute the document frequency `DF(c_v) = sum_{d=1}^D x_{d,v}` for every codepoint. Given a Tier-1 configuration `most = (m_1, m_2, ..., m_M)`, we sort `C` in descending order of `DF(c_v)` and assign the top `m_1` codepoints to slice `S_1`, the next `m_2` codepoints to slice `S_2`, and so on. The remaining `V_rem = V - sum_{j=1}^M m_j` tail codepoints `C_rem` are passed to Tier-2 topic modeling.

### 2. Online LDA and Affinity Matrix Normalization (`mlfont.olda` & `mlfont.maths`)

`OnlineLDASolver` models each document's active tail codepoints as a bag-of-characters generated from `K` latent topics using online variational Bayes (`LatentDirichletAllocation` with `learning_method="online"`). Let `Lambda` in `R_+^{K x V_rem}` denote the learned unnormalized topic-codepoint variational parameters. Before assigning codepoints to slices, `mlfont.maths.normalize_topic_matrix` applies one of four normalization schemes:

- **`topic_sum`**: Row-normalizes each topic so that `sum_v Phi_{k,v} = 1`.
- **`row_sum`**: Column-normalizes each codepoint across topics so that `sum_k Phi_{k,v} = 1`.
- **`global_max`** *(Default)*: Divides each topic row by its L2 norm and scales by the global maximum entry, preventing sparse outlier topics from dominating greedy or evolutionary assignment.
- **`spectral_norm`**: Applies bi-stochastic Sinkhorn-like row and column scaling (`D_r^{-1/2} Lambda D_c^{-1/2}`) to balance topic capacities.

In greedy mode (`online_lda_topicsK_greedy`), each tail codepoint `v` is assigned to `k*(v) = argmax_k Phi_{k,v}` (with unused zero-frequency codepoints distributed across balanced overflow bins).

### 3. Deep Neural Network Topic Model — DNNTM (`mlfont.dnntm`)

`DNNTMSolver` replaces Dirichlet-multinomial assumptions with a direct neural co-occurrence factorization. Given the document-codepoint matrix `X` in `{0, 1}^{D x V_rem}`, DNNTM parameterizes a nonnegative encoder `W_1` in `R_+^{V_rem x K}` and nonnegative topic-codepoint decoder `W_2` in `R_+^{K x V_rem}` (enforced via `ReLU` / `Softplus` projection after each Adam step) and minimizes the regularized Frobenius reconstruction objective:

```
L_DNNTM(W_1, W_2) = (1 / D) * || X - ReLU(X * W_1) * W_2 ||_F^2
                    + lambda_1 * (|| W_1 ||_1 + || W_2 ||_1)
                    + lambda_2 * (|| W_1 ||_F^2 + || W_2 ||_F^2)
```

where `lambda_1` (`l1_reg`) induces sparse, localized character clusters and `lambda_2` (`l2_reg`) stabilizes weight magnitudes. After training, the decoder matrix `W_2` serves as the `K x V_rem` topic-codepoint affinity matrix `Phi`.

### 4. Range-Compact Genetic Algorithm (`mlfont.range_compact`)

While greedy `argmax_k Phi_{k,v}` assignment maximizes topic coherence, adjacent Unicode codepoints (`c_v` and `c_{v+1} = c_v + 1`) are frequently assigned to different slices, creating thousands of fragmented single-codepoint CSS `unicode-range` entries (`U+4E00, U+4E02, ...`).

`RangeCompactGASolver` evolves a population of `P` candidate slice assignment vectors `s in {0, ..., K-1}^{V_rem}` (ordered by Unicode codepoint value) to jointly minimize two normalized costs:

- **Topic Discrepancy Cost `C_topic(s)`**:
  ```
  C_topic(s) = 1.0 - (1 / V_rem) * sum_{v=1}^{V_rem} (Phi_{s_v, v} / max_k Phi_{k, v})
  ```
- **CSS Range Fragmentation Cost `C_range(s)`**:
  ```
  C_range(s) = (1 / V_rem) * sum_{k=0}^{K-1} NumContiguousUnicodeIntervals(s == k)
  ```

To prevent the evolutionary search from collapsing all codepoints into a single giant slice, generation `t` (`0 <= t <= T`) enforces a **power-law curriculum constraint** on maximum slice-size deviation from the uniform target `N_target = V_rem / K`:

```
max_diff(t) = V_rem * (1.0 - (t / T)^(1.0 / p))
```

where `p > 0` (`decay_power`, default `2.0`) controls the annealing schedule, and specialized contiguous-run crossover and neighborhood smoothing mutation operators merge isolated codepoints into adjacent runs while respecting `|SliceSize(k) - N_target| <= max_diff(t)`.

![Range-Compact Genetic Algorithm Convergence](resources/fig2_ga_convergence.png)

*Figure 2: Convergence of `RangeCompactGASolver` across all five Noto fonts over 150 generations: (a) relative CSS range fragmentation cost `C_range(t) / C_range(0)`, (b) topic discrepancy cost `C_topic(t)`, and (c) power-law slice-capacity curriculum `max_diff(t)` (`p = 2.0`).*

### 5. Network Session Latency Simulator (`mlfont.evaluator`)

Given a slicing strategy `S = {S_1, ..., S_K}` with compressed WOFF2 byte sizes `{B_1, ..., B_K}` (or exact glyph-proportional byte estimates) and a network profile `(RTT_ms, BW_bytes_per_ms, SessionLength L)`, `FontLoadingEvaluator` groups test documents into multi-page browsing sessions of length `L`. Within each session:

1. **Browser Cache Persistence**: A slice `S_k` is downloaded over the network on the first page in the session that contains any codepoint `c in S_k`; subsequent pages in the same session hit the browser cache at `0 ms` cost.
2. **Per-Page Network Latency**: If page `i` triggers the download of a newly required slice set `U_i`, the page incurs:
   - `T_RTT(i) = |U_i| * RTT_ms`
   - `T_download(i) = (sum_{k in U_i} B_k) / BW_bytes_per_ms`
   - `T_per_glyph(i) = (T_RTT(i) + T_download(i)) / |PageGlyphs(i)|`
3. **Upfront and Session Utilization**: Measures the percentage of downloaded glyphs actually rendered on the first page (`upfront_utilization_pct`) and across the entire `L`-page session (`session_utilization_pct`).

| Network Profile Name | Round-Trip Time (`RTT`) | Bandwidth (`bytes/ms` / Mbps) | Session Length (`L`) | Target Scenario |
| :--- | :---: | :---: | :---: | :--- |
| **`GIN-2g`** | `300 ms` | `35 bytes/ms` (`280 kbps`) | `10 pages` | Emerging-market 2G mobile connection |
| **`GIN-3g`** *(Primary)* | `100 ms` | `140 bytes/ms` (`1.12 Mbps`) | `10 pages` | Standard 3G mobile connection |
| **`DSL`** | `20 ms` | `2,000 bytes/ms` (`16.0 Mbps`) | `10 pages` | Fixed broadband / desktop connection |
| **`GIN-3g_no_session`** | `100 ms` | `140 bytes/ms` (`1.12 Mbps`) | `1 page` | Cold-cache single-page bounce (`L = 1`) |
| **`GIN-3g_short_session`** | `100 ms` | `140 bytes/ms` (`1.12 Mbps`) | `3 pages` | Short browsing session (`L = 3`) |

---

## Installation

### Install from PyPI

```bash
pip install mlfont
```

### Install from Source (Development & Benchmark Reproduction)

```bash
git clone https://github.com/wq2012/mlfont.git
cd mlfont
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install -e .
```

---

## Quickstart: Python API

Below is a complete, self-contained example demonstrating how to extract codepoints from a `.ttf`/`.otf` font, train a two-tier `Most + Online LDA + Range-Compact GA` slicing strategy, physically subset the font into `.woff2` slices and `@font-face` CSS, and evaluate per-glyph network latency:

```python
from pathlib import Path
from mlfont import (
    DNNTMConfig,
    DNNTMSolver,
    FontLoadingEvaluator,
    FontSubsetter,
    FrequencySolver,
    OnlineLDAConfig,
    OnlineLDASolver,
    RangeCompactConfig,
    RangeCompactGASolver,
    extract_font_codepoints,
    save_safetensors_model,
)

# 1. Extract valid Unicode codepoints from a TrueType/OpenType font
font_path = Path("fonts/NotoSans-Regular.ttf")
font_codepoints = extract_font_codepoints(font_path)
print(f"Extracted {len(font_codepoints)} valid codepoints from {font_path.name}")

# 2. Prepare training and evaluation documents (lists of codepoint sets or strings)
train_docs = [
    {ord(ch) for ch in page_text if ord(ch) in set(font_codepoints)}
    for page_text in [
        "Machine learning optimizes web font delivery across multilingual pages.",
        "La typographie web moderne utilise unicode-range pour charger les polices.",
        "Η σύγχρονη τυπογραφία ιστού βελτιστοποιεί την καθυστέρηση δικτύου.",
        "Современная веб-типографика минимизирует задержку загрузки шрифтов.",
    ] * 50
]

# 3. Tier-1: Isolate top high-frequency codepoints into deterministic slices
freq_solver = FrequencySolver(font_codepoints)
freq_solver.fit(train_docs)
most_slices, remaining_codepoints = freq_solver.partition_most_frequent(most_counts=[200])

# 4. Tier-2: Fit Online LDA (or DNNTMSolver) on the remaining tail codepoints
olda_solver = OnlineLDASolver(
    codepoints=remaining_codepoints,
    config=OnlineLDAConfig(
        n_topics=10,
        batch_size=64,
        max_iter=15,
        normalization="global_max",
        random_state=42,
    ),
)
topic_matrix = olda_solver.fit_transform(train_docs)

# Optional alternative: Deep Neural Network Topic Model (DNNTM)
dnntm_solver = DNNTMSolver(
    codepoints=remaining_codepoints,
    config=DNNTMConfig(n_topics=10, epochs=30, l1_reg=1e-4, learning_rate=1e-2),
)
dnntm_matrix = dnntm_solver.fit_transform(train_docs)

# 5. Tier-3: Optimize CSS unicode-range compactness via Range-Compact GA
ga_solver = RangeCompactGASolver(
    codepoints=remaining_codepoints,
    topic_matrix=topic_matrix,
    config=RangeCompactConfig(
        population_size=40,
        num_iterations=100,
        decay_power=2.0,
        topic_weight=0.65,
        range_weight=0.35,
        random_state=42,
    ),
)
strategy = ga_solver.optimize(
    strategy_name="range_compact_most200_topics10",
    font_name="Noto Sans",
    prefix_slices=most_slices,
)
print(f"Optimized strategy has {strategy.num_slices} slices and {strategy.total_num_ranges()} CSS ranges.")

# 6. Tier-4: Physically slice the font into .woff2 files and @font-face CSS
subsetter = FontSubsetter(font_path=font_path, font_family="Noto Sans Sliced")
manifest = subsetter.slice_font(strategy=strategy, output_dir=Path("output/notosans_sliced"))
print(f"Generated CSS at output/notosans_sliced/font_sliced.css ({len(manifest['slices'])} WOFF2 files)")

# 7. Evaluate multi-page browsing latency across GIN-2g, GIN-3g, and DSL
evaluator = FontLoadingEvaluator(font_codepoints=font_codepoints, random_state=42)
results = evaluator.evaluate_strategy(strategy, eval_docs=train_docs[:40])
for res in results:
    if res.params_name == "GIN-3g":
        print(
            f"[GIN-3g] Mean Latency: {res.total_per_glyph_ms.mean:.2f} ms/glyph | "
            f"P95: {res.total_per_glyph_ms.p95:.2f} ms | "
            f"Session Utilization: {res.session_utilization_pct:.2f}%"
        )

# 8. Export trained weights and slice assignments to Hugging Face Safetensors
save_safetensors_model(
    output_path=Path("output/notosans_sliced/model.safetensors"),
    codepoints=font_codepoints,
    frequency_array=freq_solver.frequencies_,
    topic_matrix=topic_matrix,
    strategy=strategy,
)
```

---

## Command-Line Interface (CLI) and Reproducible Pipelines

Installing `mlfont` registers five CLI entry points in your environment:

| CLI Command | Script Equivalent | Description |
| :--- | :--- | :--- |
| **`mlfont-prepare-data`** | `python3 scripts/prepare_data.py` | Downloads Noto `.ttf`/`.otf` fonts, extracts `cmap` codepoints, and streams multilingual Wikipedia corpora (`wikimedia/wikipedia`) into JSONL train/eval splits. |
| **`mlfont-train`** | `python3 scripts/train.py` | Trains `FrequencySolver`, `OnlineLDASolver`, `DNNTMSolver`, and `RangeCompactGASolver` models and saves strategy JSON + `model.safetensors`. |
| **`mlfont-evaluate`** | `python3 scripts/evaluate.py` | Runs the multi-page browsing simulator across `GIN-2g`, `GIN-3g`, `DSL`, `GIN-3g_no_session`, and `GIN-3g_short_session`. |
| **`mlfont-export`** | `python3 scripts/export_model.py` | Packages a trained strategy and physically sliced `.woff2` bundle into a Hugging Face Hub repository with `.safetensors`, `.nam`, and Model Card `README.md`. |
| **`mlfont-slice`** | `python3 scripts/inference.py` | Slices any `.ttf`/`.otf` font file using a pretrained strategy JSON or `.safetensors` checkpoint and outputs `.woff2` subsets + `@font-face` CSS. |

### Reproducing All 5 Noto Font Experiments & Figures in One Command

To execute the full end-to-end training, physical WOFF2 subsetting, multi-network evaluation, ablation studies, and figure generation across all 5 Noto fonts (`Noto Sans LGC`, `Noto Sans KR`, `Noto Sans JP`, `Noto Sans TC`, and `Noto Sans SC`):

```bash
# 1. Run full training, WOFF2 subsetting, and evaluation across all 5 Noto fonts
python3 scripts/run_noto_experiments.py \
    --data-dir data \
    --experiments-dir experiments \
    --pretrained-dir pretrained_models

# 2. Generate all publication-quality PNG (resources/) and PDF figures
python3 scripts/generate_figures.py \
    --experiments-dir experiments \
    --pretrained-dir pretrained_models \
    --png-dir resources
```

---

## Pretrained Models and Sliced Noto WOFF2 Bundles

All 5 pretrained models—complete with `model.safetensors` weights, `strategy.json` definitions, `.nam` subset files, `font_sliced.css` `@font-face` stylesheets, and physically sliced `.woff2` binaries—are included under [`pretrained_models/`](pretrained_models/) and published on the **Hugging Face Hub** in the [**MLFont Collection (`wq2012/mlfont`)**](https://huggingface.co/collections/wq2012/mlfont):

| Hugging Face Model Repo | Local Directory | Target Font | Script Coverage | Valid Codepoints | Tier-1 (`most`) | Tier-2 (`K` Topics) | Total WOFF2 Slices | Total WOFF2 Size |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| [`wq2012/mlfont-notosans-lgc`](https://huggingface.co/wq2012/mlfont-notosans-lgc) | [`mlfont-notosans-lgc`](pretrained_models/mlfont-notosans-lgc/) | `NotoSans-Regular.ttf` | Latin, Greek, Cyrillic (`en, de, fr, pl, ru, el`) | `2,297` | None | `20` | `20` | `172.0 KB` |
| [`wq2012/mlfont-notosanskr`](https://huggingface.co/wq2012/mlfont-notosanskr) | [`mlfont-notosanskr`](pretrained_models/mlfont-notosanskr/) | `NotoSansKR-Regular.otf` | Korean Hangul + Hanja (`ko`) | `12,254` | `[300, 300]` | `50` | `52` | `1.97 MB` |
| [`wq2012/mlfont-notosansjp`](https://huggingface.co/wq2012/mlfont-notosansjp) | [`mlfont-notosansjp`](pretrained_models/mlfont-notosansjp/) | `NotoSansJP-Regular.otf` | Japanese Kanji + Kana (`ja`) | `16,665` | `[500, 500]` | `60` | `62` | `2.65 MB` |
| [`wq2012/mlfont-notosanstc`](https://huggingface.co/wq2012/mlfont-notosanstc) | [`mlfont-notosanstc`](pretrained_models/mlfont-notosanstc/) | `NotoSansTC-Regular.otf` | Traditional Chinese (`zh-Hant`) | `20,637` | `[600, 600]` | `80` | `82` | `3.57 MB` |
| [`wq2012/mlfont-notosanssc`](https://huggingface.co/wq2012/mlfont-notosanssc) | [`mlfont-notosanssc`](pretrained_models/mlfont-notosanssc/) | `NotoSansSC-Regular.otf` | Simplified Chinese (`zh-Hans`) | `30,759` | `[1000, 500]` | `100` | `102` | `5.32 MB` |

Each pretrained bundle in `pretrained_models/<model_name>/` contains:
- `model.safetensors`: Complete `codepoints`, `frequency_array`, `topic_matrix`, and `slice_assignments` tensors.
- `strategy.json` & `config.json`: Full slicing strategy and hyperparameter metadata.
- `font_sliced.css`: Drop-in `@font-face` CSS stylesheet with compact `unicode-range` descriptors pointing to `woff2/slice_XXX.woff2`.
- `woff2/slice_000.woff2 .. slice_NNN.woff2`: Physically subsetted and Brotli-compressed WOFF2 font slices.
- `nam/slice_000.nam .. slice_NNN.nam`: Google Fonts `font-subsets` compatible `.nam` codepoint definition files.
- `evaluation_metrics.json` & `README.md`: Verified multi-network latency benchmarks and Hugging Face Model Card.

---

## Original Production Study and Google Fonts Deployment Results

### 1. Original Internal Benchmark Summary (Wang and Mao, 2016–2017)

In the original Google internal study evaluated on Google web index `DocJoins` across `GIN-3g` (`RTT = 100 ms`, `BW = 140 bytes/ms`, `SessionLength = 10`), MLFont demonstrated substantial latency reductions over both single-file delivery and static language/equal-bin baselines:

| Font Family | Total Codepoints (`cmap`) | Baseline (`Equal-Bin` / `Language`) Mean Latency (`GIN-3g`) | MLFont (`Most + Online LDA / DNNTM / GA`) Mean Latency (`GIN-3g`) | Relative Latency Reduction |
| :--- | :---: | :---: | :---: | :---: |
| **Noto Sans LGC** (`Latin-Greek-Cyrillic`) | `2,677` | `7.12 ms/glyph` (`Language Slice`, misses glyphs) | **`5.38 ms/glyph`** (`Online LDA 20` / `Range-Compact GA`) | **`-24.4%`** (0 missing glyphs) |
| **Noto Sans CJK SC** (`Simplified Chinese`) | `44,683` | `34.20 ms/glyph` (`Most + Equal-Bin 100`) | **`14.10 ms/glyph`** (`Most 1000+500 + Online LDA 100`) | **`-58.8%`** |
| **Noto Sans CJK TC** (`Traditional Chinese`) | `44,683` | `52.80 ms/glyph` (`Most + Equal-Bin 80`) | **`25.40 ms/glyph`** (`Most 600+600 + Online LDA 80`) | **`-51.9%`** |
| **Noto Sans CJK JP** (`Japanese`) | `44,683` | `21.90 ms/glyph` (`Most + Equal-Bin 60`) | **`8.45 ms/glyph`** (`Most 500+500 + Online LDA 60`) | **`-61.4%`** |
| **Noto Sans CJK KR** (`Korean`) | `44,683` | `22.40 ms/glyph` (`Most + Equal-Bin 50`) | **`9.85 ms/glyph`** (`Most 300+300 + Online LDA 50`) | **`-56.0%`** |

### 2. Real-World Google Fonts Production Traffic (`notosansscsliced.css`, 2017–2021)

Following the empirical validation, MLFont's 102-slice `Noto Sans SC Sliced` strategy (`most=[1000, 500]` + 100 topic slices) was launched in production on **Google Fonts Early Access** (`https://fonts.googleapis.com/earlyaccess/notosansscsliced.css`) in June 2017 (CL 157726565). Over a four-year production lifecycle tracked before graduation to the main Google Fonts dynamic serving pipeline:

- **Launch Date (June 1, 2017)**: `63` daily CSS requests.
- **One-Year Milestone (June 1, 2018)**: `261,955` daily CSS requests.
- **Peak Production Traffic (August 4, 2021)**: **`4,050,624` daily CSS requests** (`>1.66M` sustained daily requests in November 2021).

![Google Fonts Early Access Production Traffic](resources/fig4_production_traffic.png)

*Figure 3: Real-world daily CSS request volume for MLFont's `Noto Sans SC Sliced` (`notosansscsliced.css`) on Google Fonts Early Access from launch (June 2017) through November 2021, peaking at 4.05 million requests/day.*

---

## Open-Source Reproduction Benchmarks

All open-source reproduction benchmarks below were executed using `scripts/run_noto_experiments.py` on real Wikipedia corpora (`wikimedia/wikipedia`) and open-source Noto fonts. Full raw metrics are stored in [`experiments/summary_report.json`](experiments/summary_report.json).

![Cross-Font Latency and Session Utilization Benchmark](resources/fig3_benchmark_latency.png)

*Figure 4: Cross-font comparison across all five Noto fonts on `GIN-3g` (`SessionLength L = 10`): (a) Mean per-glyph download latency (`ms`, lower is better) and (b) Session glyph cache utilization (`%`, higher is better).*

### 1. Noto Sans LGC (`NotoSans-Regular.ttf` — 2,297 Codepoints, 300 Eval Pages across `en, de, fr, pl, ru, el`)

| Slicing Strategy | Slices | Missing Glyphs | CSS Ranges | `GIN-2g` Mean (P95) `ms` | `GIN-3g` Mean (P95) `ms` | `DSL` Mean (P95) `ms` | Upfront Util `%` | Session Util `%` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `single_slice` | 1 | 0 | 123 | 58.52 (542.17) | 14.78 (136.97) | 1.04 (9.65) | 3.01% | 4.63% |
| `equal_bin_20` | 20 | 0 | 140 | 22.55 (157.92) | 6.38 (44.68) | 0.84 (5.87) | 22.28% | 20.11% |
| `random_slice_20` | 20 | 0 | 2,180 | 92.96 (849.22) | 26.30 (240.26) | 3.45 (31.54) | 3.07% | 4.63% |
| `language_slice` *(misses glyphs)* | 12 | 875 | 100 | 22.99 (140.49) | 6.25 (36.78) | 0.70 (4.60) | 26.30% | 18.43% |
| **`online_lda_topics20_greedy`** | 20 | **0** | 516 | **20.15 (98.53)** | **5.70 (27.87)** | **0.75 (3.66)** | **29.87%** | **21.26%** |
| **`range_compact_topics20` (GA)** | 20 | **0** | **411 (-20.3%)** | **19.95 (101.93)** | **5.62 (28.72)** | **0.73 (3.73)** | **28.01%** | **21.17%** |
| **`dnntm_topics20_greedy`** | 20 | **0** | 584 | **18.00 (118.44)** | **5.09 (33.51)** | **0.67 (4.40)** | **23.98%** | **24.30%** |

### 2. Noto Sans KR (`NotoSansKR-Regular.otf` — 12,254 Codepoints, 400 Eval Pages in Korean `ko`)

| Slicing Strategy | Slices | Missing Glyphs | CSS Ranges | `GIN-2g` Mean (P95) `ms` | `GIN-3g` Mean (P95) `ms` | `DSL` Mean (P95) `ms` | Upfront Util `%` | Session Util `%` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `single_slice` | 1 | 0 | 90 | 271.50 (2532.95) | 68.10 (635.25) | 4.78 (44.62) | 1.33% | 2.55% |
| `equal_bin_52` | 52 | 0 | 141 | 258.59 (2061.07) | 68.28 (544.20) | 6.29 (50.14) | 1.54% | 2.88% |
| `random_slice_52` | 52 | 0 | 12,039 | 318.17 (2738.85) | 84.01 (723.16) | 7.74 (66.63) | 1.35% | 2.58% |
| `language_slice` *(misses glyphs)* | 9 | 4,625 | 1,659 | 62.12 (568.52) | 15.72 (143.87) | 1.19 (10.90) | 6.08% | 11.81% |
| `most_300_300_equal_bin_50` | 52 | 0 | 1,114 | 74.16 (325.13) | 19.62 (85.96) | 1.82 (7.94) | 5.21% | 7.52% |
| **`online_lda_topics50_greedy`** | 52 | **0** | 3,092 | **39.32 (251.97)** | **10.42 (66.67)** | **0.97 (6.17)** | **9.66%** | **16.14%** |
| **`range_compact_most300_300_topics50` (GA)** | 52 | **0** | **3,022** | **39.42 (251.97)** | **10.44 (66.67)** | **0.97 (6.17)** | **9.53%** | **16.02%** |
| **`dnntm_topics50_greedy`** | 52 | **0** | 3,108 | **31.95 (232.68)** | **8.47 (61.57)** | **0.79 (5.71)** | **12.50%** | **19.68%** |

### 3. Noto Sans JP (`NotoSansJP-Regular.otf` — 16,665 Codepoints, 400 Eval Pages in Japanese `ja`)

| Slicing Strategy | Slices | Missing Glyphs | CSS Ranges | `GIN-2g` Mean (P95) `ms` | `GIN-3g` Mean (P95) `ms` | `DSL` Mean (P95) `ms` | Upfront Util `%` | Session Util `%` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `single_slice` | 1 | 0 | 4,831 | 82.34 (642.02) | 20.63 (160.89) | 1.45 (11.29) | 2.68% | 7.05% |
| `equal_bin_62` | 62 | 0 | 4,876 | 89.24 (663.26) | 23.31 (173.24) | 2.02 (15.03) | 3.16% | 7.98% |
| `random_slice_62` | 62 | 0 | 16,489 | 106.25 (840.10) | 27.75 (219.43) | 2.41 (19.04) | 2.69% | 7.05% |
| `language_slice` *(misses glyphs)* | 12 | 2,289 | 5,966 | 46.29 (354.51) | 11.68 (89.52) | 0.87 (6.67) | 4.89% | 12.97% |
| `most_500_500_equal_bin_60` | 62 | 0 | 5,982 | 75.34 (391.47) | 19.71 (102.34) | 1.72 (8.90) | 3.74% | 8.52% |
| **`online_lda_topics60_greedy`** | 62 | **0** | 9,521 | **32.84 (191.40)** | **8.61 (50.09)** | **0.76 (4.38)** | **9.56%** | **19.92%** |
| **`range_compact_most500_500_topics60` (GA)** | 62 | **0** | **9,380** | **33.24 (193.53)** | **8.71 (50.64)** | **0.76 (4.42)** | **9.48%** | **19.61%** |
| **`dnntm_topics60_greedy`** | 62 | **0** | 9,524 | **30.69 (150.25)** | **8.05 (39.33)** | **0.71 (3.44)** | **11.12%** | **21.68%** |

### 4. Noto Sans TC (`NotoSansTC-Regular.otf` — 20,637 Codepoints, 400 Eval Pages in Traditional Chinese `zh-Hant`)

| Slicing Strategy | Slices | Missing Glyphs | CSS Ranges | `GIN-2g` Mean (P95) `ms` | `GIN-3g` Mean (P95) `ms` | `DSL` Mean (P95) `ms` | Upfront Util `%` | Session Util `%` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `single_slice` | 1 | 0 | 5,036 | 237.29 (2365.65) | 59.43 (592.49) | 4.17 (41.55) | 1.56% | 3.84% |
| `equal_bin_82` | 82 | 0 | 5,098 | 246.30 (1873.69) | 63.95 (486.50) | 5.33 (40.53) | 2.27% | 4.81% |
| `random_slice_82` | 82 | 0 | 20,436 | 312.24 (2589.18) | 81.07 (672.28) | 6.75 (56.01) | 1.56% | 3.85% |
| `language_slice` *(misses glyphs)* | 12 | 5,658 | 5,970 | 108.19 (1017.90) | 27.26 (256.58) | 2.01 (18.99) | 3.74% | 8.59% |
| `most_600_600_equal_bin_80` | 82 | 0 | 6,659 | 189.26 (740.31) | 49.24 (192.33) | 4.14 (16.10) | 2.91% | 5.50% |
| **`online_lda_topics80_greedy`** | 82 | **0** | 13,041 | **110.42 (556.98)** | **28.78 (144.97)** | **2.44 (12.19)** | **4.65%** | **9.46%** |
| **`range_compact_most600_600_topics80` (GA)** | 82 | **0** | **12,787** | **112.40 (565.79)** | **29.29 (147.25)** | **2.48 (12.38)** | **4.61%** | **9.27%** |
| **`dnntm_topics80_greedy`** | 82 | **0** | 13,116 | **92.00 (523.78)** | **23.99 (136.35)** | **2.04 (11.45)** | **6.10%** | **11.34%** |

### 5. Noto Sans SC (`NotoSansSC-Regular.otf` — 30,759 Codepoints, 400 Eval Pages in Simplified Chinese `zh-Hans`)

| Slicing Strategy | Slices | Missing Glyphs | CSS Ranges | `GIN-2g` Mean (P95) `ms` | `GIN-3g` Mean (P95) `ms` | `DSL` Mean (P95) `ms` | Upfront Util `%` | Session Util `%` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `single_slice` | 1 | 0 | 418 | 190.89 (1046.15) | 47.78 (261.86) | 3.35 (18.36) | 1.33% | 5.00% |
| `equal_bin_102` | 102 | 0 | 519 | 145.67 (880.84) | 37.53 (226.94) | 2.99 (18.08) | 1.98% | 6.90% |
| `random_slice_102` | 102 | 0 | 30,457 | 209.68 (1371.99) | 54.02 (353.49) | 4.30 (28.16) | 1.35% | 5.00% |
| `language_slice` *(misses glyphs)* | 12 | 14,918 | 7,087 | 69.03 (378.23) | 17.36 (95.16) | 1.26 (6.94) | 3.74% | 13.95% |
| `most_1000_500_equal_bin_100` | 102 | 0 | 2,978 | 114.61 (510.15) | 29.58 (131.64) | 2.38 (10.58) | 3.92% | 7.17% |
| **`online_lda_topics100_greedy`** | 102 | **0** | 11,848 | **59.35 (313.55)** | **15.36 (81.13)** | **1.25 (6.58)** | **6.18%** | **14.99%** |
| **`range_compact_most1000_500_topics100` (GA)** | 102 | **0** | **11,741** | **59.56 (314.70)** | **15.41 (81.43)** | **1.25 (6.61)** | **6.15%** | **14.93%** |
| **`dnntm_topics100_greedy`** | 102 | **0** | 11,928 | **49.65 (214.96)** | **12.86 (55.64)** | **1.05 (4.51)** | **8.22%** | **17.46%** |

---

## Ablation Studies

### 1. Effect of Topic Matrix Normalization (`mlfont.maths.normalize_topic_matrix` on Noto Sans LGC, `K = 20`)

Proper normalization of the `K x V_rem` topic-codepoint matrix before greedy or evolutionary slice assignment prevents high-volume topics from monopolizing tail codepoints:

| Normalization Method | CSS Ranges | `GIN-2g` Mean (P95) `ms` | `GIN-3g` Mean (P95) `ms` | Session Utilization `%` |
| :--- | :---: | :---: | :---: | :---: |
| `row_sum` (Column-normalized per codepoint) | 514 | 22.81 (129.69) | 6.45 (36.71) | 18.15% |
| `topic_sum` (Row-normalized L1 per topic) | 550 | 20.57 (148.35) | 5.82 (41.97) | 20.98% |
| **`global_max`** *(L2 row-norm + global max scaling)* | 516 | **20.15 (98.53)** | **5.70 (27.87)** | **21.26%** |
| **`spectral_norm`** *(Bi-stochastic Sinkhorn scaling)* | 516 | **20.15 (98.53)** | **5.70 (27.87)** | **21.26%** |

### 2. Effect of Number of Topic Slices `K` (`OnlineLDASolver` on Noto Sans LGC)

Increasing `K` improves glyph utilization by creating finer-grained character subsets, but excessive `K` (`K >= 40` on a 2,297-codepoint font) increases per-page HTTP request RTT overhead:

| Number of Topic Slices (`K`) | CSS Ranges | `GIN-2g` Mean `ms` | `GIN-3g` Mean `ms` | `DSL` Mean `ms` | Session Utilization `%` |
| :---: | :---: | :---: | :---: | :---: | :---: |
| `K = 5` | 381 | 27.83 | 7.28 | 1.36 | 10.81% |
| `K = 10` | 486 | 23.17 | 6.27 | 1.11 | 14.68% |
| **`K = 20`** *(Optimal on `GIN-3g`)* | **560** | **20.56** | **5.82** | **0.96** | **21.02%** |
| `K = 40` | 645 | 27.07 | 8.07 | 1.22 | 21.94% |
| `K = 60` | 690 | 24.87 | 7.58 | 1.11 | 30.18% |

### 3. Effect of Browsing Session Length `L` (`Noto Sans SC`, `GIN-3g`)

Even on cold-cache single-page bounces (`GIN-3g_no_session`, `L = 1`), MLFont reduces per-glyph latency by **31.5% to 45.9%** over `most_1000_500_equal_bin_100`, and by **53.5% to 63.2%** over `single_slice`:

| Strategy (`Noto Sans SC`) | `L = 1` (`GIN-3g_no_session`) Mean `ms` | `L = 3` (`GIN-3g_short_session`) Mean `ms` | `L = 10` (`GIN-3g`) Mean `ms` |
| :--- | :---: | :---: | :---: |
| `single_slice` | 200.51 ms | 155.76 ms | 47.78 ms |
| `equal_bin_102` | 210.84 ms | 122.62 ms | 37.53 ms |
| `most_1000_500_equal_bin_100` | 136.37 ms | 74.70 ms | 29.58 ms |
| **`online_lda_topics100_greedy`** | **93.20 ms** | **41.73 ms** | **15.36 ms** |
| **`range_compact_most1000_500_topics100` (GA)** | **93.46 ms** | **41.87 ms** | **15.41 ms** |
| **`dnntm_topics100_greedy`** | **73.75 ms** | **32.42 ms** | **12.86 ms** |

---

## Testing, Code Quality, and PyPI Packaging

### Run Linter, Unit Tests, and Code Coverage

```bash
bash run_tests.sh
```

`run_tests.sh` enforces `flake8 --max-line-length 100` across `mlfont`, `scripts`, and `tests` and executes the `pytest` unit test suite with `coverage` reporting.

### Build and Check PyPI Source & Wheel Distributions

```bash
bash publish.sh --build-only
```

To upload to PyPI (requires valid PyPI credentials or API token):

```bash
bash publish.sh
```

---

## Citations

If you use **MLFont**, its pretrained Noto font slices, or our benchmark suite in your research or engineering systems, please cite:

```bibtex
@article{wang2017learning,
  title={Learning Better Font Slicing Strategies from Data},
  author={Wang, Quan and Mao, Yiran},
  journal={Technical Disclosure Commons, Defensive Publications Series},
  volume={906},
  year={2017},
  url={https://www.tdcommons.org/dpubs_series/906}
}

@article{wang2026mlfont,
  title={MLFont: Machine Learning-Driven Font Subsetting and Optimal Unicode-Range Slicing for Low-Latency Web Typography},
  author={Wang, Quan and Mao, Yiran},
  journal={arXiv preprint},
  year={2026},
  note={Open-source code at \url{https://github.com/wq2012/mlfont} and pretrained models at \url{https://huggingface.co/collections/wq2012/mlfont}}
}
```
