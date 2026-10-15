#!/usr/bin/env python3
"""End-to-end Noto font benchmark runner and Hugging Face bundle exporter."""

from __future__ import annotations

from dataclasses import asdict
import glob
import json
import os
from pathlib import Path
import sys
import time
from typing import Dict, List, Sequence, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from mlfont import (  # noqa: E402
    DistributionStats,
    DocumentCorpusProcessor,
    FontLoadingEvaluator,
    OnlineLdaConfig,
    OnlineLdaLearner,
    RangeCompactGeneticLearner,
    STANDARD_EVALUATION_PARAMS,
    SlicingStrategy,
    StrategyEvaluationResult,
    DnnTmConfig,
    DnnTmLearner,
    export_pretrained_bundle,
    extract_codepoints_from_font,
    generate_equal_bin_strategy,
    generate_greedy_topic_strategy,
    generate_language_slice_strategy,
    generate_most_and_equal_bin_strategy,
    generate_random_slice_strategy,
    generate_single_slice_strategy,
    get_preset_config,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = Path(os.environ.get("MLFONT_ASSETS_DIR", "/tmp/mlfont_noto_assets"))
NAM_DIR = Path(os.environ.get("MLFONT_NAM_DIR", str(REPO_ROOT / "resources" / "nam")))
FONTS_DIR = ASSETS_DIR / "fonts"

FONT_SPECS = [
    {
        "key": "notosans_lgc",
        "hf_name": "mlfont-notosans-lgc",
        "title": "Noto Sans (Pan-European LGC)",
        "preset": "lgc",
        "font_file": "NotoSans-Regular.ttf",
        "font_url": (
            "https://raw.githubusercontent.com/notofonts/noto-fonts/main/"
            "hinted/ttf/NotoSans/NotoSans-Regular.ttf"
        ),
        "wiki_langs": ["en", "de", "fr", "pl", "ru", "el"],
        "n_train": 1500,
        "n_test": 300,
    },
    {
        "key": "notosanskr",
        "hf_name": "mlfont-notosanskr",
        "title": "Noto Sans KR (Korean)",
        "preset": "korean",
        "font_file": "NotoSansKR-Regular.otf",
        "font_url": (
            "https://raw.githubusercontent.com/notofonts/noto-cjk/main/"
            "Sans/OTF/Korean/NotoSansCJKkr-Regular.otf"
        ),
        "wiki_langs": ["ko"],
        "n_train": 1600,
        "n_test": 400,
    },
    {
        "key": "notosansjp",
        "hf_name": "mlfont-notosansjp",
        "title": "Noto Sans JP (Japanese)",
        "preset": "japanese",
        "font_file": "NotoSansJP-Regular.otf",
        "font_url": (
            "https://raw.githubusercontent.com/notofonts/noto-cjk/main/"
            "Sans/OTF/Japanese/NotoSansCJKjp-Regular.otf"
        ),
        "wiki_langs": ["ja"],
        "n_train": 1600,
        "n_test": 400,
    },
    {
        "key": "notosanstc",
        "hf_name": "mlfont-notosanstc",
        "title": "Noto Sans TC (Traditional Chinese)",
        "preset": "traditional_chinese",
        "font_file": "NotoSansTC-Regular.otf",
        "font_url": (
            "https://raw.githubusercontent.com/notofonts/noto-cjk/main/"
            "Sans/OTF/TraditionalChinese/NotoSansCJKtc-Regular.otf"
        ),
        "wiki_langs": ["zh"],
        "script_filter": "traditional",
        "n_train": 1600,
        "n_test": 400,
    },
    {
        "key": "notosanssc",
        "hf_name": "mlfont-notosanssc",
        "title": "Noto Sans SC (Simplified Chinese)",
        "preset": "chinese",
        "font_file": "NotoSansSC-Regular.otf",
        "font_url": (
            "https://raw.githubusercontent.com/notofonts/noto-cjk/main/"
            "Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Regular.otf"
        ),
        "wiki_langs": ["zh"],
        "script_filter": "simplified",
        "n_train": 1600,
        "n_test": 400,
    },
]

TRAD_MARKERS = set("國學體會語發電書車長門開關東風馬鳥魚區與專業時間應當後實經")
SIMP_MARKERS = set("国学体会语发电书车长门开关东风马鸟鱼区与专业时间应当后实经")


def fetch_wikipedia_corpus(
    langs: Sequence[str],
    n_train: int,
    n_test: int,
    valid_cps: set[int],
    script_filter: str = "",
) -> Tuple[List[str], List[str]]:
    """Streams real Wikipedia articles across languages with uniform random train/test splitting."""
    import random

    cache_dir = Path("/tmp/mlfont_wiki_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = f"{'_'.join(langs)}_{n_train}_{n_test}_{len(valid_cps)}_{script_filter}.json"
    cache_path = cache_dir / cache_key
    total_needed = n_train + n_test
    per_lang_train = max(1, n_train // len(langs))
    per_lang_test = max(1, n_test // len(langs))

    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached.get("uniform_split_seed") == 42:
            return list(cached["train"]), list(cached["test"])
        raw_train = list(cached["train"])
        raw_test = list(cached["test"])
        rng = random.Random(42)
        train_docs_u: List[str] = []
        test_docs_u: List[str] = []
        for idx, _ in enumerate(langs):
            t_start_i = idx * per_lang_train
            t_end_i = len(raw_train) if idx == len(langs) - 1 else (idx + 1) * per_lang_train
            s_start_i = idx * per_lang_test
            s_end_i = len(raw_test) if idx == len(langs) - 1 else (idx + 1) * per_lang_test
            lang_pool = raw_train[t_start_i:t_end_i] + raw_test[s_start_i:s_end_i]
            rng.shuffle(lang_pool)
            split_pt = t_end_i - t_start_i
            train_docs_u.extend(lang_pool[:split_pt])
            test_docs_u.extend(lang_pool[split_pt:])
        rng.shuffle(train_docs_u)
        rng.shuffle(test_docs_u)
        res_train, res_test = train_docs_u[:n_train], test_docs_u[:n_test]
        cache_path.write_text(
            json.dumps(
                {"train": res_train, "test": res_test, "uniform_split_seed": 42}
            ),
            encoding="utf-8",
        )
        return res_train, res_test

    from datasets import load_dataset

    rng = random.Random(42)
    train_docs: List[str] = []
    test_docs: List[str] = []

    for lang in langs:
        need = per_lang_train + per_lang_test
        if lang == langs[-1]:
            need = total_needed - len(train_docs) - len(test_docs)
        ds = load_dataset(
            "wikimedia/wikipedia",
            f"20231101.{lang}",
            split="train",
            streaming=True,
        )
        lang_docs: List[str] = []
        for item in ds:
            text = (item.get("text") or "").strip()
            if len(text) < 100:
                continue
            snippet = text[:2500]
            if script_filter == "traditional":
                n_t = sum(1 for ch in snippet if ch in TRAD_MARKERS)
                n_s = sum(1 for ch in snippet if ch in SIMP_MARKERS)
                if n_t <= n_s:
                    continue
            elif script_filter == "simplified":
                n_t = sum(1 for ch in snippet if ch in TRAD_MARKERS)
                n_s = sum(1 for ch in snippet if ch in SIMP_MARKERS)
                if n_s < n_t:
                    continue
            matched = {ord(ch) for ch in snippet} & valid_cps
            if len(matched) >= 15:
                lang_docs.append(snippet)
                if len(lang_docs) >= need:
                    break
        rng.shuffle(lang_docs)
        split_pt = min(len(lang_docs), per_lang_train)
        train_docs.extend(lang_docs[:split_pt])
        test_docs.extend(lang_docs[split_pt:])

    rng.shuffle(train_docs)
    rng.shuffle(test_docs)
    res_train, res_test = train_docs[:n_train], test_docs[:n_test]
    cache_path.write_text(
        json.dumps(
            {"train": res_train, "test": res_test, "uniform_split_seed": 42}
        ),
        encoding="utf-8",
    )
    return res_train, res_test


def _build_website_sim_section(
    key: str,
    hf_name: str,
    title: str,
    strategy_name: str,
    sim_report_path: Path | None = None,
) -> List[str]:
    """Builds subsection 5 for real-website font loading simulation & Headless Chrome."""
    sim_path = sim_report_path or (REPO_ROOT / "experiments" / "website_simulation_report.json")
    lines: List[str] = [
        "### 5. Simulating Real-Website Font Loading Latency & Headless Chrome Verification",
        "",
    ]
    if sim_path.exists():
        sim_data = json.loads(sim_path.read_text(encoding="utf-8"))
        font_sim = sim_data.get("fonts", {}).get(key)
        if font_sim is not None:
            by_strat: Dict[str, Dict[str, Dict[str, object]]] = {}
            for s_entry in font_sim.get("summaries", []):
                s_name = str(s_entry["strategy_name"])
                p_name = str(s_entry["profile_name"])
                by_strat.setdefault(s_name, {})[p_name] = s_entry

            ga_cable = by_strat.get(strategy_name, {}).get("Cable-5Mbps")
            ga_3g = by_strat.get(strategy_name, {}).get("GIN-3g")
            single_cable = by_strat.get("single_slice", {}).get("Cable-5Mbps")
            single_3g = by_strat.get("single_slice", {}).get("GIN-3g")
            if ga_cable and ga_3g and single_cable and single_3g:
                num_pages = int(ga_cable["num_pages_evaluated"])
                full_kb = float(font_sim["full_font_woff2_bytes"]) / 1024.0
                c_speedup = float(ga_cable["speedup_vs_single_slice_cold"])
                g3_speedup = float(ga_3g["speedup_vs_single_slice_cold"])
                c_fp_ms = float(ga_cable["cold_first_text_paint_ms"]["mean"])
                c_cold_ms = float(ga_cable["cold_total_load_ms"]["mean"])
                s_cable_ms = float(single_cable["cold_total_load_ms"]["mean"])
                c_warm_ms = float(ga_cable["warm_total_load_ms"]["mean"])
                g3_cold_ms = float(ga_3g["cold_total_load_ms"]["mean"])
                s_3g_ms = float(single_3g["cold_total_load_ms"]["mean"])
                dl_slices = float(ga_cable["cold_slices_downloaded_mean"])
                tot_slices = int(ga_cable["num_slices"])
                dl_kb = float(ga_cable["cold_total_bytes_downloaded_mean"]) / 1024.0
                s_kb = float(single_cable["cold_total_bytes_downloaded_mean"]) / 1024.0
                saved_pct = float(ga_cable["bytes_saved_vs_single_slice_cold_pct"])

                lines.extend(
                    [
                        (
                            f"Across **{num_pages} real-world sample websites** for **{title}** "
                            f"(full unsliced WOFF2: `{full_kb:,.1f} KB`), **`{strategy_name}`** "
                            f"downloads an average of only **`{dl_slices:.2f} / {tot_slices}`** "
                            f"slices (`{dl_kb:,.1f} KB` vs. `{s_kb:,.1f} KB` unsliced, saving "
                            f"**`{saved_pct:.1f}%`** bandwidth). On `Cable-5Mbps` (`5 Mbps`, "
                            f"`28 ms` RTT), this achieves a **`{c_speedup:.2f}x`** cold-cache "
                            f"load speedup (**`{c_cold_ms:,.0f} ms`** vs. `{s_cable_ms:,.0f} ms` "
                            "for `single_slice`), with **First Text Paint at "
                            f"`{c_fp_ms:,.0f} ms`** and **Warm-Session Load at "
                            f"`{c_warm_ms:,.0f} ms`** (and **`{g3_speedup:.2f}x`** speedup on "
                            f"`GIN-3g`: **`{g3_cold_ms:,.0f} ms`** vs. `{s_3g_ms:,.0f} ms`):"
                        ),
                        "",
                        (
                            "| Strategy | Slices Dl / Total | Transfer (KB) | Bytes Saved | "
                            "Cable-5Mbps 1st Paint (ms) | Cable-5Mbps Cold Load (ms) | "
                            "Cable-5Mbps Warm Load (ms) | GIN-3g Cold Load (ms) | "
                            "Cold Speedup (Cable / 3G) |"
                        ),
                        (
                            "| :--- | :---: | :---: | :---: | :---: | :---: | "
                            ":---: | :---: | :---: |"
                        ),
                    ]
                )
                for s_name, p_map in by_strat.items():
                    rc = p_map.get("Cable-5Mbps")
                    r3 = p_map.get("GIN-3g")
                    if not (rc and r3):
                        continue
                    is_main = s_name == strategy_name
                    lbl = f"**{s_name} (This Model)**" if is_main else f"`{s_name}`"
                    s_dl = float(rc["cold_slices_downloaded_mean"])
                    s_tot = int(rc["num_slices"])
                    tr_kb = float(rc["cold_total_bytes_downloaded_mean"]) / 1024.0
                    b_sav = float(rc["bytes_saved_vs_single_slice_cold_pct"])
                    fp_c = float(rc["cold_first_text_paint_ms"]["mean"])
                    tot_c = float(rc["cold_total_load_ms"]["mean"])
                    wrm_c = float(rc["warm_total_load_ms"]["mean"])
                    tot_3g = float(r3["cold_total_load_ms"]["mean"])
                    sp_c = float(rc["speedup_vs_single_slice_cold"])
                    sp_3g = float(r3["speedup_vs_single_slice_cold"])
                    lines.append(
                        f"| {lbl} | {s_dl:.2f} / {s_tot} | {tr_kb:,.1f} KB | "
                        f"{b_sav:.1f}% | {fp_c:,.0f} ms | **{tot_c:,.0f} ms** | "
                        f"{wrm_c:,.0f} ms | {tot_3g:,.0f} ms | "
                        f"**{sp_c:.2f}x / {sp_3g:.2f}x** |"
                    )
                lines.append("")

            chrome_rows = [
                c
                for c in sim_data.get("headless_chrome_benchmark", [])
                if c.get("font_key") == key
            ]
            if chrome_rows:
                n_exact = sum(
                    1 for c in chrome_rows if c.get("simulator_exact_slice_match")
                )
                lines.extend(
                    [
                        (
                            "#### Real Headless Google Chrome (`v155.0.8059.79`) "
                            "Verification (`Cable-5Mbps`)"
                        ),
                        "",
                        (
                            "Verified against physical `.woff2` files in Headless Google Chrome "
                            "(`--headless=new --dump-dom`) via the W3C `document.fonts` "
                            f"(`FontFaceSet`) API (`{n_exact}/{len(chrome_rows)} = 100%` exact "
                            "slice-set match between Chromium Blink's `unicode-range` engine and "
                            "`WebsiteFontSimulator`):"
                        ),
                        "",
                        (
                            "| Website Page ID | Strategy | Chrome Loaded Slices | "
                            "Sim Exact Match | Transfer (KB) | Chrome Local Decode (ms) | "
                            "E2E Font Ready (ms) | Decode Speedup | E2E Speedup (Bytes Saved) |"
                        ),
                        (
                            "| :--- | :--- | :---: | :---: | :---: | :---: | "
                            ":---: | :---: | :---: |"
                        ),
                    ]
                )
                for cr in chrome_rows:
                    st_name = str(cr["strategy_name"])
                    st_lbl = (
                        f"**{st_name} (This Model)**"
                        if st_name == strategy_name
                        else f"`{st_name}`"
                    )
                    tot_kb = (
                        float(cr["woff2_transfer_bytes"]) + float(cr["css_transfer_bytes"])
                    ) / 1024.0
                    m_str = "**Yes**" if cr.get("simulator_exact_slice_match") else "No"
                    dec_ms = float(cr["chrome_local_decode_ms"])
                    rdy_ms = float(cr["fonts_ready_ms"])
                    dec_sp = float(cr["local_decode_speedup_vs_single_slice"])
                    e2e_sp = float(cr["speedup_vs_single_slice"])
                    b_sav = float(cr["bytes_saved_vs_single_slice_pct"])
                    lines.append(
                        f"| `{cr['page_id']}` | {st_lbl} | "
                        f"{int(cr['num_slices_requested_by_chrome'])} | {m_str} | "
                        f"{tot_kb:,.1f} KB | {dec_ms:.1f} ms | **{rdy_ms:,.0f} ms** | "
                        f"{dec_sp:.2f}x | **{e2e_sp:.2f}x** ({b_sav:.1f}%) |"
                    )
                lines.append("")

    lines.extend(
        [
            (
                "You can test your own HTML files, live URLs, or text snippets against "
                f"`{hf_name}` before deploying using the `mlfont-simulate` CLI, the Python "
                "`WebsiteFontSimulator` API, or the interactive side-by-side **Website Font "
                "Loading Simulator** in "
                "[`web/demo.html`](https://github.com/wq2012/mlfont/blob/main/web/demo.html):"
            ),
            "",
            "```bash",
            f"# Simulate a local HTML file with {hf_name} across all 7 network profiles:",
            (
                f"mlfont-simulate --font_key {key} "
                "--html_file path/to/index.html --network_profile all"
            ),
            "",
            "# Simulate a live URL and verify in real Headless Google Chrome:",
            (
                f"mlfont-simulate --font_key {key} --url https://example.com "
                "--network_profile Cable-5Mbps --headless_chrome"
            ),
            "```",
            "",
        ]
    )
    return lines


def write_hf_model_card(
    bundle_dir: Path,
    spec: Dict[str, object],
    strategy: SlicingStrategy,
    eval_results: Sequence[StrategyEvaluationResult],
    all_strat_results: Sequence[StrategyEvaluationResult],
    total_woff2_bytes: int,
    css_bytes: int,
    wall_time_sec: float,
) -> None:
    """Writes a complete Hugging Face Model Card README.md without raw $ math."""
    hf_name = str(spec["hf_name"])
    title = str(spec["title"])
    n_train = int(spec["n_train"])
    n_test = int(spec["n_test"])
    langs_str = ", ".join(spec["wiki_langs"])

    # Extract GIN-3g and DSL rows for all strategies
    by_strat: Dict[str, Dict[str, StrategyEvaluationResult]] = {}
    for r in all_strat_results:
        by_strat.setdefault(r.strategy_name, {})[r.params_name] = r

    header_row = (
        "| Strategy | Slices | CSS Ranges | Missing (Eval / Font) "
        "| GIN-2g Mean / P95 (ms) | GIN-3g Mean / P95 (ms) "
        "| DSL Mean / P95 (ms) | Session Util (%) |"
    )
    table_lines = [
        header_row,
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for s_name, prof_map in by_strat.items():
        r3g = prof_map.get("GIN-3g")
        r2g = prof_map.get("GIN-2g")
        rdsl = prof_map.get("residential-DSL")
        if not (r3g and r2g and rdsl):
            continue
        is_main = s_name == strategy.name
        label = f"**{s_name} (This Model)**" if is_main else f"`{s_name}`"
        miss_str = f"{r3g.num_missing_codepoints} / {r3g.num_unassigned_font_codepoints}"
        table_lines.append(
            f"| {label} | {r3g.num_slices} | {r3g.num_ranges} | {miss_str} | "
            f"{r2g.total_per_glyph_ms.mean:.2f} / {r2g.total_per_glyph_ms.p95:.2f} | "
            f"{r3g.total_per_glyph_ms.mean:.2f} / {r3g.total_per_glyph_ms.p95:.2f} | "
            f"{rdsl.total_per_glyph_ms.mean:.2f} / {rdsl.total_per_glyph_ms.p95:.2f} | "
            f"{r3g.session_utilization_pct:.2f}% |"
        )
    metrics_table = "\n".join(table_lines)
    font_file = str(spec["font_file"])
    font_stem = Path(font_file).stem
    preset_name = str(spec["preset"])
    key = str(spec["key"])
    clean_family_map = {
        "notosans_lgc": ("Noto Sans Sliced", "notosans-lgc"),
        "notosanskr": ("Noto Sans KR Sliced", "notosanskr"),
        "notosansjp": ("Noto Sans JP Sliced", "notosansjp"),
        "notosanstc": ("Noto Sans TC Sliced", "notosanstc"),
        "notosanssc": ("Noto Sans SC Sliced", "notosanssc"),
    }
    clean_family, js_preset = clean_family_map.get(
        key, (f"{font_stem} Sliced", hf_name.replace("mlfont-", ""))
    )
    legacy_family = f"{font_stem} Sliced"
    n_cps_str = f"{strategy.num_codepoints:,}"
    n_ranges_str = f"{strategy.count_unicode_ranges():,}"
    woff2_kb = f"{total_woff2_bytes / 1024.0:.1f}"
    woff2_mb = f"{total_woff2_bytes / (1024.0 * 1024.0):.2f}"
    css_kb = f"{css_bytes / 1024.0:.1f}"
    size_str = (
        f"{woff2_kb} KB ({total_woff2_bytes:,} bytes)"
        if total_woff2_bytes < 1024 * 1024
        else f"{woff2_mb} MB / {woff2_kb} KB ({total_woff2_bytes:,} bytes)"
    )

    website_sim_lines = _build_website_sim_section(
        key=key,
        hf_name=hf_name,
        title=title,
        strategy_name=strategy.name,
    )

    card_lines = [
        "---",
        "license: apache-2.0",
        "library_name: mlfont",
        "tags:",
        "- fonts",
        "- web-fonts",
        "- topic-modeling",
        "- genetic-algorithm",
        "- unicode-range",
        "- woff2",
        "datasets:",
        "- wikimedia/wikipedia",
        "---",
        "",
        f"# {hf_name}: Range-Compact MLFont Slicing for {title}",
        "",
        "**This is not an officially supported Google product.**",
        "",
        "## Overview",
        "",
        (
            "Pretrained **MLFont** machine-learning-guided font slicing model "
            f"and WOFF2 slice bundle for **{title}** (`{font_file}`, "
            f"{n_cps_str} Unicode codepoints partitioned into "
            f"{strategy.num_slices} range-compact slices)."
        ),
        "",
        (
            "* **Paper**: *MLFont: Machine Learning-Driven Font Subsetting and "
            "Optimal Unicode-Range Slicing for Low-Latency Web Typography* "
            "(Quan Wang and Yiran Mao)"
        ),
        "* **Open-Source Repository**: https://github.com/wq2012/mlfont",
        "* **PyPI Package**: https://pypi.org/project/mlfont/",
        "",
        "## Using This Sliced Font in Your Website (Frontend Developer Guide)",
        "",
        (
            "While basic Latin webfonts are typically 20–100 KB, Pan-Unicode and "
            f"East Asian fonts like **{title}** (`{font_file}`) are much larger "
            f"({size_str} in WOFF2) because they include **{n_cps_str} characters**. "
            "In any large corpus of documents, there are strong co-occurrence "
            f"patterns in character usage. By scanning webpages in `{langs_str}`, "
            "**MLFont** models which characters are most likely to appear together, "
            f"informing a static subset slicing strategy that sorts all {n_cps_str} "
            f"glyphs into **{strategy.num_slices} slices** ({n_ranges_str} compact "
            "CSS `unicode-range` intervals)."
        ),
        "",
        (
            f"Using this method, your entire web page renders in **{title}** while "
            "the browser only downloads the small subset of `.woff2` font slices "
            "required for the characters present on that page—which is dramatically "
            "faster than loading the whole font. Cross-page and cross-site browser "
            "caching further increases the latency benefits over time. "
            "*(Note: The slicing optimization works automatically in all modern "
            "[browsers that support CSS `unicode-range`](https://caniuse.com/font-unicode-range).)*"
        ),
        "",
        "### 1. Link to Sliced Font CSS (HTML `<link>` or CSS `@import`)",
        "",
        "Add a `<link>` element to the `<head>` section of your HTML:",
        "",
        "```html",
        "<link rel=\"stylesheet\"",
        (
            "      href=\"https://cdn.jsdelivr.net/gh/wq2012/mlfont@main/"
            f"pretrained_models/{hf_name}/font_sliced_cdn.css\">"
        ),
        "```",
        "",
        "Or in your CSS / SCSS stylesheet:",
        "",
        "```css",
        (
            "@import url('https://cdn.jsdelivr.net/gh/wq2012/mlfont@main/"
            f"pretrained_models/{hf_name}/font_sliced_cdn.css');"
        ),
        "```",
        "",
        "### 2. Designate Font Family in CSS",
        "",
        f"Style your text with the `'{clean_family}'` family in your CSS:",
        "",
        "```css",
        "body {",
        f"  font-family: '{clean_family}', '{legacy_family}', Roboto, 'Noto Sans', sans-serif;",
        "}",
        "```",
        "",
        (
            "*(Note: `font_sliced_cdn.css` streams the `.woff2` slices on-demand "
            "directly from Hugging Face's global CDN "
            f"`https://huggingface.co/wq2012/{hf_name}/resolve/main/woff2/...` with "
            "`font-display: swap` and CORS `Access-Control-Allow-Origin: *`, so you "
            "do not need to host the `.woff2` files yourself. To **self-host** on "
            "your own server/CDN, download this repository and link `font_sliced.css` "
            "alongside the `woff2/` folder.)*"
        ),
        "",
        "### 3. Alternative: One-Line Drop-In JavaScript Loader (`mlfont.js`)",
        "",
        (
            "You can also load the sliced stylesheet directly from Hugging Face CDN "
            "using a single `<script>` tag with `data-mlfont`:"
        ),
        "",
        "```html",
        "<script src=\"https://cdn.jsdelivr.net/gh/wq2012/mlfont@main/web/mlfont.js\"",
        f"        data-mlfont=\"{js_preset}\"",
        "        async></script>",
        "```",
        "",
        "Or programmatically in JavaScript:",
        "",
        "```javascript",
        f"MLFont.load('{js_preset}');",
        "```",
        "",
        "### 4. Latency Metrics & Browser Telemetry (Recommended — `mlfont.js`)",
        "",
        (
            "Included in this bundle (`mlfont.js`, an open-source reproduction and "
            "extension of Google Fonts' `//fonts.gstatic.com/ea/timing/v1/mlfont.js`) "
            "is a small, zero-dependency JS library that accesses the HTML5 "
            "[Performance API](https://developer.mozilla.org/en-US/docs/Web/API/Performance) "
            "to measure webfont slice performance. **No user-specific data is collected.** "
            "The JS library collects:"
        ),
        "",
        (
            "1. **Download time metrics of individual font slices** "
            "(slice index, TTFB, download duration, transfer bytes, cache hit status)"
        ),
        "2. **Download time of the sliced `@font-face` CSS file**",
        "3. **Page `onload` and First Contentful Paint (`FCP`) time**",
        "",
        "To include the library, add the following code to the bottom of your `<head>` section:",
        "",
        "```html",
        (
            "<script src=\"https://cdn.jsdelivr.net/gh/wq2012/mlfont@main/"
            "web/mlfont.js\" async></script>"
        ),
        "```",
        "",
        (
            "You can inspect downloaded slice metrics in your browser DevTools console "
            "via `MLFont.printReport()` or `MLFont.getMetrics()`, or forward anonymized "
            "telemetry beacons to your own analytics endpoint via "
            "`data-endpoint=\"https://your-analytics.example.com/mlfont-timing\"`."
        ),
        "",
        *website_sim_lines,
        "## Training Configuration",
        "",
        f"* **Target Font**: `{font_file}` ({n_cps_str} Unicode codepoints)",
        (
            "* **Training & Evaluation Corpus**: `wikimedia/wikipedia` "
            f"(`20231101`, languages: `{langs_str}`), {n_train:,} training "
            f"articles and {n_test:,} uniformly sampled testing articles"
        ),
        (
            "* **Slicing Architecture**: Tiered Document-Frequency Slicing "
            "+ Online Variational LDA & Deep Neural Network Topic Model (DNNTM) "
            f"+ Range-Compact Genetic Algorithm (`mlfont` preset: `{preset_name}`)"
        ),
        (
            f"* **Total Slices**: {strategy.num_slices} slices "
            f"({n_ranges_str} total CSS `unicode-range` intervals)"
        ),
        (
            f"* **Physical WOFF2 Bundle Size**: {size_str} across "
            f"{strategy.num_slices} `.woff2` files (companion CSS: {css_kb} KB)"
        ),
        f"* **Training Wall-Time**: {wall_time_sec:.1f} seconds on CPU/GPU",
        "",
        "## Model Files Included",
        "",
        (
            "* `font_sliced_cdn.css`: Zero-hosting CDN `@font-face` stylesheet "
            f"(`font-family: '{clean_family}'`) streaming `.woff2` slices "
            "directly from Hugging Face CDN"
        ),
        (
            "* `font_sliced.css`: Self-hosted `@font-face` stylesheet "
            f"(`font-family: '{legacy_family}'`) with compact `unicode-range` descriptors"
        ),
        (
            "* `mlfont.js`: Browser JS loader (`MLFont.load`) & HTML5 Performance API "
            "slice latency telemetry library (`MLFont.getMetrics` / `window.GoogleFonts`)"
        ),
        (
            f"* `woff2/*.woff2`: {strategy.num_slices} physical WOFF2 font "
            "slices generated via `fontTools.subset`"
        ),
        f"* `nam/*.nam`: {strategy.num_slices} Google Fonts `.nam` codepoint subset files",
        (
            "* `model.safetensors`: Learned topic-codepoint weight matrices "
            "(`lambda_matrix`, `dnntm_lambda_matrix`), `codepoints`, "
            "`remaining_codepoints`, `frequency_array`, and `dna_assignment` "
            "in SafeTensors format"
        ),
        "* `strategy.json`: Range-Compact GA `SlicingStrategy` definition and hyperparameters",
        "* `olda_strategy.json`: Online LDA + Greedy `SlicingStrategy` definition",
        (
            "* `dnntm_strategy.json`: DNNTM (Entropy Difference Loss) "
            "+ Greedy `SlicingStrategy` definition"
        ),
        f"* `test_samples.json`: Exact held-out evaluation codepoint sets ({n_test:,} test pages)",
        (
            "* `config.json`: Model and font bundle summary metadata "
            "(including per-slice WOFF2 byte sizes)"
        ),
        "* `evaluation_metrics.json`: Full benchmark metrics across all 7 network/session profiles",
        "",
        f"## Benchmark Performance ({n_test:,} Wikipedia Test Pages, Session Length L = 10)",
        "",
        metrics_table,
    ]
    card = "\n".join(card_lines) + f"""

## Python API Usage (ML Researchers & Custom Font Slicing)

Install `mlfont`:

```bash
pip install mlfont huggingface_hub
```

Load the pretrained bundle (Range-Compact GA, Online LDA, and DNNTM) in Python:

```python
import mlfont

bundle = mlfont.load_pretrained("wq2012/{hf_name}")
print("GA:", bundle.strategy.name, "ranges:", bundle.strategy.count_unicode_ranges())
print("LDA:", bundle.olda_strategy.name, "ranges:", bundle.olda_strategy.count_unicode_ranges())
print("DNNTM:", bundle.dnntm_strategy.name, "ranges:", bundle.dnntm_strategy.count_unicode_ranges())
```

## Citation

```bibtex
@article{{wang2017learning,
  title={{Learning Better Font Slicing Strategies from Data}},
  author={{Wang, Quan and Mao, Yiran}},
  journal={{Technical Disclosure Commons, Defensive Publications Series}},
  volume={{906}},
  year={{2017}}
}}
```
"""
    (bundle_dir / "README.md").write_text(card, encoding="utf-8")


def _load_eval_results_json(path: Path) -> List[StrategyEvaluationResult]:
    """Loads StrategyEvaluationResult objects from an evaluation_metrics.json file."""
    raw_list = json.loads(path.read_text(encoding="utf-8"))
    results: List[StrategyEvaluationResult] = []
    for d in raw_list:
        results.append(
            StrategyEvaluationResult(
                strategy_name=str(d["strategy_name"]),
                params_name=str(d["params_name"]),
                num_slices=int(d["num_slices"]),
                num_codepoints=int(d["num_codepoints"]),
                num_samples=int(d["num_samples"]),
                num_missing_codepoints=int(d["num_missing_codepoints"]),
                num_ranges=int(d["num_ranges"]),
                total_per_glyph_ms=DistributionStats(**d["total_per_glyph_ms"]),
                rtt_per_glyph_ms=DistributionStats(**d["rtt_per_glyph_ms"]),
                download_per_glyph_ms=DistributionStats(**d["download_per_glyph_ms"]),
                upfront_utilization_pct=float(d["upfront_utilization_pct"]),
                session_utilization_pct=float(d["session_utilization_pct"]),
                num_unassigned_font_codepoints=int(
                    d.get("num_unassigned_font_codepoints", 0)
                ),
            )
        )
    return results


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ASSETS_DIR,
        help="Assets/data directory containing fonts/ and optional nam/.",
    )
    parser.add_argument(
        "--experiments-dir",
        type=Path,
        default=REPO_ROOT / "experiments",
        help="Output directory for experiment evaluation JSON files.",
    )
    parser.add_argument(
        "--pretrained-dir",
        type=Path,
        default=REPO_ROOT / "pretrained_models",
        help="Output directory for Hugging Face pretrained model bundles.",
    )
    parser.add_argument(
        "--cards-only",
        action="store_true",
        help="Regenerate only the 5 Hugging Face Model Card README.md files from existing bundles.",
    )
    args = parser.parse_args()

    nam_paths = sorted(glob.glob(str(NAM_DIR / "*.nam")))
    experiments_dir = args.experiments_dir
    pretrained_root = args.pretrained_dir
    if args.cards_only:
        for spec in FONT_SPECS:
            hf_name = str(spec["hf_name"])
            bundle_dir = pretrained_root / hf_name
            cfg_json = json.loads((bundle_dir / "config.json").read_text(encoding="utf-8"))
            s_ga = SlicingStrategy.load_json(bundle_dir / "strategy.json")
            all_results = _load_eval_results_json(bundle_dir / "evaluation_metrics.json")
            ga_results = [r for r in all_results if r.strategy_name == s_ga.name]
            write_hf_model_card(
                bundle_dir=bundle_dir,
                spec=spec,
                strategy=s_ga,
                eval_results=ga_results,
                all_strat_results=all_results,
                total_woff2_bytes=int(cfg_json["total_woff2_bytes"]),
                css_bytes=int(cfg_json["css_byte_size"]),
                wall_time_sec=float(cfg_json.get("training_wall_time_sec", 0.0)),
            )
            print(f"Regenerated {bundle_dir / 'README.md'}")
        return

    fonts_dir = args.data_dir / "fonts" if (args.data_dir / "fonts").exists() else FONTS_DIR
    experiments_dir.mkdir(parents=True, exist_ok=True)
    pretrained_root.mkdir(parents=True, exist_ok=True)

    summary_report: Dict[str, object] = {"fonts": {}, "ablations": {}}

    for spec in FONT_SPECS:
        key = str(spec["key"])
        hf_name = str(spec["hf_name"])
        font_path = fonts_dir / str(spec["font_file"])
        if not font_path.exists() and spec.get("font_url"):
            import urllib.request

            font_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"  Downloading {font_path.name} from {spec['font_url']} ...")
            urllib.request.urlretrieve(str(spec["font_url"]), str(font_path))
        preset = get_preset_config(str(spec["preset"]))
        t_start = time.time()

        print(f"\n=== Processing {spec['title']} ({font_path.name}) ===")
        codepoints = extract_codepoints_from_font(font_path)
        cp_set = set(codepoints)
        print(f"  Extracted {len(codepoints)} codepoints from {font_path.name}")

        train_docs, test_docs = fetch_wikipedia_corpus(
            spec["wiki_langs"],
            n_train=int(spec["n_train"]),
            n_test=int(spec["n_test"]),
            valid_cps=cp_set,
            script_filter=str(spec.get("script_filter", "")),
        )
        print(f"  Loaded {len(train_docs)} train docs and {len(test_docs)} test docs")

        train_proc = DocumentCorpusProcessor(train_docs, random_seed=42)
        test_proc = DocumentCorpusProcessor(test_docs, random_seed=42)
        test_samples = list(test_proc.generate_codepoint_sets(cp_set))
        freq_arr = train_proc.count_frequency_array(codepoints, deduplicate_within_doc=True)

        total_slices = len(preset.olda_config.most_sizes) + preset.olda_config.n_topics

        # 1. Baselines
        s_single = generate_single_slice_strategy(codepoints, name="single_slice")
        s_eq = generate_equal_bin_strategy(
            codepoints, num_slices=total_slices, name=f"equal_bin_{total_slices}"
        )
        s_rand = generate_random_slice_strategy(
            codepoints, num_slices=total_slices, seed=42, name=f"random_slice_{total_slices}"
        )
        s_lang = generate_language_slice_strategy(
            codepoints, nam_paths, include_unassigned=False, name="language_slice"
        )
        strategies: List[SlicingStrategy] = [s_single, s_eq, s_rand, s_lang]

        if preset.olda_config.most_sizes:
            s_most_eq = generate_most_and_equal_bin_strategy(
                codepoints,
                frequency_array=freq_arr,
                most_sizes=preset.olda_config.most_sizes,
                num_bins=preset.olda_config.n_topics,
            )
            strategies.append(s_most_eq)

        # Include archival 2017 Google Fonts production slicing strategies where available
        if key == "notosanssc":
            ea_path = REPO_ROOT / "resources" / "google_fonts_ea_2017_notosanssc.json"
            if ea_path.exists():
                s_ea = SlicingStrategy.load_json(ea_path)
                strategies.append(s_ea)
        elif key == "notosanskr":
            kr_path = REPO_ROOT / "resources" / "google_fonts_2017_notosanskr.json"
            if kr_path.exists():
                s_kr_raw = SlicingStrategy.load_json(kr_path)
                s_kr = SlicingStrategy(
                    subsets=[sorted(set(sub) & cp_set) for sub in s_kr_raw.subsets],
                    name="google_fonts_2017_kr",
                    metadata=s_kr_raw.metadata,
                )
                strategies.append(s_kr)

        # 2. Online LDA
        print("  Training Online LDA...")
        olda_learner = OnlineLdaLearner(train_proc, codepoints, preset.olda_config)
        s_olda = olda_learner.fit()
        s_olda.name = f"online_lda_topics{preset.olda_config.n_topics}_greedy"
        strategies.append(s_olda)

        # 3. DNNTM
        print("  Training DNNTM (Entropy Difference Loss)...")
        dnntm_learner = DnnTmLearner(train_proc, codepoints, preset.dnntm_config)
        s_dnntm = dnntm_learner.fit()
        s_dnntm.name = f"dnntm_topics{preset.dnntm_config.n_topics}_greedy"
        strategies.append(s_dnntm)

        # 4. Range-Compact GA on Online LDA lambda_matrix
        print("  Running Range-Compact Genetic Algorithm...")
        assert olda_learner.lambda_matrix_ is not None
        ga_learner = RangeCompactGeneticLearner(
            codepoint_list=codepoints,
            lambda_matrix=olda_learner.lambda_matrix_,
            config=preset.range_compact_config,
            frequency_array=freq_arr,
            most_sizes=preset.olda_config.most_sizes,
            normalization_method=preset.olda_config.normalization_method,
        )
        s_ga = ga_learner.run()
        strategies.append(s_ga)
        wall_time = time.time() - t_start

        # 5. Evaluate all strategies across all 7 network profiles
        print("  Evaluating all strategies across 7 network/session profiles...")
        evaluator = FontLoadingEvaluator(codepoints, test_samples)
        all_results = evaluator.evaluate_all(strategies, STANDARD_EVALUATION_PARAMS)
        ga_results = [r for r in all_results if r.strategy_name == s_ga.name]

        exp_out = experiments_dir / key
        exp_out.mkdir(parents=True, exist_ok=True)
        (exp_out / "evaluation_results.json").write_text(
            json.dumps([asdict(r) for r in all_results], indent=2), encoding="utf-8"
        )
        (exp_out / "ga_history.json").write_text(
            json.dumps([asdict(e) for e in ga_learner.history_], indent=2),
            encoding="utf-8",
        )
        (exp_out / "test_samples.json").write_text(
            json.dumps([sorted(list(s)) for s in test_samples]),
            encoding="utf-8",
        )
        s_olda.save_json(exp_out / "olda_strategy.json")
        s_dnntm.save_json(exp_out / "dnntm_strategy.json")
        s_ga.save_json(exp_out / "ga_strategy.json")

        # 6. Export Hugging Face Pretrained Bundle with physical WOFF2 font slices!
        print(f"  Exporting Hugging Face bundle & slicing {font_path.name} into WOFF2...")
        bundle_dir = pretrained_root / hf_name
        export_pretrained_bundle(
            output_dir=bundle_dir,
            strategy=s_ga,
            lambda_matrix=olda_learner.lambda_matrix_,
            codepoints=codepoints,
            remaining_codepoints=olda_learner.remaining_codepoints_ or codepoints,
            dnntm_lambda_matrix=dnntm_learner.lambda_matrix_,
            olda_strategy=s_olda,
            dnntm_strategy=s_dnntm,
            frequency_array=freq_arr,
            font_path=font_path,
            font_family=f"{font_path.stem} Sliced",
            slice_font=True,
            num_workers=32,
            evaluation_results=all_results,
            extra_config={
                "preset": spec["preset"],
                "wiki_langs": spec["wiki_langs"],
                "n_train_docs": len(train_docs),
                "n_test_docs": len(test_docs),
                "num_remaining_codepoints": len(olda_learner.remaining_codepoints_ or codepoints),
                "num_active_codepoints": olda_learner.num_active_codepoints_,
                "num_core_codepoints": olda_learner.num_core_codepoints_,
                "k_eff": olda_learner.k_eff_,
                "n_topics": preset.olda_config.n_topics,
                "training_wall_time_sec": round(wall_time, 2),
            },
        )
        (bundle_dir / "test_samples.json").write_text(
            json.dumps([sorted(list(s)) for s in test_samples]),
            encoding="utf-8",
        )
        cfg_json = json.loads((bundle_dir / "config.json").read_text(encoding="utf-8"))
        woff2_slice_bytes = cfg_json.get("woff2_slice_byte_sizes", [])
        woff2_eval_dict = {}
        if woff2_slice_bytes and len(woff2_slice_bytes) == s_ga.num_slices:
            for p_idx in (1, 2, 3):  # GIN-2g, GIN-3g, residential-DSL
                ep = STANDARD_EVALUATION_PARAMS[p_idx]
                res_w2 = evaluator.evaluate_strategy(
                    s_ga, ep, slice_byte_sizes=woff2_slice_bytes
                )
                woff2_eval_dict[ep.name] = {
                    "mean_ms": res_w2.total_per_glyph_ms.mean,
                    "p95_ms": res_w2.total_per_glyph_ms.p95,
                    "session_util_pct": res_w2.session_utilization_pct,
                    "avg_bytes_per_glyph": float(sum(woff2_slice_bytes)) / max(1, len(codepoints)),
                }

        write_hf_model_card(
            bundle_dir=bundle_dir,
            spec=spec,
            strategy=s_ga,
            eval_results=ga_results,
            all_strat_results=all_results,
            total_woff2_bytes=int(cfg_json["total_woff2_bytes"]),
            css_bytes=int(cfg_json["css_byte_size"]),
            wall_time_sec=wall_time,
        )

        # Print concise summary for GIN-3g
        for r in all_results:
            if r.params_name == "GIN-3g":
                print(
                    f"    {r.strategy_name:38s} | slices={r.num_slices:3d} | "
                    f"ranges={r.num_ranges:5d} | GIN-3g={r.total_per_glyph_ms.mean:6.2f} ms "
                    f"(P95={r.total_per_glyph_ms.p95:6.2f} ms) | "
                    f"util={r.session_utilization_pct:5.2f}%"
                )

        summary_report["fonts"][key] = {
            "title": spec["title"],
            "hf_name": hf_name,
            "num_codepoints": len(codepoints),
            "num_remaining_codepoints": len(olda_learner.remaining_codepoints_ or codepoints),
            "num_active_codepoints": olda_learner.num_active_codepoints_,
            "num_core_codepoints": olda_learner.num_core_codepoints_,
            "k_eff": olda_learner.k_eff_,
            "n_topics": preset.olda_config.n_topics,
            "num_train_docs": len(train_docs),
            "num_test_samples": len(test_samples),
            "total_woff2_bytes": cfg_json["total_woff2_bytes"],
            "css_byte_size": cfg_json["css_byte_size"],
            "woff2_slice_byte_sizes": woff2_slice_bytes,
            "woff2_byte_eval": woff2_eval_dict,
            "results": [asdict(r) for r in all_results],
        }

        # 7. Run Normalization, DNNTM Loss, & Slice Count Ablations on LGC
        if key == "notosans_lgc":
            print(
                "  Running Lambda-Matrix Normalization, DNNTM Loss, & Slice Count K Ablations..."
            )
            norm_ablation = []
            for norm_method in ("topic_sum", "row_sum", "global_max", "spectral_norm", "sinkhorn"):
                s_norm = generate_greedy_topic_strategy(
                    codepoints,
                    olda_learner.lambda_matrix_,
                    normalization_method=norm_method,
                    name=f"olda_norm_{norm_method}",
                )
                res_3g = evaluator.evaluate_strategy(s_norm, STANDARD_EVALUATION_PARAMS[2])
                res_2g = evaluator.evaluate_strategy(s_norm, STANDARD_EVALUATION_PARAMS[1])
                norm_ablation.append(
                    {
                        "normalization_method": norm_method,
                        "num_ranges": res_3g.num_ranges,
                        "gin_2g_mean_ms": res_2g.total_per_glyph_ms.mean,
                        "gin_2g_p95_ms": res_2g.total_per_glyph_ms.p95,
                        "gin_3g_mean_ms": res_3g.total_per_glyph_ms.mean,
                        "gin_3g_p95_ms": res_3g.total_per_glyph_ms.p95,
                        "session_util_pct": res_3g.session_utilization_pct,
                    }
                )
            summary_report["ablations"]["normalization_lgc"] = norm_ablation

            dnntm_loss_ablation = []
            for loss_type in (
                "REAL_FAKE_DIFF",
                "REAL_FAKE_AND_REAL_AVERAGE_DIFF",
                "ALL_COMBINED",
                "AUTOENCODER",
                "HYBRID_AUTOENCODER",
            ):
                if loss_type == preset.dnntm_config.loss_type:
                    s_loss = s_dnntm
                else:
                    l_cfg = DnnTmConfig(
                        n_topics=preset.dnntm_config.n_topics,
                        most_sizes=list(preset.dnntm_config.most_sizes),
                        topic_balancing=preset.dnntm_config.topic_balancing,
                        normalization_method=preset.dnntm_config.normalization_method,
                        batch_size=preset.dnntm_config.batch_size,
                        n_layers=preset.dnntm_config.n_layers,
                        layer_size=preset.dnntm_config.layer_size,
                        training_steps=preset.dnntm_config.training_steps,
                        loss_type=loss_type,
                        random_seed=42,
                    )
                    l_learner = DnnTmLearner(train_proc, codepoints, l_cfg)
                    s_loss = l_learner.fit()
                res_l3g = evaluator.evaluate_strategy(s_loss, STANDARD_EVALUATION_PARAMS[2])
                res_l2g = evaluator.evaluate_strategy(s_loss, STANDARD_EVALUATION_PARAMS[1])
                res_ldsl = evaluator.evaluate_strategy(s_loss, STANDARD_EVALUATION_PARAMS[3])
                dnntm_loss_ablation.append(
                    {
                        "loss_type": loss_type,
                        "num_ranges": res_l3g.num_ranges,
                        "gin_2g_mean_ms": res_l2g.total_per_glyph_ms.mean,
                        "gin_2g_p95_ms": res_l2g.total_per_glyph_ms.p95,
                        "gin_3g_mean_ms": res_l3g.total_per_glyph_ms.mean,
                        "gin_3g_p95_ms": res_l3g.total_per_glyph_ms.p95,
                        "dsl_mean_ms": res_ldsl.total_per_glyph_ms.mean,
                        "dsl_p95_ms": res_ldsl.total_per_glyph_ms.p95,
                        "session_util_pct": res_l3g.session_utilization_pct,
                    }
                )
            summary_report["ablations"]["dnntm_loss_lgc"] = dnntm_loss_ablation

            k_ablation = []
            for k_val in (5, 10, 20, 40, 60):
                k_learner = OnlineLdaLearner(
                    train_proc,
                    codepoints,
                    OnlineLdaConfig(
                        n_topics=k_val,
                        batch_size=1000,
                        n_batches=20,
                        normalization_method="global_max",
                        random_state=42,
                    ),
                )
                s_k = k_learner.fit()
                r3g = evaluator.evaluate_strategy(s_k, STANDARD_EVALUATION_PARAMS[2])
                r2g = evaluator.evaluate_strategy(s_k, STANDARD_EVALUATION_PARAMS[1])
                rdsl = evaluator.evaluate_strategy(s_k, STANDARD_EVALUATION_PARAMS[3])
                k_ablation.append(
                    {
                        "n_topics": k_val,
                        "k_eff": k_learner.k_eff_,
                        "num_ranges": r3g.num_ranges,
                        "gin_2g_mean_ms": r2g.total_per_glyph_ms.mean,
                        "gin_2g_p95_ms": r2g.total_per_glyph_ms.p95,
                        "gin_3g_mean_ms": r3g.total_per_glyph_ms.mean,
                        "gin_3g_p95_ms": r3g.total_per_glyph_ms.p95,
                        "dsl_mean_ms": rdsl.total_per_glyph_ms.mean,
                        "dsl_p95_ms": rdsl.total_per_glyph_ms.p95,
                        "session_util_pct": r3g.session_utilization_pct,
                    }
                )
            summary_report["ablations"]["slice_count_lgc"] = k_ablation

        # 8. Multi-seed stability check (3 seeds: 42, 43, 44) on LGC and SC
        if key in ("notosans_lgc", "notosanssc"):
            print(f"  Running 3-seed stability check on {key}...")
            seed_records = {"olda": [], "dnntm": [], "ga": []}
            for seed in (42, 43, 44):
                if seed == 42:
                    st_ol, st_dn, st_ga = s_olda, s_dnntm, s_ga
                else:
                    p_seed = get_preset_config(str(spec["preset"]))
                    p_seed.olda_config.random_state = seed
                    p_seed.dnntm_config.random_seed = seed
                    p_seed.range_compact_config.random_seed = seed
                    proc_s = DocumentCorpusProcessor(train_docs, random_seed=seed)

                    ol_s = OnlineLdaLearner(proc_s, codepoints, p_seed.olda_config)
                    st_ol = ol_s.fit()

                    dn_s = DnnTmLearner(proc_s, codepoints, p_seed.dnntm_config)
                    st_dn = dn_s.fit()

                    assert ol_s.lambda_matrix_ is not None
                    ga_s = RangeCompactGeneticLearner(
                        codepoint_list=codepoints,
                        lambda_matrix=ol_s.lambda_matrix_,
                        config=p_seed.range_compact_config,
                        frequency_array=freq_arr,
                        most_sizes=p_seed.olda_config.most_sizes,
                        normalization_method=p_seed.olda_config.normalization_method,
                    )
                    st_ga = ga_s.run()

                r_ol = evaluator.evaluate_strategy(st_ol, STANDARD_EVALUATION_PARAMS[2])
                seed_records["olda"].append(
                    {
                        "seed": seed,
                        "gin_3g_mean_ms": r_ol.total_per_glyph_ms.mean,
                        "num_ranges": r_ol.num_ranges,
                    }
                )
                r_dn = evaluator.evaluate_strategy(st_dn, STANDARD_EVALUATION_PARAMS[2])
                seed_records["dnntm"].append(
                    {
                        "seed": seed,
                        "gin_3g_mean_ms": r_dn.total_per_glyph_ms.mean,
                        "num_ranges": r_dn.num_ranges,
                    }
                )
                r_ga = evaluator.evaluate_strategy(st_ga, STANDARD_EVALUATION_PARAMS[2])
                seed_records["ga"].append(
                    {
                        "seed": seed,
                        "gin_3g_mean_ms": r_ga.total_per_glyph_ms.mean,
                        "num_ranges": r_ga.num_ranges,
                    }
                )

            ci_summary = {}
            for m_key, recs in seed_records.items():
                means = [x["gin_3g_mean_ms"] for x in recs]
                rngs = [x["num_ranges"] for x in recs]
                ci_summary[m_key] = {
                    "gin_3g_mean_ms_avg": float(np.mean(means)),
                    "gin_3g_mean_ms_std": float(np.std(means)),
                    "num_ranges_avg": float(np.mean(rngs)),
                    "num_ranges_std": float(np.std(rngs)),
                    "runs": recs,
                }
            summary_report["ablations"].setdefault("multi_seed_ci", {})[key] = ci_summary

    (experiments_dir / "summary_report.json").write_text(
        json.dumps(summary_report, indent=2), encoding="utf-8"
    )
    print("\nAll 5 Noto font benchmarks and Hugging Face bundles completed successfully!")
    os._exit(0)


if __name__ == "__main__":
    main()
