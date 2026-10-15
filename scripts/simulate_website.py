#!/usr/bin/env python3
"""CLI tool for simulating web font loading on real websites and Headless Chrome."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import List, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mlfont.simulator import (  # noqa: E402
    BrowserNetworkProfile,
    ChromeBrowserPageResult,
    STANDARD_BROWSER_NETWORK_PROFILES,
    WebsiteComparisonReport,
    WebsiteFontSimulator,
    get_network_profile,
    get_real_website_benchmark_pages,
    load_webpage_sample,
    run_headless_chrome_benchmark,
    save_website_simulation_report_json,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

FONT_BUNDLE_DIRS = {
    "notosanssc": "mlfont-notosanssc",
    "notosanstc": "mlfont-notosanstc",
    "notosansjp": "mlfont-notosansjp",
    "notosanskr": "mlfont-notosanskr",
    "notosans_lgc": "mlfont-notosans-lgc",
}


def _format_strategy_table(report: WebsiteComparisonReport, profile_name: str) -> str:
    rows = [s for s in report.summaries if s.profile_name == profile_name]
    if not rows:
        return ""
    lines = [
        f"\n### {report.font_family} ({report.font_key}) — Network Profile: {profile_name}",
        (
            f"Full Unsliced WOFF2: {report.full_font_woff2_bytes / 1024:.1f} KB "
            f"({report.num_codepoints:,} codepoints) | "
            f"Pages Evaluated: {rows[0].num_pages_evaluated}"
        ),
        (
            "| Strategy | Slices | CSS Ranges | CSS Gzip | Cold Slices | "
            "Cold Transfer | Cold 1st Paint | Cold Total Load | Warm Load | "
            "Cold Speedup | Bytes Saved |"
        ),
        (
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | "
            ":---: | :---: | :---: | :---: |"
        ),
    ]
    for s in rows:
        lines.append(
            f"| `{s.strategy_name}` | {s.num_slices} | {s.num_unicode_ranges:,} | "
            f"{s.css_gzip_bytes / 1024:.1f} KB | {s.cold_slices_downloaded_mean:.2f} | "
            f"{s.cold_total_bytes_downloaded_mean / 1024:.1f} KB | "
            f"{s.cold_first_text_paint_ms.mean:.0f} ms | "
            f"**{s.cold_total_load_ms.mean:.0f} ms** | "
            f"{s.warm_total_load_ms.mean:.0f} ms | "
            f"**{s.speedup_vs_single_slice_cold:.2f}x** | "
            f"**{s.bytes_saved_vs_single_slice_cold_pct:.1f}%** |"
        )
    return "\n".join(lines)


def _format_per_page_table(
    report: WebsiteComparisonReport,
    profile_name: str,
    ga_strategy_name: str,
) -> str:
    cold_single = {
        r.page_id: r
        for r in report.page_results
        if r.profile_name == profile_name
        and r.strategy_name == "single_slice"
        and r.cache_mode == "cold"
    }
    cold_ga = [
        r
        for r in report.page_results
        if r.profile_name == profile_name
        and r.strategy_name == ga_strategy_name
        and r.cache_mode == "cold"
    ]
    if not cold_ga:
        return ""
    lines = [
        f"\n#### Per-Website Breakdown ({report.font_key}, {profile_name}, Cold Cache)",
        (
            "| Website / Page | Category | Unique CPs | HF Token Cov | "
            "Full Font Load | MLFont Slices | MLFont Transfer | MLFont 1st Paint | "
            "MLFont Total Load | Speedup |"
        ),
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for r in cold_ga:
        s_res = cold_single.get(r.page_id)
        full_ms = s_res.total_font_load_ms if s_res else 0.0
        speedup = full_ms / max(1e-6, r.total_font_load_ms) if full_ms > 0 else 1.0
        lines.append(
            f"| `{r.page_id}` | {r.page_category} | {r.covered_codepoints} | "
            f"{r.high_freq_token_coverage_pct:.1f}% | {full_ms:.0f} ms | "
            f"{r.num_slices_downloaded} (`{r.downloaded_slice_indices}`) | "
            f"{r.total_bytes_downloaded / 1024:.1f} KB | "
            f"{r.first_text_paint_ms:.0f} ms | **{r.total_font_load_ms:.0f} ms** | "
            f"**{speedup:.2f}x** |"
        )
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pretrained_dir",
        type=Path,
        default=REPO_ROOT / "pretrained_models",
        help="Root directory containing mlfont-* pretrained model bundles.",
    )
    parser.add_argument(
        "--model_dir",
        type=Path,
        default=None,
        help="Optional specific pretrained bundle directory (overrides --font_key).",
    )
    parser.add_argument(
        "--font_key",
        type=str,
        default="notosanssc",
        help=(
            "Font key to simulate: notosanssc, notosanstc, notosansjp, "
            "notosanskr, notosans_lgc, or 'all'."
        ),
    )
    parser.add_argument(
        "--network_profile",
        "--profiles",
        dest="network_profile",
        type=str,
        default="all",
        help=(
            "Network profile name (GIN-2g, GIN-3g, Fast-3G, residential-DSL, "
            "Cable-5Mbps, Mobile-4G-LTE, Broadband-Fiber, or 'all')."
        ),
    )
    parser.add_argument(
        "--url",
        type=str,
        default=None,
        help="Optional live HTTP/HTTPS URL to fetch and simulate.",
    )
    parser.add_argument(
        "--html_file",
        type=Path,
        default=None,
        help="Optional local HTML or text file to simulate.",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Optional inline text or HTML snippet to simulate.",
    )
    parser.add_argument(
        "--headless_chrome",
        "--run_chrome",
        dest="headless_chrome",
        action="store_true",
        help="Also run real Headless Google Chrome benchmark against local throttled server.",
    )
    parser.add_argument(
        "--output_json",
        type=Path,
        default=None,
        help="Optional path to save the full JSON simulation report.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    if args.network_profile.lower() == "all":
        profiles: List[BrowserNetworkProfile] = list(STANDARD_BROWSER_NETWORK_PROFILES)
    else:
        profiles = [
            get_network_profile(p.strip())
            for p in args.network_profile.split(",")
            if p.strip()
        ]
        if not profiles:
            raise ValueError("At least one valid network profile must be specified.")

    if args.model_dir is not None:
        bundle_dirs = [args.model_dir]
    elif args.font_key.lower() == "all":
        bundle_dirs = [
            args.pretrained_dir / FONT_BUNDLE_DIRS[k]
            for k in [
                "notosanssc",
                "notosanstc",
                "notosansjp",
                "notosanskr",
                "notosans_lgc",
            ]
        ]
    else:
        norm = args.font_key.lower().replace("-", "_").replace("mlfont_", "")
        if norm == "notosans":
            norm = "notosans_lgc"
        if norm not in FONT_BUNDLE_DIRS:
            valid = list(FONT_BUNDLE_DIRS.keys())
            raise ValueError(f"Unknown font_key {args.font_key!r}. Choose from {valid} or 'all'.")
        bundle_dirs = [args.pretrained_dir / FONT_BUNDLE_DIRS[norm]]

    all_reports: List[WebsiteComparisonReport] = []
    chrome_results: List[ChromeBrowserPageResult] = []

    for bdir in bundle_dirs:
        sim = WebsiteFontSimulator.from_pretrained(bdir, include_baselines=True)
        if args.url is not None:
            pages = [
                load_webpage_sample(
                    args.url, page_id="custom_url", title=args.url, font_key=sim.font_key
                )
            ]
        elif args.html_file is not None:
            pages = [
                load_webpage_sample(
                    args.html_file,
                    page_id=args.html_file.stem,
                    title=args.html_file.name,
                    font_key=sim.font_key,
                )
            ]
        elif args.text is not None:
            pages = [
                load_webpage_sample(
                    args.text, page_id="inline_text", title="Inline Text", font_key=sim.font_key
                )
            ]
        else:
            pages = get_real_website_benchmark_pages(sim.font_key)

        report = sim.compare_strategies(pages=pages, profiles=profiles)
        all_reports.append(report)

        ga_name = next(
            (name for name in sim.strategies if name.startswith("range_compact")),
            list(sim.strategies.keys())[-1],
        )
        display_profiles = (
            ["GIN-3g", "Cable-5Mbps", "Mobile-4G-LTE"]
            if args.network_profile.lower() == "all"
            else [p.name for p in profiles]
        )
        for pname in display_profiles:
            print(_format_strategy_table(report, pname))
        print(_format_per_page_table(report, display_profiles[0], ga_name))

        if args.headless_chrome:
            chrome_prof = (
                profiles[0]
                if len(profiles) == 1
                else get_network_profile("Cable-5Mbps")
            )
            print(
                f"\n[Headless Chrome] Running real browser benchmark for {sim.font_key} "
                f"on {chrome_prof.name}..."
            )
            c_res = run_headless_chrome_benchmark(
                bundle_dir=bdir,
                pages=pages[:3],
                profile=chrome_prof,
            )
            chrome_results.extend(c_res)
            print(
                "| Page ID | Strategy | Chrome Slices | Sim Match | Transfer (KB) | "
                "Chrome Decode (ms) | E2E Ready (ms) | Decode Speedup | E2E Speedup | Bytes Saved |"
            )
            print(
                "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
            )
            for cr in c_res:
                tot_kb = (cr.woff2_transfer_bytes + cr.css_transfer_bytes) / 1024.0
                match_str = "Yes" if cr.simulator_exact_slice_match else "No"
                print(
                    f"| `{cr.page_id}` | `{cr.strategy_name}` | "
                    f"{cr.num_slices_requested_by_chrome} | {match_str} | {tot_kb:.1f} KB | "
                    f"{cr.chrome_local_decode_ms:.1f} ms | **{cr.fonts_ready_ms:.0f} ms** | "
                    f"{cr.local_decode_speedup_vs_single_slice:.2f}x | "
                    f"**{cr.speedup_vs_single_slice:.2f}x** | "
                    f"**{cr.bytes_saved_vs_single_slice_pct:.1f}%** |"
                )

    if args.output_json is not None:
        save_website_simulation_report_json(all_reports, chrome_results, args.output_json)
        print(f"\nSaved complete website simulation report to {args.output_json}")


if __name__ == "__main__":
    main()
