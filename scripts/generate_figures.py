#!/usr/bin/env python3
"""Generates publication-quality figures for the MLFont paper and README."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import struct
from typing import Dict, List

import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

try:
    from safetensors.numpy import load_file
except ImportError:
    _DTYPE_MAP = {"F32": np.float32, "F64": np.float64, "I32": np.int32, "I64": np.int64}

    def load_file(path: str) -> Dict[str, np.ndarray]:
        raw = Path(path).read_bytes()
        header_len = struct.unpack("<Q", raw[:8])[0]
        header = json.loads(raw[8 : 8 + header_len].decode("utf-8"))
        data_start = 8 + header_len
        out: Dict[str, np.ndarray] = {}
        for k, meta in header.items():
            if k == "__metadata__":
                continue
            start, end = meta["data_offsets"]
            arr = np.frombuffer(
                raw[data_start + start : data_start + end],
                dtype=_DTYPE_MAP[meta["dtype"]],
            )
            out[k] = arr.reshape(meta["shape"]).copy()
        return out

matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parent.parent

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "legend.fontsize": 8.5,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 300,
    }
)

COLORS = {
    "single": "#7f8c8d",
    "equal_bin": "#95a5a6",
    "random": "#bdc3c7",
    "language": "#e67e22",
    "most_eq": "#f39c12",
    "olda": "#2980b9",
    "dnntm": "#8e44ad",
    "ga": "#27ae60",
}


def _save_both(fig: plt.Figure, stem: str, png_dir: Path, pdf_dir: Path | None) -> None:
    png_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_dir / f"{stem}.png", bbox_inches="tight", dpi=300)
    if pdf_dir is not None:
        pdf_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(pdf_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_fig1_pareto_and_zipf(
    summary: Dict[str, object],
    pretrained_dir: Path,
    png_dir: Path,
    pdf_dir: Path | None,
) -> None:
    """Fig 1: Zipfian codepoint frequency curve & Latency vs CSS Range Pareto frontier."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))

    font_labels = [
        ("notosanssc", "mlfont-notosanssc", "Noto Sans SC", "#c0392b"),
        ("notosanstc", "mlfont-notosanstc", "Noto Sans TC", "#8e44ad"),
        ("notosansjp", "mlfont-notosansjp", "Noto Sans JP", "#2980b9"),
        ("notosanskr", "mlfont-notosanskr", "Noto Sans KR", "#16a085"),
        ("notosans_lgc", "mlfont-notosans-lgc", "Noto Sans LGC", "#d35400"),
    ]
    for fkey, hf_name, base_label, color in font_labels:
        st_path = pretrained_dir / hf_name / "model.safetensors"
        if not st_path.exists():
            continue
        tensors = load_file(str(st_path))
        freqs = np.sort(tensors["frequency_array"])[::-1]
        pos = freqs[freqs > 0]
        ranks = np.arange(1, len(pos) + 1)
        n_cps = summary["fonts"].get(fkey, {}).get(
            "num_codepoints", len(tensors["frequency_array"])
        )
        label = f"{base_label} ({n_cps:,} cps, {len(pos):,} active)"
        ax1.plot(ranks, pos / max(1.0, float(pos[0])), label=label, color=color, lw=1.8)

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.axvline(500, color="#34495e", ls="--", lw=1.1, alpha=0.75)
    ax1.text(
        560,
        0.35,
        "High-Freq Tier\n(most_1..k)",
        fontsize=8.5,
        color="#2c3e50",
    )
    ax1.set_title("(a) Zipfian Codepoint Document-Frequency Decay")
    ax1.set_xlabel("Active Codepoint Rank (log scale)")
    ax1.set_ylabel("Normalized Document Frequency (log scale)")
    ax1.grid(True, which="both", ls=":", alpha=0.45)
    ax1.legend(loc="lower left", frameon=True, fontsize=7.8)

    # Panel (b): Pareto trade-off on Noto Sans SC (GIN-3g latency vs CSS unicode-range count)
    sc_results = summary["fonts"]["notosanssc"]["results"]
    gin3g_sc = [r for r in sc_results if r["params_name"] == "GIN-3g"]

    markers = {
        "single": ("s", COLORS["single"], "Single Slice"),
        "equal_bin": ("^", COLORS["equal_bin"], "Equal-Bin"),
        "random": ("v", COLORS["random"], "Random Slice"),
        "language": ("D", COLORS["language"], "Language Slice (.nam)"),
        "most_": ("P", COLORS["most_eq"], "Most + Equal-Bin"),
        "google_fonts_ea": ("p", "#16a085", "Google Fonts EA (2017)"),
        "online_lda": ("o", COLORS["olda"], "Online LDA (Greedy)"),
        "dnntm": ("X", COLORS["dnntm"], "DNNTM (Greedy)"),
        "range_compact": ("*", COLORS["ga"], "MLFont Range-Compact GA"),
    }

    for r in gin3g_sc:
        sname = r["strategy_name"]
        m_char, m_col, m_lbl = "o", "#333333", sname
        for prefix, (ch, col, lbl) in markers.items():
            if sname.startswith(prefix):
                m_char, m_col, m_lbl = ch, col, lbl
                break
        size = 190 if m_char == "*" else 95
        ax2.scatter(
            r["num_ranges"],
            r["total_per_glyph_ms"]["mean"],
            marker=m_char,
            s=size,
            color=m_col,
            edgecolors="black",
            linewidths=0.7,
            label=m_lbl,
            zorder=5 if m_char == "*" else 3,
        )

    ax2.set_xscale("log")
    ax2.set_title("(b) Latency vs. CSS Unicode-Range Pareto Frontier (Noto Sans SC)")
    ax2.set_xlabel("Total CSS unicode-range Intervals (log scale, lower is more compact)")
    ax2.set_ylabel("Mean Per-Glyph Latency on GIN-3g (ms, lower is faster)")
    ax2.grid(True, which="both", ls=":", alpha=0.45)
    ax2.legend(loc="upper right", frameon=True, fontsize=8.0)

    fig.tight_layout()
    _save_both(fig, "fig1_architecture", png_dir, pdf_dir)


def plot_fig2_ga_convergence(
    experiments_dir: Path,
    png_dir: Path,
    pdf_dir: Path | None,
) -> None:
    """Fig 2: Range-Compact Genetic Algorithm convergence and curriculum decay."""
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.9))

    font_keys = [
        ("notosans_lgc", "Noto Sans LGC", "#d35400"),
        ("notosanskr", "Noto Sans KR", "#16a085"),
        ("notosansjp", "Noto Sans JP", "#2980b9"),
        ("notosanstc", "Noto Sans TC", "#8e44ad"),
        ("notosanssc", "Noto Sans SC", "#c0392b"),
    ]

    for key, label, color in font_keys:
        hist_file = experiments_dir / key / "ga_history.json"
        if not hist_file.exists():
            continue
        hist = json.loads(hist_file.read_text(encoding="utf-8"))
        iters = [h["iteration"] for h in hist]
        r_cost = np.array([h["avg_range_cost"] for h in hist], dtype=np.float64)
        t_cost = np.array([h["best_topic_cost"] for h in hist], dtype=np.float64)
        axes[0].plot(iters, r_cost / max(1e-9, r_cost[0]), label=label, color=color, lw=1.8)
        axes[1].plot(iters, t_cost, label=label, color=color, lw=1.8)

    axes[0].set_title("(a) Population Range Cost Trajectory")
    axes[0].set_xlabel("GA Generation (Iteration)")
    axes[0].set_ylabel("Relative Range Cost C_range(t) / C_range(0)")
    axes[0].grid(True, ls=":", alpha=0.5)
    axes[0].legend(loc="upper right", frameon=True)

    axes[1].set_title("(b) Topic Consistency Cost C_topic(t)")
    axes[1].set_xlabel("GA Generation (Iteration)")
    axes[1].set_ylabel("Topic Discrepancy Cost C_topic(t) (Lower is Better)")
    axes[1].grid(True, ls=":", alpha=0.5)
    axes[1].legend(loc="upper right", ncol=2, frameon=True, fontsize=8.0)

    sc_hist_file = experiments_dir / "notosanssc" / "ga_history.json"
    if sc_hist_file.exists():
        sc_hist = json.loads(sc_hist_file.read_text(encoding="utf-8"))
        iters = [h["iteration"] for h in sc_hist]
        max_diffs = [h["max_diff"] for h in sc_hist]
        axes[2].plot(
            iters,
            max_diffs,
            color="#2c3e50",
            lw=2.2,
            label="Curriculum max_diff(t) (p=3.0)",
        )
        axes[2].fill_between(iters, max_diffs, color="#3498db", alpha=0.18)
    axes[2].set_title("(c) Power-Law Slice-Size Constraint Decay")
    axes[2].set_xlabel("GA Generation (Iteration)")
    axes[2].set_ylabel("Max Allowed Slice Deviation (Codepoints)")
    axes[2].grid(True, ls=":", alpha=0.5)
    axes[2].legend(loc="upper right", frameon=True)

    fig.tight_layout()
    _save_both(fig, "fig2_ga_convergence", png_dir, pdf_dir)


def plot_fig3_benchmark_latency(
    summary: Dict[str, object],
    png_dir: Path,
    pdf_dir: Path | None,
) -> None:
    """Fig 3: Cross-font GIN-3g per-glyph latency and session glyph utilization."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.4))

    font_order = [
        ("notosans_lgc", "Noto Sans\nLGC"),
        ("notosanskr", "Noto Sans\nKR"),
        ("notosansjp", "Noto Sans\nJP"),
        ("notosanstc", "Noto Sans\nTC"),
        ("notosanssc", "Noto Sans\nSC"),
    ]
    strat_specs = [
        ("single_slice", "Single Slice", COLORS["single"]),
        ("equal_bin", "Equal-Bin", COLORS["equal_bin"]),
        ("language_slice", "Language (.nam)", COLORS["language"]),
        ("most_", "Most + Equal-Bin", COLORS["most_eq"]),
        ("online_lda", "Online LDA", COLORS["olda"]),
        ("dnntm", "DNNTM", COLORS["dnntm"]),
        ("range_compact", "MLFont (GA)", COLORS["ga"]),
    ]

    x = np.arange(len(font_order))
    width = 0.11

    for idx, (prefix, label, color) in enumerate(strat_specs):
        means: List[float] = []
        utils: List[float] = []
        for fkey, _ in font_order:
            res_list = summary["fonts"][fkey]["results"]
            match = None
            for r in res_list:
                if r["params_name"] == "GIN-3g" and r["strategy_name"].startswith(prefix):
                    match = r
                    break
            if match is None and prefix == "most_":
                # LGC has no most_sizes tier by default; fall back to equal_bin
                for r in res_list:
                    if r["params_name"] == "GIN-3g" and r["strategy_name"].startswith("equal_bin"):
                        match = r
                        break
            means.append(match["total_per_glyph_ms"]["mean"] if match else 0.0)
            utils.append(match["session_utilization_pct"] if match else 0.0)

        offset = (idx - len(strat_specs) / 2.0 + 0.5) * width
        ax1.bar(x + offset, means, width, label=label, color=color, edgecolor="black", lw=0.4)
        ax2.bar(x + offset, utils, width, label=label, color=color, edgecolor="black", lw=0.4)

    ax1.set_xticks(x)
    ax1.set_xticklabels([lbl for _, lbl in font_order])
    ax1.set_ylabel("Mean Per-Glyph Loading Latency on GIN-3g (ms)")
    ax1.set_title("(a) Mean Per-Glyph Latency Across 5 Noto Fonts (Lower is Better)")
    ax1.set_ylim(0, 78)
    ax1.grid(True, axis="y", ls=":", alpha=0.5)
    ax1.legend(loc="upper right", ncol=2, frameon=True, fontsize=8)

    ax2.set_xticks(x)
    ax2.set_xticklabels([lbl for _, lbl in font_order])
    ax2.set_ylabel("Session Glyph Utilization (%, L = 10 Pages)")
    ax2.set_title("(b) Session Downloaded Glyph Utilization (Higher is Better)")
    ax2.set_ylim(0, 38.0)
    ax2.grid(True, axis="y", ls=":", alpha=0.5)
    ax2.legend(loc="upper right", ncol=2, frameon=True, fontsize=8)

    fig.tight_layout()
    _save_both(fig, "fig3_benchmark_latency", png_dir, pdf_dir)


def plot_fig4_production_traffic(
    csv_path: Path,
    png_dir: Path,
    pdf_dir: Path | None,
) -> None:
    """Fig 4: 2017-2021 Google Fonts Early Access Noto Sans SC Sliced production traffic."""
    daily_max: Dict[str, int] = {}
    for line in csv_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split(",")
        if len(parts) != 3:
            continue
        date_str, _, count_str = parts
        try:
            cnt = int(count_str)
        except ValueError:
            continue
        if date_str not in daily_max or cnt > daily_max[date_str]:
            daily_max[date_str] = cnt

    sorted_dates = sorted(daily_max.keys())
    dates = [datetime.strptime(d, "%Y%m%d") for d in sorted_dates]
    counts = np.array([daily_max[d] for d in sorted_dates], dtype=np.float64) / 1e6

    window = 30
    rolling = np.convolve(counts, np.ones(window) / window, mode="valid")
    rolling_dates = dates[window - 1 :]

    fig, ax = plt.subplots(figsize=(8.8, 4.0))
    ax.plot(
        dates,
        counts,
        color="#3498db",
        alpha=0.35,
        lw=0.9,
        label="Daily Primary URL CSS Requests (Millions)",
    )
    ax.plot(
        rolling_dates,
        rolling,
        color="#c0392b",
        lw=2.2,
        label="30-Day Moving Average",
    )
    ax.set_title(
        "Google Fonts Early Access: Noto Sans SC Sliced Production Traffic (2017-2021)"
    )
    ax.set_xlabel("Date (June 2017 - November 2021)")
    ax.set_ylabel("Daily CSS Stylesheet Requests (Millions)")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.grid(True, ls=":", alpha=0.5)
    ax.annotate(
        "Launch: June 1, 2017\n(63 requests/day)",
        xy=( mdates.date2num(dates[0]), counts[0]),
        xytext=(mdates.date2num(dates[120]), 0.35),
        arrowprops={"arrowstyle": "->", "color": "#2c3e50", "lw": 1.1},
        fontsize=8.5,
    )
    peak_idx = int(np.argmax(counts))
    ax.annotate(
        f"Peak: {counts[peak_idx]:.2f}M requests/day\n({sorted_dates[peak_idx]})",
        xy=(mdates.date2num(dates[peak_idx]), counts[peak_idx]),
        xytext=(mdates.date2num(dates[max(0, peak_idx - 420)]), counts[peak_idx] * 0.88),
        arrowprops={"arrowstyle": "->", "color": "#c0392b", "lw": 1.1},
        fontsize=8.5,
    )
    ax.legend(loc="upper left", frameon=True)
    fig.tight_layout()
    _save_both(fig, "fig4_production_traffic", png_dir, pdf_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiments-dir",
        type=Path,
        default=REPO_ROOT / "experiments",
        help="Path to experiments directory containing summary_report.json.",
    )
    parser.add_argument(
        "--pretrained-dir",
        type=Path,
        default=REPO_ROOT / "pretrained_models",
        help="Path to pretrained_models directory.",
    )
    parser.add_argument(
        "--png-dir",
        type=Path,
        default=REPO_ROOT / "resources",
        help="Output directory for PNG figures.",
    )
    parser.add_argument(
        "--pdf-dir",
        type=Path,
        default=None,
        help="Optional output directory for vector PDF figures.",
    )
    args = parser.parse_args()

    summary_path = args.experiments_dir / "summary_report.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    plot_fig1_pareto_and_zipf(summary, args.pretrained_dir, args.png_dir, args.pdf_dir)
    plot_fig2_ga_convergence(args.experiments_dir, args.png_dir, args.pdf_dir)
    plot_fig3_benchmark_latency(summary, args.png_dir, args.pdf_dir)

    traffic_csv = args.experiments_dir / "production_traffic_2017_2021.csv"
    if traffic_csv.exists():
        plot_fig4_production_traffic(traffic_csv, args.png_dir, args.pdf_dir)
    print(f"Successfully generated all 4 figures in {args.png_dir} and {args.pdf_dir}")


if __name__ == "__main__":
    main()
