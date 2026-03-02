#!/usr/bin/env python3
"""
pipelines/06_make_figures.py

Make thesis-ready figures for neuro-symbolic mechanistic enrichment.

Inputs:
  data/03_features/faers_report_features.csv
  data/results/mechanistic_enrichment_results.csv

Outputs (PNG):
  data/figures/box_mech_weighted_sum.png
  data/figures/box_mech_sum_paths.png
  data/figures/box_mech_sum_unique_go.png
  data/figures/box_mech_sum_unique_proteins.png
  data/figures/effect_sizes.png
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


REPORTS = Path("data/03_features/faers_report_features.csv")
RESULTS = Path("data/results/mechanistic_enrichment_results.csv")
OUT_DIR = Path("data/figures")

FEATURES = [
    "mech_weighted_sum",
    "mech_sum_paths",
    "mech_sum_unique_go",
    "mech_sum_unique_proteins",
]


def box_with_jitter(df: pd.DataFrame, feature: str, out_path: Path) -> None:
    s = df[df["y_serious"] == 1][feature].astype(float).to_numpy()
    n = df[df["y_serious"] == 0][feature].astype(float).to_numpy()

    fig = plt.figure(figsize=(7.5, 4.5))
    ax = plt.gca()

    data = [n, s]
    ax.boxplot(data, labels=["Non-serious", "Serious"], showfliers=False)

    # jitter scatter (downsample if huge)
    rng = np.random.default_rng(42)
    max_points = 800
    for i, arr in enumerate(data, start=1):
        if len(arr) > max_points:
            idx = rng.choice(len(arr), size=max_points, replace=False)
            arr_plot = arr[idx]
        else:
            arr_plot = arr

        x = rng.normal(i, 0.06, size=len(arr_plot))
        ax.scatter(x, arr_plot, s=10, alpha=0.25)

    ax.set_title(f"{feature} by FAERS seriousness")
    ax.set_ylabel(feature)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def effect_size_plot(res: pd.DataFrame, out_path: Path) -> None:
    # sort by p-value
    res = res.sort_values("p_value").copy()

    labels = res["feature"].tolist()
    cliffs = res["cliffs_delta"].astype(float).to_numpy()
    cohens = res["cohens_d"].astype(float).to_numpy()

    fig = plt.figure(figsize=(9, 4.5))
    ax = plt.gca()

    x = np.arange(len(labels))
    ax.bar(x - 0.15, cliffs, width=0.3, label="Cliff's delta")
    ax.bar(x + 0.15, cohens, width=0.3, label="Cohen's d")

    ax.axhline(0, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_title("Effect sizes (Serious vs Non-serious)")
    ax.set_ylabel("Effect size")
    ax.legend()

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    if not REPORTS.exists():
        raise FileNotFoundError(f"Missing {REPORTS}. Build report features first.")
    if not RESULTS.exists():
        raise FileNotFoundError(f"Missing {RESULTS}. Run 05_mechanistic_enrichment_test.py first.")

    df = pd.read_csv(REPORTS)
    res = pd.read_csv(RESULTS)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for feat in FEATURES:
        if feat not in df.columns:
            print(f"[skip] missing {feat}")
            continue
        out_path = OUT_DIR / f"box_{feat}.png"
        box_with_jitter(df, feat, out_path)
        print(f"✅ wrote: {out_path}")

    effect_size_plot(res, OUT_DIR / "effect_sizes.png")
    print(f"✅ wrote: {OUT_DIR / 'effect_sizes.png'}")


if __name__ == "__main__":
    main()