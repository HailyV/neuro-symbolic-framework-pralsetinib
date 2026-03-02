#!/usr/bin/env python3
"""
pipelines/05_mechanistic_enrichment_test.py

Neuro-symbolic hypothesis test:
Are mechanistic scores enriched in serious FAERS reports (Pralsetinib)?

Input:
  data/03_features/faers_report_features.csv

Output:
  data/results/mechanistic_enrichment_results.csv

Methods:
- Mann–Whitney U test (two-sided)
- Effect sizes:
  - Cliff's delta (robust, nonparametric)
  - Cohen's d (mean diff standardized; assume continuous)
- Bootstrap 95% CI for mean difference (serious - non-serious)
- Multiple testing correction: Benjamini–Hochberg (FDR)
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu


IN_PATH = Path("data/03_features/faers_report_features.csv")
OUT_PATH = Path("data/results/mechanistic_enrichment_results.csv")

FEATURES = [
    "mech_weighted_sum",
    "mech_sum_paths",
    "mech_sum_unique_go",
    "mech_sum_unique_proteins",
]

N_BOOT = 5000
RNG_SEED = 42


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    # (mean(a)-mean(b)) / pooled_sd
    a = a.astype(float)
    b = b.astype(float)
    va = np.var(a, ddof=1) if len(a) > 1 else 0.0
    vb = np.var(b, ddof=1) if len(b) > 1 else 0.0
    pooled = np.sqrt((va + vb) / 2.0) if (va + vb) > 0 else np.nan
    if pooled == 0 or np.isnan(pooled):
        return np.nan
    return (np.mean(a) - np.mean(b)) / pooled


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cliff's delta: P(a > b) - P(a < b).
    Range [-1, 1]. Positive means a tends larger than b.
    Efficient O(n log n) approach via sorting.
    """
    a = np.sort(a.astype(float))
    b = np.sort(b.astype(float))
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return np.nan

    # count pairs where a > b and a < b using two pointers
    gt = 0
    lt = 0
    j = 0
    k = 0

    # a > b
    for i in range(n):
        while j < m and b[j] < a[i]:
            j += 1
        gt += j  # b[0..j-1] < a[i]

    # a < b
    for i in range(n):
        while k < m and b[k] <= a[i]:
            k += 1
        lt += (m - k)  # b[k..] > a[i]

    return (gt - lt) / (n * m)


def bootstrap_mean_diff(a: np.ndarray, b: np.ndarray, n_boot: int, seed: int) -> tuple[float, float, float]:
    """
    Bootstrap mean difference: mean(a) - mean(b).
    Returns (diff, ci_low, ci_high) for 95% CI.
    """
    rng = np.random.default_rng(seed)
    a = a.astype(float)
    b = b.astype(float)
    if len(a) == 0 or len(b) == 0:
        return (np.nan, np.nan, np.nan)

    obs = float(np.mean(a) - np.mean(b))
    diffs = np.empty(n_boot, dtype=float)

    for i in range(n_boot):
        sa = rng.choice(a, size=len(a), replace=True)
        sb = rng.choice(b, size=len(b), replace=True)
        diffs[i] = np.mean(sa) - np.mean(sb)

    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return (obs, float(lo), float(hi))


def benjamini_hochberg(pvals: list[float]) -> list[float]:
    """
    Return BH-adjusted q-values in original order.
    """
    p = np.array(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    out = np.empty(n, dtype=float)
    out[order] = q
    return out.tolist()


def main() -> None:
    if not IN_PATH.exists():
        raise FileNotFoundError(f"Missing {IN_PATH}")

    df = pd.read_csv(IN_PATH)
    if "y_serious" not in df.columns:
        raise ValueError("Missing y_serious column in report features.")

    serious = df[df["y_serious"] == 1]
    nonser = df[df["y_serious"] == 0]

    rows = []
    pvals = []

    for feat in FEATURES:
        if feat not in df.columns:
            print(f"[skip] missing feature: {feat}")
            continue

        a = serious[feat].to_numpy()
        b = nonser[feat].to_numpy()

        # MWU test
        stat, p = mannwhitneyu(a, b, alternative="two-sided")
        pvals.append(float(p))

        # Effects
        mean_a = float(np.mean(a))
        mean_b = float(np.mean(b))
        med_a = float(np.median(a))
        med_b = float(np.median(b))

        d = cohens_d(a, b)
        cd = cliffs_delta(a, b)
        diff, ci_lo, ci_hi = bootstrap_mean_diff(a, b, n_boot=N_BOOT, seed=RNG_SEED)

        rows.append(
            {
                "feature": feat,
                "n_serious": int(len(a)),
                "n_nonserious": int(len(b)),
                "mean_serious": mean_a,
                "mean_nonserious": mean_b,
                "median_serious": med_a,
                "median_nonserious": med_b,
                "mean_diff_serious_minus_nonserious": diff,
                "mean_diff_ci95_low": ci_lo,
                "mean_diff_ci95_high": ci_hi,
                "mannwhitney_u": float(stat),
                "p_value": float(p),
                "cohens_d": float(d) if d is not None else np.nan,
                "cliffs_delta": float(cd) if cd is not None else np.nan,
            }
        )

    if not rows:
        raise RuntimeError("No features found to test.")

    res = pd.DataFrame(rows)

    # BH correction across tested features
    res["q_value_bh_fdr"] = benjamini_hochberg(res["p_value"].tolist())

    # Sort by p-value (or q)
    res = res.sort_values(["p_value"]).reset_index(drop=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT_PATH, index=False)

    print("\n=== Mechanistic Enrichment Test (Serious vs Non-serious) ===")
    print(res[[
        "feature",
        "mean_serious",
        "mean_nonserious",
        "mean_diff_serious_minus_nonserious",
        "mean_diff_ci95_low",
        "mean_diff_ci95_high",
        "p_value",
        "q_value_bh_fdr",
        "cliffs_delta",
        "cohens_d",
    ]].to_string(index=False))

    print(f"\n✅ wrote: {OUT_PATH}")


if __name__ == "__main__":
    main()