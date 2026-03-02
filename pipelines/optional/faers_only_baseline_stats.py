#!/usr/bin/env python3
"""
pipelines/05a_faers_only_enrichment_test.py

Test whether FAERS frequency information alone separates serious vs non-serious reports.

Inputs:
  - data/03_features/faers_report_features.csv  (has_theme__* columns + y_serious)
  - data/results/theme_rankings.csv             (theme + faers_count)

Output:
  - prints results
  - writes data/results/faers_only_enrichment_results.csv

Method:
  - Build per-report FAERS-only score:
      faers_only_sum = sum_over_themes_present( faers_count(theme) )
  - Mann–Whitney U test + Cliff's delta + Cohen's d
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

REPORTS = Path("data/03_features/faers_report_features.csv")
THEME_RANK = Path("data/results/theme_rankings.csv")
OUT = Path("data/results/faers_only_enrichment_results.csv")


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(float)
    b = b.astype(float)
    va = np.var(a, ddof=1) if len(a) > 1 else 0.0
    vb = np.var(b, ddof=1) if len(b) > 1 else 0.0
    pooled = np.sqrt((va + vb) / 2.0) if (va + vb) > 0 else np.nan
    if pooled == 0 or np.isnan(pooled):
        return np.nan
    return (np.mean(a) - np.mean(b)) / pooled


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    a = np.sort(a.astype(float))
    b = np.sort(b.astype(float))
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return np.nan

    gt = 0
    lt = 0
    j = 0
    k = 0

    for i in range(n):
        while j < m and b[j] < a[i]:
            j += 1
        gt += j

    for i in range(n):
        while k < m and b[k] <= a[i]:
            k += 1
        lt += (m - k)

    return (gt - lt) / (n * m)


def main() -> None:
    if not REPORTS.exists():
        raise FileNotFoundError(f"Missing {REPORTS}")
    if not THEME_RANK.exists():
        raise FileNotFoundError(f"Missing {THEME_RANK}. Run pipelines/04_run_models.py first.")

    df = pd.read_csv(REPORTS)
    tr = pd.read_csv(THEME_RANK)

    if "y_serious" not in df.columns:
        raise ValueError("Missing y_serious in report features.")
    if "theme" not in tr.columns or "faers_count" not in tr.columns:
        raise ValueError(f"{THEME_RANK} must have columns theme, faers_count")

    # Map theme label -> faers_count
    # theme file uses "tox:Immune System" format
    theme_to_count = dict(zip(tr["theme"].astype(str), pd.to_numeric(tr["faers_count"], errors="coerce").fillna(0)))

    # Find report theme columns
    theme_cols = sorted([c for c in df.columns if c.startswith("has_theme__")])
    if not theme_cols:
        raise ValueError("No has_theme__ columns found in report features.")

    # Convert has_theme__cell_death -> tox:Cell Death
    def col_to_theme(c: str) -> str:
        label = c.replace("has_theme__", "")
        label = label.replace("___", " / ").replace("__", " ").replace("_", " ")
        # Your file uses "Other / Unmapped" and similar labels; this reconstruction matches most cases.
        # If a theme label mismatches, it will just get 0.
        # Capitalization: title-case generally matches your tox labels.
        label = " ".join([w.capitalize() if w not in ["/"] else w for w in label.split(" ")])
        return "tox:" + label

    col_theme = {c: col_to_theme(c) for c in theme_cols}

    # Build faers-only score
    faers_only = np.zeros(len(df), dtype=float)
    for c in theme_cols:
        theme = col_theme[c]
        w = float(theme_to_count.get(theme, 0.0))
        faers_only += w * df[c].fillna(0).astype(float).to_numpy()

    df["faers_only_sum"] = faers_only

    serious = df[df["y_serious"] == 1]["faers_only_sum"].to_numpy()
    nonser = df[df["y_serious"] == 0]["faers_only_sum"].to_numpy()

    stat, p = mannwhitneyu(serious, nonser, alternative="two-sided")
    cd = cliffs_delta(serious, nonser)
    d = cohens_d(serious, nonser)

    out = pd.DataFrame([{
        "feature": "faers_only_sum",
        "mean_serious": float(np.mean(serious)),
        "mean_nonserious": float(np.mean(nonser)),
        "mean_diff_serious_minus_nonserious": float(np.mean(serious) - np.mean(nonser)),
        "mannwhitney_u": float(stat),
        "p_value": float(p),
        "cliffs_delta": float(cd),
        "cohens_d": float(d),
        "n_serious": int(len(serious)),
        "n_nonserious": int(len(nonser)),
    }])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)

    print("\n=== FAERS-only enrichment test (Serious vs Non-serious) ===")
    print(out.to_string(index=False))
    print(f"\n✅ wrote: {OUT}")

    # sanity: show theme label mismatches (optional)
    missing = [t for t in col_theme.values() if t not in theme_to_count]
    if missing:
        print("\n[warn] Some theme labels didn't match theme_rankings.csv (these got weight 0).")
        print("Examples:", missing[:10])


if __name__ == "__main__":
    main()