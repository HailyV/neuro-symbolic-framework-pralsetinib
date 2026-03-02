#!/usr/bin/env python3
"""
pipelines/04_run_models.py

A) Theme-level ranking models (baseline / KG / hybrid) from data/03_features/theme_features.csv
B) Report-level logistic regression (predict seriousness) from data/03_features/faers_report_features.csv

Outputs:
- data/results/theme_rankings.csv
- data/results/report_logreg_coefficients.csv
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score


THEME_FEATURES = Path("data/03_features/theme_features.csv")
REPORT_FEATURES = Path("data/03_features/faers_report_features.csv")
OUT_DIR = Path("data/results")


def safe_log1p(x: pd.Series) -> pd.Series:
    return np.log1p(pd.to_numeric(x, errors="coerce").fillna(0))


def run_theme_rankings() -> None:
    df = pd.read_csv(THEME_FEATURES)

    df["score_faers"] = df["faers_count"]
    df["score_kg"] = df["kg_path_count"]
    df["score_hybrid"] = safe_log1p(df["faers_count"]) + 2.0 * df["kg_path_count"]

    out = df[
        [
            "theme",
            "faers_count",
            "kg_path_count",
            "num_unique_go",
            "num_unique_proteins",
            "score_faers",
            "score_kg",
            "score_hybrid",
        ]
    ].copy()

    out = out.sort_values("score_hybrid", ascending=False).reset_index(drop=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "theme_rankings.csv"
    out.to_csv(out_path, index=False)
    print(f"✅ wrote: {out_path}")
    print(out.head(12).to_string(index=False))


def run_report_logistic_regression() -> None:
    df = pd.read_csv(REPORT_FEATURES)
    if "y_serious" not in df.columns:
        raise ValueError("report features missing y_serious. Run 03b builder first.")

    # Build feature list (prefer weighted mechanistic score)
    feature_cols: list[str] = []

    for c in ["num_reactions", "mech_weighted_sum", "mech_sum_paths", "mech_sum_unique_go", "mech_sum_unique_proteins"]:
        if c in df.columns:
            feature_cols.append(c)

    # Mechanistic-only features (no theme flags)
    feature_cols = []

    for c in [
        "num_reactions",
        "mech_weighted_sum",
        "mech_sum_paths",
        "mech_sum_unique_go",
        "mech_sum_unique_proteins",
    ]:
        if c in df.columns:
            feature_cols.append(c)

    X = df[feature_cols].fillna(0)
    y = df["y_serious"].astype(int)

    # 🚫 Remove leakage feature if present
    X = X.drop(columns=["has_theme__severe_outcome"], errors="ignore")

    model = LogisticRegression(max_iter=5000, solver="liblinear")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs, accs = [], []
    for train, test in skf.split(X, y):
        model.fit(X.iloc[train], y.iloc[train])
        prob = model.predict_proba(X.iloc[test])[:, 1]
        pred = (prob >= 0.5).astype(int)
        aucs.append(roc_auc_score(y.iloc[test], prob))
        accs.append(accuracy_score(y.iloc[test], pred))

    model.fit(X, y)

    coef = pd.DataFrame({"feature": X.columns.to_list(), "coef": model.coef_[0]})
    coef = coef.sort_values("coef", ascending=False).reset_index(drop=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    coef_path = OUT_DIR / "report_logreg_coefficients.csv"
    coef.to_csv(coef_path, index=False)

    print("\nReport-level Logistic Regression (5-fold CV)")
    print(f"AUC mean={np.mean(aucs):.3f} std={np.std(aucs):.3f}")
    print(f"ACC mean={np.mean(accs):.3f} std={np.std(accs):.3f}")
    print(f"✅ wrote: {coef_path}")

    print("\nTop positive coefficients (more severe):")
    print(coef.head(15).to_string(index=False))

    print("\nTop negative coefficients (less severe):")
    print(coef.tail(15).to_string(index=False))


def main() -> None:
    if THEME_FEATURES.exists():
        print("=== Theme-level rankings ===")
        run_theme_rankings()
    else:
        print(f"[skip] missing {THEME_FEATURES}")

    if REPORT_FEATURES.exists():
        print("\n=== Report-level ML (severity prediction) ===")
        run_report_logistic_regression()
    else:
        print(f"[skip] missing {REPORT_FEATURES}")


if __name__ == "__main__":
    main()