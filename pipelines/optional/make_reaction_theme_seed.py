#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import pandas as pd

IN_PATH = Path("data/01_clean/reaction_to_theme.csv")
OUT_PATH = Path("data/01_clean/reaction_to_theme.csv")  # overwrite in place

UPDATES = {
    # Medication/admin
    "Product Prescribing Issue": "Medication Error",
    "Off Label Use": "Medication Error",
    "Product Dose Omission Issue": "Medication Error",

    # Severe outcomes
    "Death": "Severe Outcome",
    "Hospitalisation": "Severe Outcome",

    # Immune / hematologic / infection / pneumonitis bucket
    "White Blood Cell Count Decreased": "Immune System",
    "Neutropenia": "Immune System",
    "Platelet Count Decreased": "Immune System",
    "Anaemia": "Immune System",
    "Haemoglobin Decreased": "Immune System",
    "Pyrexia": "Immune System",
    "Pneumonia": "Immune System",
    "Pneumonitis": "Immune System",
    "Interstitial Lung Disease": "Immune System",
    "Rash": "Immune System",
    "Cough": "Immune System",
    "Arthralgia": "Immune System",

    # Neuro
    "Dizziness": "Neurological",
    "Headache": "Neurological",

    # Vascular/adhesion/hemodynamic
    "Hypertension": "Cell Adhesion",
    "Blood Pressure Increased": "Cell Adhesion",
    "Dyspnoea": "Cell Adhesion",

    # Cell death / injury
    "Hepatic Function Abnormal": "Cell Death",
    "Disease Progression": "Cell Death",

    # Systemic stress bucket
    "Fatigue": "Oxidative Stress",
    "Asthenia": "Oxidative Stress",
    "Decreased Appetite": "Oxidative Stress",
    "Nausea": "Oxidative Stress",
    "Diarrhoea": "Oxidative Stress",
    "Dry Mouth": "Oxidative Stress",

    # Leave these as noise bucket
    "Constipation": "Other / Unmapped",
    "Pain": "Other / Unmapped",
    "No Adverse Event": "Other / Unmapped",
}

def main():
    if not IN_PATH.exists():
        raise FileNotFoundError(f"Missing {IN_PATH}")

    df = pd.read_csv(IN_PATH)
    if "reaction" not in df.columns or "toxicity_theme" not in df.columns:
        raise ValueError(f"{IN_PATH} must have columns reaction,toxicity_theme. Found: {list(df.columns)}")

    before = df["toxicity_theme"].value_counts(dropna=False)

    # apply updates
    df["reaction"] = df["reaction"].astype(str)
    df.loc[df["reaction"].isin(UPDATES.keys()), "toxicity_theme"] = df["reaction"].map(UPDATES)

    # sanity check: show which updates did not match any row
    missing = [r for r in UPDATES.keys() if (df["reaction"] == r).sum() == 0]
    if missing:
        print("[warn] these reactions were not found in your CSV (spelling/case mismatch):")
        for r in missing:
            print("  -", r)

    after = df["toxicity_theme"].value_counts(dropna=False)

    df.to_csv(OUT_PATH, index=False)
    print(f"✅ patched and wrote: {OUT_PATH}")
    print("\n--- theme counts BEFORE ---")
    print(before.to_string())
    print("\n--- theme counts AFTER ---")
    print(after.to_string())

if __name__ == "__main__":
    main()