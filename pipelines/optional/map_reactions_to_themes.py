#!/usr/bin/env python3

import pandas as pd
import re
from pathlib import Path

# -----------------------------
# Paths
# -----------------------------
FAERS_PATH = Path("data/01_clean/pralsetinib_faers_reports_clean.csv")
OUT_PATH = Path("data/01_clean/faers_ontology_grouped.csv")

# -----------------------------
# Ontology Map
# -----------------------------
ONTOLOGY_MAP = {
    "Death": "Severe Outcome",
    "Hospitalisation": "Severe Outcome",
    "Hospitalized": "Severe Outcome",

    "Infection": "Immune System",
    "Sepsis": "Immune System",

    "Anxiety": "Neurological",
    "Insomnia": "Neurological",

    "Product Dose Omission Issue": "Medication Error",
    "Product Prescribing Issue": "Medication Error",
}

def split_reactions(val):
    if pd.isna(val):
        return []
    parts = re.split(r"[;,\n]+", str(val))
    return [p.strip() for p in parts if p.strip()]

def map_to_ontology(reaction):
    for key in ONTOLOGY_MAP:
        if key.lower() in reaction.lower():
            return ONTOLOGY_MAP[key]
    return "Other / Unmapped"

def main():
    if not FAERS_PATH.exists():
        raise FileNotFoundError(f"Missing {FAERS_PATH}")

    # ✅ FIX: use read_csv
    df = pd.read_csv(FAERS_PATH)

    if "reactions" not in df.columns:
        raise ValueError(f"'Reactions' column not found. Found: {list(df.columns)}")

    all_reactions = []
    for r in df["reactions"]:
        all_reactions.extend(split_reactions(r))

    reactions_df = pd.DataFrame({"reaction": all_reactions})
    reactions_df["ontology_group"] = reactions_df["reaction"].apply(map_to_ontology)

    summary = (
        reactions_df
        .groupby("ontology_group")
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT_PATH, index=False)

    print("✅ FAERS reactions grouped by ontology category")
    print(summary)

if __name__ == "__main__":
    main()