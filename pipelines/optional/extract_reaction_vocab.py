#!/usr/bin/env python3
"""
Extract unique reaction terms from FAERS Excel (Pralsetinib only).

Input:
  data/00_raw/faers_pralsetinib_reports.xlsx

Output:
  data/01_clean/reaction_vocab.csv
"""

from __future__ import annotations
from pathlib import Path
import re
import pandas as pd

FAERS_XLSX = Path("data/00_raw/faers_pralsetinib_reports.xlsx")
OUT = Path("data/01_clean/reaction_vocab.csv")

def split_reactions(s: str) -> list[str]:
    if pd.isna(s):
        return []
    txt = str(s).strip()
    if not txt:
        return []
    parts = re.split(r"\s*;\s*", txt)  # your file uses semicolons
    return [p.strip() for p in parts if p.strip()]

def main():
    if not FAERS_XLSX.exists():
        raise FileNotFoundError(f"Missing {FAERS_XLSX}")

    df = pd.read_excel(FAERS_XLSX)
    if "Reactions" not in df.columns:
        raise ValueError(f"Missing Reactions column. Found: {list(df.columns)}")

    vocab = {}
    for s in df["Reactions"]:
        for rxn in split_reactions(s):
            vocab[rxn] = vocab.get(rxn, 0) + 1

    out = pd.DataFrame({"reaction": list(vocab.keys()), "count": list(vocab.values())})
    out = out.sort_values("count", ascending=False).reset_index(drop=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)

    print(f"✅ wrote {OUT} (unique_reactions={len(out)})")
    print(out.head(30).to_string(index=False))

if __name__ == "__main__":
    main()