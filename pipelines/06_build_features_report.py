#!/usr/bin/env python3
"""
pipelines/03b_build_report_features_from_mapping.py

Build report-level features using an explicit reaction->theme mapping, and add a
biologically-weighted mechanistic score per report (mech_weighted_sum).

Inputs:
  data/00_raw/faers_pralsetinib_reports.xlsx
  data/01_clean/reaction_to_theme.csv
  data/02_kg/nodes.csv
  data/02_kg/edges.csv
  data/03_features/theme_mechanistic_weighted.csv   <-- produced by 03c

Output:
  data/03_features/faers_report_features.csv

Notes:
- Theme indicator columns are aligned to KG ToxicityTheme labels (tox: prefix removed).
- y_serious comes from FAERS "Serious" column and/or severe markers in "Outcomes".
"""

from __future__ import annotations

from pathlib import Path
import re
import pandas as pd

FAERS_XLSX = Path("data/00_raw/faers_pralsetinib_reports.xlsx")
MAP_CSV = Path("data/01_clean/reaction_to_theme.csv")

KG_NODES = Path("data/02_kg/nodes.csv")
KG_EDGES = Path("data/02_kg/edges.csv")

THEME_MECH = Path("data/03_features/theme_mechanistic_weighted.csv")

OUT = Path("data/03_features/faers_report_features.csv")


def split_reactions(s: str) -> list[str]:
    if pd.isna(s):
        return []
    txt = str(s).strip()
    if not txt:
        return []
    # Your FAERS file uses semicolons
    return [p.strip() for p in re.split(r"\s*;\s*", txt) if p.strip()]

def is_serious(row: pd.Series) -> int:
    serious = str(row.get("Serious", "")).strip().lower()
    if serious == "serious":
        return 1
    outcomes = str(row.get("Outcomes", "")).strip().lower()
    severe_markers = ["died", "death", "hospital", "life threatening", "disabled", "disability", "icu", "intensive care"]
    return int(any(m in outcomes for m in severe_markers))


def safe_col(s: str) -> str:
    return s.strip().lower().replace(" ", "_").replace("/", "_").replace("-", "_")


def build_mech_theme_counts(nodes: pd.DataFrame, edges: pd.DataFrame) -> dict[str, dict[str, int]]:
    """
    Unweighted mechanistic support per theme from 3-hop Drug->Protein->GO->Theme paths.
    Returns: {theme_node_id: {kg_path_count, num_unique_go, num_unique_proteins}}
    """
    drug_id = nodes[nodes["node_type"] == "Drug"]["node_id"].iloc[0]
    protein_ids = set(nodes[nodes["node_type"] == "Protein"]["node_id"].astype(str))
    go_ids = set(nodes[nodes["node_type"] == "GO_Process"]["node_id"].astype(str))
    theme_ids = set(nodes[nodes["node_type"] == "ToxicityTheme"]["node_id"].astype(str))

    e1 = edges[(edges["edge_type"] == "binds_to") & (edges["source"] == drug_id) & (edges["target"].isin(protein_ids))]
    e2 = edges[(edges["edge_type"] == "involved_in") & (edges["source"].isin(protein_ids)) & (edges["target"].isin(go_ids))]
    e3 = edges[(edges["edge_type"] == "maps_to") & (edges["source"].isin(go_ids)) & (edges["target"].isin(theme_ids))]

    dp = e1[["target"]].rename(columns={"target": "protein"})
    pg = e2[["source", "target"]].rename(columns={"source": "protein", "target": "go"})
    gt = e3[["source", "target"]].rename(columns={"source": "go", "target": "theme"})

    paths = dp.merge(pg, on="protein").merge(gt, on="go")

    mech: dict[str, dict[str, int]] = {}
    if len(paths) == 0:
        for tid in theme_ids:
            mech[tid] = {"kg_path_count": 0, "num_unique_go": 0, "num_unique_proteins": 0}
        return mech

    agg = paths.groupby("theme").agg(
        kg_path_count=("theme", "size"),
        num_unique_go=("go", "nunique"),
        num_unique_proteins=("protein", "nunique"),
    ).reset_index()

    for _, r in agg.iterrows():
        tid = str(r["theme"])
        mech[tid] = {
            "kg_path_count": int(r["kg_path_count"]),
            "num_unique_go": int(r["num_unique_go"]),
            "num_unique_proteins": int(r["num_unique_proteins"]),
        }

    for tid in theme_ids:
        if tid not in mech:
            mech[tid] = {"kg_path_count": 0, "num_unique_go": 0, "num_unique_proteins": 0}

    return mech


def main() -> None:
    # required files
    for p in [FAERS_XLSX, MAP_CSV, KG_NODES, KG_EDGES, THEME_MECH]:
        if not p.exists():
            raise FileNotFoundError(f"Missing {p}")

    fa = pd.read_excel(FAERS_XLSX)
    if "Case ID" not in fa.columns or "Reactions" not in fa.columns:
        raise ValueError(f"FAERS missing 'Case ID' or 'Reactions'. Found: {list(fa.columns)}")

    mapping = pd.read_csv(MAP_CSV)
    if "reaction" not in mapping.columns or "toxicity_theme" not in mapping.columns:
        raise ValueError(f"{MAP_CSV} must have columns reaction,toxicity_theme. Found: {list(mapping.columns)}")

    rxn_to_theme = dict(zip(mapping["reaction"].astype(str), mapping["toxicity_theme"].astype(str)))

    nodes = pd.read_csv(KG_NODES)
    edges = pd.read_csv(KG_EDGES)

    # Unweighted mechanistic counts per theme
    mech_counts = build_mech_theme_counts(nodes, edges)

    # Weighted mechanistic score per theme (from 03c)
    theme_mech = pd.read_csv(THEME_MECH)
    if "theme" not in theme_mech.columns or "weighted_path_score" not in theme_mech.columns:
        raise ValueError(f"{THEME_MECH} must have theme,weighted_path_score. Found: {list(theme_mech.columns)}")
    weighted_by_theme = theme_mech.set_index("theme")["weighted_path_score"].to_dict()

    # KG themes (canonical)
    theme_ids = sorted(nodes[nodes["node_type"] == "ToxicityTheme"]["node_id"].astype(str).tolist())
    theme_labels = {tid: tid.replace("tox:", "") for tid in theme_ids}

    rows = []
    for _, r in fa.iterrows():
        rxns = split_reactions(r.get("Reactions", ""))
        themes_present = set()

        for rxn in rxns:
            themes_present.add(rxn_to_theme.get(rxn, "Other / Unmapped"))

        feat = {
            "case_id": r.get("Case ID"),
            "y_serious": is_serious(r),
            "num_reactions": len(rxns),
            "sex": str(r.get("Sex", "")).strip(),
            "patient_age": str(r.get("Patient Age", "")).strip(),
            "outcomes": str(r.get("Outcomes", "")).strip(),
        }

        mech_sum_paths = 0
        mech_sum_go = 0
        mech_sum_proteins = 0
        mech_weighted_sum = 0.0

        for tid in theme_ids:
            label = theme_labels[tid]
            col = f"has_theme__{safe_col(label)}"
            present = int(label in themes_present)
            feat[col] = present

            if present:
                mech_sum_paths += mech_counts[tid]["kg_path_count"]
                mech_sum_go += mech_counts[tid]["num_unique_go"]
                mech_sum_proteins += mech_counts[tid]["num_unique_proteins"]
                mech_weighted_sum += float(weighted_by_theme.get(tid, 0.0))

        feat["mech_sum_paths"] = mech_sum_paths
        feat["mech_sum_unique_go"] = mech_sum_go
        feat["mech_sum_unique_proteins"] = mech_sum_proteins
        feat["mech_weighted_sum"] = mech_weighted_sum

        rows.append(feat)

    out = pd.DataFrame(rows)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"✅ wrote: {OUT} (rows={len(out)}, cols={len(out.columns)})")
    print("y_serious balance:", out["y_serious"].value_counts().to_dict())
    print(out.head(5).to_string(index=False))


if __name__ == "__main__":
    main()