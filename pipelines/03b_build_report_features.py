#!/usr/bin/env python3
"""
pipelines/03b_build_report_features.py

Neuro-symbolic report-level dataset builder (Pralsetinib only).

Goal:
- Increase sample size from 8 themes -> ~1000 FAERS reports
- Keep biological grounding using KG mechanistic themes (Drug->Protein->GO->Theme)
- Enable prediction of clinical severity (Serious / Died / etc.) with explanations

Inputs:
- data/02_kg/nodes.csv
- data/02_kg/edges.csv
- data/00_raw/faers_pralsetinib_reports.xlsx  (or wherever you store it)

Outputs:
- data/03_features/faers_report_features.csv

Notes:
- We infer theme presence in a report by keyword matching against the report's "Reactions" text.
  This is a transparent, editable mapping layer (not a black box).
"""

from __future__ import annotations
from pathlib import Path
import re
import pandas as pd


KG_NODES = Path("data/02_kg/nodes.csv")
KG_EDGES = Path("data/02_kg/edges.csv")

# adjust if your raw location differs
FAERS_XLSX = Path("data/00_raw/faers_pralsetinib_reports.xlsx")

OUT = Path("data/03_features/faers_report_features.csv")


def split_reactions(s: str) -> list[str]:
    if pd.isna(s):
        return []
    txt = str(s).strip()
    if not txt or txt == "-":
        return []
    # common separators
    parts = re.split(r"[;,/]\s*|\s+\|\s+", txt)
    return [p.strip().lower() for p in parts if p.strip()]


def is_serious(row: pd.Series) -> int:
    # Primary flag
    serious = str(row.get("Serious", "")).strip().lower()
    if serious == "serious":
        return 1

    # Backup: outcomes often include “Died”, “Hospitalized”, etc.
    outcomes = str(row.get("Outcomes", "")).strip().lower()
    severe_markers = ["died", "death", "hospital", "life threatening", "disabled", "disability"]
    return int(any(m in outcomes for m in severe_markers))


def build_mechanistic_theme_scores(nodes: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a dataframe with theme_id, kg_path_count, num_unique_go, num_unique_proteins
    computed from 3-hop paths Drug->Protein->GO->Theme.
    """
    drug_id = nodes[nodes["node_type"] == "Drug"]["node_id"].iloc[0]
    protein_ids = set(nodes[nodes["node_type"] == "Protein"]["node_id"].astype(str))
    go_ids = set(nodes[nodes["node_type"] == "GO_Process"]["node_id"].astype(str))
    theme_ids = set(nodes[nodes["node_type"] == "ToxicityTheme"]["node_id"].astype(str))

    e1 = edges[(edges["edge_type"] == "binds_to") & (edges["source"] == drug_id) & (edges["target"].isin(protein_ids))]
    e2 = edges[(edges["edge_type"] == "involved_in") & (edges["source"].isin(protein_ids)) & (edges["target"].isin(go_ids))]
    e3 = edges[(edges["edge_type"] == "maps_to") & (edges["source"].isin(go_ids)) & (edges["target"].isin(theme_ids))]

    dp = e1[["source", "target"]].rename(columns={"source": "drug", "target": "protein"})
    pg = e2[["source", "target"]].rename(columns={"source": "protein", "target": "go"})
    gt = e3[["source", "target"]].rename(columns={"source": "go", "target": "theme"})

    paths = dp.merge(pg, on="protein", how="inner").merge(gt, on="go", how="inner")

    if len(paths) == 0:
        out = pd.DataFrame({"theme": sorted(theme_ids)})
        out["kg_path_count"] = 0
        out["num_unique_go"] = 0
        out["num_unique_proteins"] = 0
        return out

    out = paths.groupby("theme").agg(
        kg_path_count=("theme", "size"),
        num_unique_go=("go", "nunique"),
        num_unique_proteins=("protein", "nunique"),
    ).reset_index()
    return out


def theme_label(theme_id: str) -> str:
    # tox:Immune System -> immune system
    return theme_id.replace("tox:", "").strip().lower()


def default_theme_keyword_lexicon(theme_ids: list[str]) -> dict[str, list[str]]:
    """
    Transparent heuristic mapping from reaction text -> theme presence.
    You SHOULD edit/extend this as part of your capstone's ontology layer.
    """
    lex = {tid: [] for tid in theme_ids}

    # Fill with reasonable defaults for your current themes
    for tid in theme_ids:
        t = theme_label(tid)

        if "immune" in t:
            lex[tid] += ["infection", "sepsis", "pneumonia", "fever", "neutropenia", "leukopenia"]
        if "neurolog" in t:
            lex[tid] += ["seizure", "neuropathy", "dizziness", "headache", "stroke", "confusion"]
        if "cell death" in t or "apopt" in t:
            lex[tid] += ["death", "fatal", "cardiac arrest", "multi-organ failure"]
        if "oxidative" in t or "stress" in t:
            lex[tid] += ["oxidative", "stress", "rhabdomyolysis"]
        if "cell adhesion" in t:
            lex[tid] += ["thrombosis", "embolism", "clot", "adhesion"]
        if "severe outcome" in t:
            lex[tid] += ["death", "fatal", "hospital", "icu", "life threatening"]
        if "medication error" in t:
            lex[tid] += ["overdose", "wrong dose", "medication error", "accidental"]
        if "other" in t:
            # keep empty on purpose; it's a catch-all/noise bucket
            lex[tid] += []

    # de-dup
    for k in lex:
        lex[k] = sorted(set([w.lower() for w in lex[k] if w.strip()]))

    return lex


def main() -> None:
    if not KG_NODES.exists() or not KG_EDGES.exists():
        raise FileNotFoundError("Missing KG files. Run pipelines/02_build_kg.py first.")
    if not FAERS_XLSX.exists():
        raise FileNotFoundError(f"Missing {FAERS_XLSX}. Put your FAERS xlsx in data/00_raw/.")

    nodes = pd.read_csv(KG_NODES)
    edges = pd.read_csv(KG_EDGES)

    # Mechanistic theme scores from KG
    mech = build_mechanistic_theme_scores(nodes, edges)
    theme_ids = sorted(nodes[nodes["node_type"] == "ToxicityTheme"]["node_id"].astype(str).tolist())

    # Keyword lexicon (transparent + editable)
    lex = default_theme_keyword_lexicon(theme_ids)

    # Load FAERS
    fa = pd.read_excel(FAERS_XLSX)

    # Standardize column access
    if "Case ID" not in fa.columns or "Reactions" not in fa.columns:
        raise ValueError(f"FAERS sheet missing expected columns. Found: {list(fa.columns)}")

    # Build report-level rows
    rows = []
    for _, r in fa.iterrows():
        rxns = split_reactions(r.get("Reactions", ""))
        rxn_text = " | ".join(rxns)

        y = is_serious(r)

        feat = {
            "case_id": r.get("Case ID"),
            "y_serious": y,
            "num_reactions": len(rxns),
            "sex": str(r.get("Sex", "")).strip(),
            "patient_age": str(r.get("Patient Age", "")).strip(),
            "outcomes": str(r.get("Outcomes", "")).strip(),
        }

        # Theme presence indicators + mechanistic-weighted sums
        mech_sum_paths = 0
        mech_sum_proteins = 0
        mech_sum_go = 0

        for tid in theme_ids:
            keywords = lex.get(tid, [])
            present = int(any(kw in rxn_text for kw in keywords)) if keywords else 0
            col = "has_theme__" + theme_label(tid).replace(" ", "_").replace("/", "_")
            feat[col] = present

        # Add mechanistic scores by theme presence
        mech_map = mech.set_index("theme").to_dict(orient="index")
        for tid in theme_ids:
            col = "has_theme__" + theme_label(tid).replace(" ", "_").replace("/", "_")
            if feat[col] == 1 and tid in mech_map:
                mech_sum_paths += int(mech_map[tid]["kg_path_count"])
                mech_sum_go += int(mech_map[tid]["num_unique_go"])
                mech_sum_proteins += int(mech_map[tid]["num_unique_proteins"])

        feat["mech_sum_paths"] = mech_sum_paths
        feat["mech_sum_unique_go"] = mech_sum_go
        feat["mech_sum_unique_proteins"] = mech_sum_proteins

        rows.append(feat)

    out = pd.DataFrame(rows)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)

    print(f"✅ wrote: {OUT} (rows={len(out)}, cols={len(out.columns)})")
    print("y_serious balance:", out["y_serious"].value_counts(dropna=False).to_dict())
    print(out.head(5).to_string(index=False))
    print("\nNext: run pipelines/04_run_models.py to train + explain.\n")


if __name__ == "__main__":
    main()