#!/usr/bin/env python3
"""
pipelines/05_build_features_theme.py

Build the canonical THEME-LEVEL feature table for neuro-symbolic modeling.

Core idea:
- Clinical signal: FAERS counts via Drug -> Theme edges (reported_with, count)
- Mechanistic signal: Drug -> Protein -> GO -> Theme 3-hop paths

Optionally (recommended):
- Biologically-weighted mechanistic score using:
  * protein role weight (DrugBank interaction_role on binds_to)
  * GO evidence weight (GOA human GAF evidence codes)

Inputs:
  data/02_kg/nodes.csv
  data/02_kg/edges.csv
  (optional) data/00_raw/goa_human_annotations.gaf.gz

Outputs:
  data/03_features/theme_features.csv

Columns:
  theme
  faers_count
  kg_path_count
  num_unique_proteins
  num_unique_go
  weighted_path_score (optional; 0 if not computed)
  avg_protein_role_weight (optional)
  avg_go_evidence_weight (optional)
"""

from __future__ import annotations

import argparse
import gzip
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd


# ---------------------------
# Weights (simple + defensible)
# ---------------------------

PROTEIN_ROLE_WEIGHT = {
    "target": 3.0,
    "enzyme": 2.0,
    "transporter": 1.5,
    "carrier": 1.5,
    "other": 1.0,
    "unknown": 1.0,
}

EVIDENCE_WEIGHT = {
    # Experimental
    "EXP": 3.0, "IDA": 3.0, "IPI": 3.0, "IMP": 3.0, "IGI": 3.0, "IEP": 3.0,
    # Curated / reviewed
    "IC": 2.5, "TAS": 2.0, "NAS": 1.5,
    # Computational
    "ISS": 1.2, "ISO": 1.2, "ISA": 1.2, "ISM": 1.2, "IGC": 1.2, "IBA": 1.2, "IBD": 1.2,
    "IKR": 1.2, "IRD": 1.2,
    # Electronic
    "IEA": 1.0,
    # No data
    "ND": 0.5,
}
DEFAULT_EVIDENCE_WEIGHT = 1.0


def norm_role(role: str) -> str:
    r = str(role).strip().lower()
    if not r:
        return "unknown"
    if "target" in r:
        return "target"
    if "enzyme" in r:
        return "enzyme"
    if "transport" in r:
        return "transporter"
    if "carrier" in r:
        return "carrier"
    return r if r in PROTEIN_ROLE_WEIGHT else "other"


# ---------------------------
# Core path extraction
# ---------------------------

def compute_paths(nodes: pd.DataFrame, edges: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """Return 3-hop paths dataframe and drug node_id."""
    drug_nodes = nodes[nodes["node_type"] == "Drug"]["node_id"].tolist()
    if len(drug_nodes) != 1:
        raise ValueError(f"Expected exactly 1 Drug node. Found: {drug_nodes}")
    drug_id = drug_nodes[0]

    protein_ids = set(nodes[nodes["node_type"] == "Protein"]["node_id"].astype(str))
    go_ids = set(nodes[nodes["node_type"] == "GO_Process"]["node_id"].astype(str))
    theme_ids = set(nodes[nodes["node_type"] == "ToxicityTheme"]["node_id"].astype(str))

    e1 = edges[(edges["edge_type"] == "binds_to") & (edges["source"] == drug_id) & (edges["target"].isin(protein_ids))].copy()
    e2 = edges[(edges["edge_type"] == "involved_in") & (edges["source"].isin(protein_ids)) & (edges["target"].isin(go_ids))].copy()
    e3 = edges[(edges["edge_type"] == "maps_to") & (edges["source"].isin(go_ids)) & (edges["target"].isin(theme_ids))].copy()

    dp = e1[["target"]].rename(columns={"target": "protein"})
    pg = e2[["source", "target"]].rename(columns={"source": "protein", "target": "go"})
    gt = e3[["source", "target"]].rename(columns={"source": "go", "target": "theme"})

    paths = dp.merge(pg, on="protein", how="inner").merge(gt, on="go", how="inner")
    return paths, drug_id


def compute_unweighted_mech_features(nodes: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    """Theme-level mechanistic features from path counts."""
    paths, _ = compute_paths(nodes, edges)
    if len(paths) == 0:
        # still output all themes with zeros
        theme_ids = sorted(nodes[nodes["node_type"] == "ToxicityTheme"]["node_id"].astype(str).tolist())
        out = pd.DataFrame({"theme": theme_ids})
        out["kg_path_count"] = 0
        out["num_unique_proteins"] = 0
        out["num_unique_go"] = 0
        return out

    out = paths.groupby("theme").agg(
        kg_path_count=("theme", "size"),
        num_unique_proteins=("protein", "nunique"),
        num_unique_go=("go", "nunique"),
    ).reset_index()
    return out


def compute_faers_counts(nodes: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    """FAERS counts from Drug->Theme edges (reported_with)."""
    drug_id = nodes[nodes["node_type"] == "Drug"]["node_id"].iloc[0]
    rep = edges[(edges["edge_type"] == "reported_with") & (edges["source"] == drug_id)].copy()

    if len(rep) == 0:
        theme_ids = sorted(nodes[nodes["node_type"] == "ToxicityTheme"]["node_id"].astype(str).tolist())
        return pd.DataFrame({"theme": theme_ids, "faers_count": 0})

    if "count" in rep.columns:
        rep["faers_count"] = pd.to_numeric(rep["count"], errors="coerce").fillna(0).astype(int)
    else:
        rep["faers_count"] = 1

    out = rep.groupby("target", as_index=False)["faers_count"].sum().rename(columns={"target": "theme"})
    return out


# ---------------------------
# Optional: weighted mechanistic score using GOA GAF evidence
# ---------------------------

def parse_gaf_subset(gaf_gz: Path, uniprots_needed: set[str], go_needed: set[str]) -> Dict[Tuple[str, str], float]:
    """
    Return max evidence weight per (UniProt, GO_ID) for the subset we care about.
    GO_ID should be like GO:0006915 (no go: prefix).
    """
    pair_to_weight: Dict[Tuple[str, str], float] = {}

    with gzip.open(gaf_gz, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line or line.startswith("!"):
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 7:
                continue

            db_object_id = cols[1]  # UniProtKB:P23458
            go_id = cols[4]         # GO:0006915
            ev = cols[6].strip().upper()

            # Normalize UniProt ID
            if ":" in db_object_id:
                _, acc = db_object_id.split(":", 1)
                unip = acc.strip()
            else:
                unip = db_object_id.strip()

            if unip not in uniprots_needed:
                continue
            if go_id not in go_needed:
                continue

            w = float(EVIDENCE_WEIGHT.get(ev, DEFAULT_EVIDENCE_WEIGHT))
            key = (unip, go_id)
            if key not in pair_to_weight or w > pair_to_weight[key]:
                pair_to_weight[key] = w

    return pair_to_weight


def compute_weighted_mech_features(nodes: pd.DataFrame, edges: pd.DataFrame, gaf_gz: Path) -> pd.DataFrame:
    """
    Weighted mechanistic score per theme:
      path_weight = protein_role_weight * go_evidence_weight
      theme score = sum(path_weight)
    """
    # protein -> uniprot map
    proteins = nodes[nodes["node_type"] == "Protein"][["node_id", "uniprot_id"]].copy()
    proteins["uniprot_id"] = proteins["uniprot_id"].astype(str).str.strip()
    protein_id_to_unip = dict(zip(proteins["node_id"].astype(str), proteins["uniprot_id"]))

    paths, drug_id = compute_paths(nodes, edges)
    if len(paths) == 0:
        theme_ids = sorted(nodes[nodes["node_type"] == "ToxicityTheme"]["node_id"].astype(str).tolist())
        out = pd.DataFrame({"theme": theme_ids})
        out["weighted_path_score"] = 0.0
        out["avg_protein_role_weight"] = 0.0
        out["avg_go_evidence_weight"] = 0.0
        return out

    # Get binds_to edges to attach interaction_role
    e_bind = edges[(edges["edge_type"] == "binds_to") & (edges["source"] == drug_id)][["target"]].copy()
    e_bind = e_bind.rename(columns={"target": "protein"}).drop_duplicates()

    bind_full = edges[(edges["edge_type"] == "binds_to") & (edges["source"] == drug_id)].copy()
    if "interaction_role" not in bind_full.columns:
        bind_full["interaction_role"] = "unknown"
    bind_full["role_norm"] = bind_full["interaction_role"].map(norm_role)
    bind_full["protein_role_weight"] = bind_full["role_norm"].map(lambda r: PROTEIN_ROLE_WEIGHT.get(r, 1.0))

    role_w = bind_full[["target", "protein_role_weight"]].rename(columns={"target": "protein"}).drop_duplicates()

    # Attach weights to paths
    paths = paths.merge(role_w, on="protein", how="left")
    paths["protein_role_weight"] = paths["protein_role_weight"].fillna(1.0)

    paths["uniprot"] = paths["protein"].map(lambda pid: protein_id_to_unip.get(pid, "").strip())
    paths["go_id"] = paths["go"].astype(str).str.replace("go:", "", regex=False)  # go:GO:... -> GO:...

    uniprots_needed = set(paths["uniprot"]) - {""}
    go_needed = set(paths["go_id"]) - {""}

    pair_to_ev_weight = parse_gaf_subset(gaf_gz, uniprots_needed, go_needed)

    def ev_weight(row) -> float:
        return float(pair_to_ev_weight.get((row["uniprot"], row["go_id"]), DEFAULT_EVIDENCE_WEIGHT))

    paths["go_evidence_weight"] = paths.apply(ev_weight, axis=1)
    paths["path_weight"] = paths["protein_role_weight"] * paths["go_evidence_weight"]

    out = paths.groupby("theme").agg(
        weighted_path_score=("path_weight", "sum"),
        avg_protein_role_weight=("protein_role_weight", "mean"),
        avg_go_evidence_weight=("go_evidence_weight", "mean"),
    ).reset_index()

    return out


# ---------------------------
# Main
# ---------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", type=Path, default=Path("data/02_kg/nodes.csv"))
    ap.add_argument("--edges", type=Path, default=Path("data/02_kg/edges.csv"))
    ap.add_argument("--gaf", type=Path, default=Path("data/00_raw/goa_human_annotations.gaf.gz"),
                    help="Optional. If present, compute weighted_path_score using evidence codes.")
    ap.add_argument("--out", type=Path, default=Path("data/03_features/theme_features.csv"))
    args = ap.parse_args()

    if not args.nodes.exists():
        raise FileNotFoundError(f"Missing {args.nodes}")
    if not args.edges.exists():
        raise FileNotFoundError(f"Missing {args.edges}")

    nodes = pd.read_csv(args.nodes)
    edges = pd.read_csv(args.edges)

    # Ensure expected schema
    for c in ["node_id", "node_type"]:
        if c not in nodes.columns:
            raise ValueError(f"nodes.csv missing {c}. Found: {list(nodes.columns)}")
    for c in ["source", "target", "edge_type"]:
        if c not in edges.columns:
            raise ValueError(f"edges.csv missing {c}. Found: {list(edges.columns)}")

    theme_ids = sorted(nodes[nodes["node_type"] == "ToxicityTheme"]["node_id"].astype(str).tolist())
    base = pd.DataFrame({"theme": theme_ids})

    faers = compute_faers_counts(nodes, edges)
    mech = compute_unweighted_mech_features(nodes, edges)

    out = base.merge(faers, on="theme", how="left").merge(mech, on="theme", how="left")
    out["faers_count"] = out["faers_count"].fillna(0).astype(int)
    out["kg_path_count"] = out["kg_path_count"].fillna(0).astype(int)
    out["num_unique_proteins"] = out["num_unique_proteins"].fillna(0).astype(int)
    out["num_unique_go"] = out["num_unique_go"].fillna(0).astype(int)

    # Optional weighted features
    if args.gaf.exists():
        w = compute_weighted_mech_features(nodes, edges, args.gaf)
        out = out.merge(w, on="theme", how="left")
        out["weighted_path_score"] = out["weighted_path_score"].fillna(0.0)
        out["avg_protein_role_weight"] = out["avg_protein_role_weight"].fillna(0.0)
        out["avg_go_evidence_weight"] = out["avg_go_evidence_weight"].fillna(0.0)
        weighted_note = " (with weighted_path_score)"
    else:
        out["weighted_path_score"] = 0.0
        out["avg_protein_role_weight"] = 0.0
        out["avg_go_evidence_weight"] = 0.0
        weighted_note = " (no GAF found; weighted_path_score=0)"

    # Convenience columns
    out["has_mechanistic_support"] = (out["kg_path_count"] > 0).astype(int)

    # Stable sort for inspection
    out = out.sort_values(["weighted_path_score", "kg_path_count", "faers_count"], ascending=False).reset_index(drop=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    print(f"✅ wrote: {args.out} (rows={len(out)}, cols={len(out.columns)}){weighted_note}")
    print(out.head(12).to_string(index=False))


if __name__ == "__main__":
    main()