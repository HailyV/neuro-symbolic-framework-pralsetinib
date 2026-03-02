#!/usr/bin/env python3
"""
pipelines/03_build_features.py

Build theme-level feature table from canonical KG.

Reads:
  data/02_kg/nodes.csv
  data/02_kg/edges.csv

Writes:
  data/03_features/theme_features.csv

Features per ToxicityTheme:
  - faers_count: from Drug->Theme reported_with edge attribute "count"
  - kg_path_count: number of Drug->Protein->GO->Theme 3-hop paths
  - num_unique_go: number of distinct GO nodes participating in those paths
  - num_unique_proteins: number of distinct proteins participating in those paths
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd


NODES_PATH = Path("data/02_kg/nodes.csv")
EDGES_PATH = Path("data/02_kg/edges.csv")
OUT_PATH = Path("data/03_features/theme_features.csv")


def main() -> None:
    if not NODES_PATH.exists():
        raise FileNotFoundError(f"Missing {NODES_PATH}")
    if not EDGES_PATH.exists():
        raise FileNotFoundError(f"Missing {EDGES_PATH}")

    nodes = pd.read_csv(NODES_PATH)
    edges = pd.read_csv(EDGES_PATH)

    # Basic schema checks
    for c in ["node_id", "node_type"]:
        if c not in nodes.columns:
            raise ValueError(f"nodes.csv missing {c}. Found: {list(nodes.columns)}")
    for c in ["source", "target", "edge_type"]:
        if c not in edges.columns:
            raise ValueError(f"edges.csv missing {c}. Found: {list(edges.columns)}")

    # Identify drug node
    drug_nodes = nodes[nodes["node_type"] == "Drug"]["node_id"].tolist()
    if len(drug_nodes) != 1:
        raise ValueError(f"Expected exactly 1 Drug node, found {len(drug_nodes)}: {drug_nodes}")
    drug_id = drug_nodes[0]

    # Identify node sets
    protein_ids = set(nodes[nodes["node_type"] == "Protein"]["node_id"].astype(str))
    go_ids = set(nodes[nodes["node_type"] == "GO_Process"]["node_id"].astype(str))
    theme_ids = set(nodes[nodes["node_type"] == "ToxicityTheme"]["node_id"].astype(str))

    # --- FAERS counts: Drug -> Theme edges
    rep = edges[(edges["edge_type"] == "reported_with") & (edges["source"] == drug_id)].copy()
    if "count" in rep.columns:
        rep["faers_count"] = pd.to_numeric(rep["count"], errors="coerce").fillna(0).astype(int)
    else:
        # fallback: treat each edge as 1 if no count column
        rep["faers_count"] = 1

    faers_by_theme = rep.groupby("target", as_index=False)["faers_count"].sum()
    faers_by_theme = faers_by_theme.rename(columns={"target": "theme"})

    # --- 3-hop paths: Drug -> Protein -> GO -> Theme
    e1 = edges[(edges["edge_type"] == "binds_to") & (edges["source"] == drug_id) & (edges["target"].isin(protein_ids))]
    e2 = edges[(edges["edge_type"] == "involved_in") & (edges["source"].isin(protein_ids)) & (edges["target"].isin(go_ids))]
    e3 = edges[(edges["edge_type"] == "maps_to") & (edges["source"].isin(go_ids)) & (edges["target"].isin(theme_ids))]

    dp = e1[["source", "target"]].rename(columns={"source": "drug", "target": "protein"})
    pg = e2[["source", "target"]].rename(columns={"source": "protein", "target": "go"})
    gt = e3[["source", "target"]].rename(columns={"source": "go", "target": "theme"})

    paths = dp.merge(pg, on="protein", how="inner").merge(gt, on="go", how="inner")

    # Aggregate mechanistic features per theme
    if len(paths) == 0:
        mech = pd.DataFrame({"theme": sorted(theme_ids)})
        mech["kg_path_count"] = 0
        mech["num_unique_go"] = 0
        mech["num_unique_proteins"] = 0
    else:
        mech = paths.groupby("theme").agg(
            kg_path_count=("theme", "size"),
            num_unique_go=("go", "nunique"),
            num_unique_proteins=("protein", "nunique"),
        ).reset_index()

    # --- combine into final feature table
    features = pd.DataFrame({"theme": sorted(theme_ids)})
    features = features.merge(faers_by_theme, on="theme", how="left")
    features = features.merge(mech, on="theme", how="left")

    # fill missing values
    for c in ["faers_count", "kg_path_count", "num_unique_go", "num_unique_proteins"]:
        if c not in features.columns:
            features[c] = 0
        features[c] = features[c].fillna(0).astype(int)

    # Helpful derived columns
    features["has_mechanistic_support"] = (features["kg_path_count"] > 0).astype(int)

    # Sort: most interesting first
    features = features.sort_values(["kg_path_count", "faers_count"], ascending=False).reset_index(drop=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(OUT_PATH, index=False)

    print(f"✅ wrote: {OUT_PATH} (rows={len(features)}, cols={len(features.columns)})")
    print(features.head(12).to_string(index=False))


if __name__ == "__main__":
    main()