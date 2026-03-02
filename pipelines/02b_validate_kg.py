#!/usr/bin/env python3
"""
pipelines/02b_validate_kg.py

Sanity checks for the canonical knowledge graph.

Reads:
  data/02_kg/nodes.csv
  data/02_kg/edges.csv

Prints:
  - node counts by type
  - edge counts by relation/type
  - existence of Drug->Protein->GO->Theme paths
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd


def main() -> None:
    nodes_path = Path("data/02_kg/nodes.csv")
    edges_path = Path("data/02_kg/edges.csv")

    if not nodes_path.exists():
        raise FileNotFoundError(f"Missing {nodes_path}")
    if not edges_path.exists():
        raise FileNotFoundError(f"Missing {edges_path}")

    nodes = pd.read_csv(nodes_path)
    edges = pd.read_csv(edges_path)

    # Your KG schema (confirmed from your printout)
    # nodes: node_id, node_type, label, ...
    # edges: (likely) source, target, edge_type OR similar
    node_id_col = "node_id" if "node_id" in nodes.columns else "id"
    node_type_col = "node_type" if "node_type" in nodes.columns else "type"

    # Edges can vary by script; detect columns
    src_col = "source" if "source" in edges.columns else ("src" if "src" in edges.columns else None)
    tgt_col = "target" if "target" in edges.columns else ("dst" if "dst" in edges.columns else None)
    rel_col = None
    for cand in ["edge_type", "relation", "type", "predicate"]:
        if cand in edges.columns:
            rel_col = cand
            break

    if src_col is None or tgt_col is None or rel_col is None:
        raise ValueError(
            f"edges.csv missing expected columns. Found: {list(edges.columns)}. "
            f"Need source/target + one of edge_type/relation/type/predicate."
        )

    nodes = nodes.rename(columns={node_id_col: "id", node_type_col: "type"})
    edges = edges.rename(columns={src_col: "source", tgt_col: "target", rel_col: "relation"})

    print("\n=== KG VALIDATION ===")
    print(f"Nodes: {len(nodes):,}")
    print(f"Edges: {len(edges):,}")

    print("\n-- Node counts by type --")
    print(nodes["type"].value_counts(dropna=False).to_string())

    print("\n-- Edge counts by relation --")
    print(edges["relation"].value_counts(dropna=False).to_string())

    # Try to confirm 3-hop paths: Drug -> Protein -> GO -> Theme
    drug_ids = set(nodes.loc[nodes["type"].astype(str).str.lower().eq("drug"), "id"])
    protein_ids = set(nodes.loc[nodes["type"].astype(str).str.lower().eq("protein"), "id"])

    # GO types can be named differently; match anything containing 'go'
    go_ids = set(nodes.loc[nodes["type"].astype(str).str.lower().str.contains("go"), "id"])
    # Theme types can also vary; match 'toxicity' or 'theme'
    theme_ids = set(nodes.loc[nodes["type"].astype(str).str.lower().str.contains("toxicity|theme"), "id"])

    e1 = edges[(edges["relation"] == "binds_to") & (edges["source"].isin(drug_ids)) & (edges["target"].isin(protein_ids))]
    e2 = edges[(edges["relation"] == "involved_in") & (edges["source"].isin(protein_ids)) & (edges["target"].isin(go_ids))]
    e3 = edges[(edges["relation"] == "maps_to") & (edges["source"].isin(go_ids)) & (edges["target"].isin(theme_ids))]

    print("\n-- 3-hop reasoning path check --")
    print(f"binds_to (Drug->Protein): {len(e1):,}")
    print(f"involved_in (Protein->GO): {len(e2):,}")
    print(f"maps_to (GO->Theme): {len(e3):,}")

    if len(e1) and len(e2) and len(e3):
        dp = e1[["source", "target"]].rename(columns={"source": "drug", "target": "protein"})
        pg = e2[["source", "target"]].rename(columns={"source": "protein", "target": "go"})
        gt = e3[["source", "target"]].rename(columns={"source": "go", "target": "theme"})
        paths = dp.merge(pg, on="protein", how="inner").merge(gt, on="go", how="inner")
        print(f"[ok] total Drug→Protein→GO→Theme paths: {len(paths):,}")
        print("Top themes by path count (top 10):")
        print(paths.groupby('theme').size().sort_values(ascending=False).head(10).to_string())
    else:
        print("[warn] Full 3-hop paths not confirmed (often due to missing maps_to in edges, or theme node_type naming).")

    print("\n[ok] done.\n")


if __name__ == "__main__":
    main()