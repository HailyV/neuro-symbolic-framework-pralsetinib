#!/usr/bin/env python3
"""
Normalize legacy KG IDs so edges match nodes.

Input (legacy, read-only):
  legacy/analysis/data/interim/kg_nodes_v2.csv
  legacy/analysis/data/interim/kg_edges_with_maps_to.csv

Output (normalized copy):
  data/02_kg/legacy/nodes.csv
  data/02_kg/legacy/edges.csv
"""

from pathlib import Path
import pandas as pd

IN_NODES = Path("legacy/analysis/data/interim/kg_nodes_v2.csv")
IN_EDGES = Path("legacy/analysis/data/interim/kg_edges_with_maps_to.csv")

OUT_DIR = Path("data/02_kg/legacy")
OUT_NODES = OUT_DIR / "nodes.csv"
OUT_EDGES = OUT_DIR / "edges.csv"


def normalize_node_id(x: str) -> str:
    if pd.isna(x):
        return x
    s = str(x).strip()

    # Legacy maps_to sources use GO:xxxxxxx, but node ids often use go:GO:xxxxxxx
    if s.startswith("GO:"):
        return "go:" + s

    # Some legacy nodes might already be go:GO:...
    return s


def normalize_edge_endpoint(x: str) -> str:
    if pd.isna(x):
        return x
    s = str(x).strip()

    # Convert GO:xxxxxxx -> go:GO:xxxxxxx to match nodes
    if s.startswith("GO:"):
        return "go:" + s

    # Leave tox:... and drug:/protein:... alone
    return s


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    nodes = pd.read_csv(IN_NODES)
    edges = pd.read_csv(IN_EDGES)

    # Normalize node ids
    if "node_id" not in nodes.columns:
        raise ValueError(f"Expected node_id in nodes. Found: {list(nodes.columns)}")
    nodes["node_id"] = nodes["node_id"].map(normalize_node_id)

    # Normalize edge endpoints
    for c in ["source", "target"]:
        if c not in edges.columns:
            raise ValueError(f"Expected {c} in edges. Found: {list(edges.columns)}")
        edges[c] = edges[c].map(normalize_edge_endpoint)

    # Drop edges whose endpoints don't exist (optional but helpful)
    node_ids = set(nodes["node_id"].astype(str))
    before = len(edges)
    edges = edges[edges["source"].astype(str).isin(node_ids) & edges["target"].astype(str).isin(node_ids)].copy()
    after = len(edges)

    nodes.to_csv(OUT_NODES, index=False)
    edges.to_csv(OUT_EDGES, index=False)

    print(f"[ok] wrote {OUT_NODES}")
    print(f"[ok] wrote {OUT_EDGES}")
    print(f"[ok] kept {after}/{before} edges after endpoint filter")


if __name__ == "__main__":
    main()