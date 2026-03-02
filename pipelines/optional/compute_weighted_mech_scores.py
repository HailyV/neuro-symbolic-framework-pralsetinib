#!/usr/bin/env python3
"""
pipelines/03c_build_weighted_mech_scores.py

Compute biologically-weighted mechanistic support per ToxicityTheme by summing
Drug->Protein->GO->Theme paths with weights:

- Protein role weight: from binds_to edge attribute "interaction_role" (DrugBank)
- GO evidence weight: derived from GOA human GAF evidence codes for (UniProt, GO)

Inputs:
  data/02_kg/nodes.csv
  data/02_kg/edges.csv
  data/00_raw/goa_human_annotations.gaf.gz

Output:
  data/03_features/theme_mechanistic_weighted.csv
"""

from __future__ import annotations
from pathlib import Path
import gzip
import pandas as pd

NODES = Path("data/02_kg/nodes.csv")
EDGES = Path("data/02_kg/edges.csv")
GOA_GAF_GZ = Path("data/00_raw/goa_human_annotations.gaf.gz")

OUT = Path("data/03_features/theme_mechanistic_weighted.csv")


# --- weights you can justify in writeup (simple + defensible) ---
PROTEIN_ROLE_WEIGHT = {
    "target": 3.0,
    "enzyme": 2.0,
    "transporter": 1.5,
    "carrier": 1.5,
    "other": 1.0,
    "unknown": 1.0,
}

# GO evidence code weights (curated experimental > author statement > electronic)
EVIDENCE_WEIGHT = {
    # Experimental
    "EXP": 3.0, "IDA": 3.0, "IPI": 3.0, "IMP": 3.0, "IGI": 3.0, "IEP": 3.0,
    # Curated / reviewed
    "IC": 2.5, "TAS": 2.0, "NAS": 1.5,
    # Computational
    "ISS": 1.2, "ISO": 1.2, "ISA": 1.2, "ISM": 1.2, "IGC": 1.2, "IBA": 1.2, "IBD": 1.2, "IKR": 1.2, "IRD": 1.2,
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
    # common DrugBank-ish values: target / enzyme / transporter / carrier
    if "target" in r:
        return "target"
    if "enzyme" in r:
        return "enzyme"
    if "transport" in r:
        return "transporter"
    if "carrier" in r:
        return "carrier"
    return r if r in PROTEIN_ROLE_WEIGHT else "other"


def parse_gaf_subset(gaf_gz: Path, uniprots_needed: set[str], go_needed: set[str]) -> dict[tuple[str, str], float]:
    """
    Parse GAF and return max evidence weight per (UniProt, GO_ID) for the subset we care about.
    GO_ID should be like GO:0006915 (no go: prefix).
    """
    pair_to_weight: dict[tuple[str, str], float] = {}

    with gzip.open(gaf_gz, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line or line.startswith("!"):
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 7:
                continue

            # GAF 2.x columns:
            # 1 DB, 2 DB Object ID, 3 DB Object Symbol, 4 Qualifier, 5 GO ID, 6 DB:Reference, 7 Evidence Code, ...
            db_object_id = cols[1]  # e.g., UniProtKB:P23458 or Q9...
            go_id = cols[4]         # e.g., GO:0006915
            ev = cols[6].strip().upper()

            # Normalize UniProt ID
            if ":" in db_object_id:
                prefix, acc = db_object_id.split(":", 1)
                # UniProtKB:P23458 -> P23458
                unip = acc.strip()
            else:
                unip = db_object_id.strip()

            if unip not in uniprots_needed:
                continue
            if go_id not in go_needed:
                continue

            w = EVIDENCE_WEIGHT.get(ev, DEFAULT_EVIDENCE_WEIGHT)
            key = (unip, go_id)
            if key not in pair_to_weight or w > pair_to_weight[key]:
                pair_to_weight[key] = w

    return pair_to_weight


def main():
    for p in [NODES, EDGES, GOA_GAF_GZ]:
        if not p.exists():
            raise FileNotFoundError(f"Missing {p}")

    nodes = pd.read_csv(NODES)
    edges = pd.read_csv(EDGES)

    drug_id = nodes[nodes["node_type"] == "Drug"]["node_id"].iloc[0]

    # Node sets + UniProt mapping for proteins
    proteins = nodes[nodes["node_type"] == "Protein"][["node_id", "uniprot_id"]].copy()
    proteins["uniprot_id"] = proteins["uniprot_id"].astype(str).str.strip()
    protein_id_to_unip = dict(zip(proteins["node_id"].astype(str), proteins["uniprot_id"]))

    go_nodes = set(nodes[nodes["node_type"] == "GO_Process"]["node_id"].astype(str))
    theme_nodes = set(nodes[nodes["node_type"] == "ToxicityTheme"]["node_id"].astype(str))

    # Edges
    e_bind = edges[(edges["edge_type"] == "binds_to") & (edges["source"] == drug_id)].copy()
    e_go = edges[edges["edge_type"] == "involved_in"].copy()
    e_map = edges[edges["edge_type"] == "maps_to"].copy()

    # Keep only endpoints that exist in node sets
    e_bind = e_bind[e_bind["target"].isin(protein_id_to_unip.keys())].copy()
    e_go = e_go[e_go["source"].isin(protein_id_to_unip.keys()) & e_go["target"].isin(go_nodes)].copy()
    e_map = e_map[e_map["source"].isin(go_nodes) & e_map["target"].isin(theme_nodes)].copy()

    # Build 3-hop paths with protein role attribute
    # binds_to edges should have interaction_role from DrugBank builder
    if "interaction_role" not in e_bind.columns:
        e_bind["interaction_role"] = "unknown"
    e_bind["role_norm"] = e_bind["interaction_role"].map(norm_role)
    e_bind["protein_role_weight"] = e_bind["role_norm"].map(lambda r: PROTEIN_ROLE_WEIGHT.get(r, 1.0))

    dp = e_bind[["target", "protein_role_weight"]].rename(columns={"target": "protein"})
    pg = e_go[["source", "target"]].rename(columns={"source": "protein", "target": "go"})
    gt = e_map[["source", "target"]].rename(columns={"source": "go", "target": "theme"})

    paths = dp.merge(pg, on="protein").merge(gt, on="go")

    if len(paths) == 0:
        raise RuntimeError("No 3-hop paths found. Run KG validation first.")

    # Determine which (UniProt, GO) pairs we need evidence for
    paths["uniprot"] = paths["protein"].map(lambda pid: protein_id_to_unip.get(pid, "").strip())
    paths["go_id"] = paths["go"].str.replace("go:", "", regex=False)  # go:GO:0006915 -> GO:0006915

    uniprots_needed = set(paths["uniprot"]) - {""}
    go_needed = set(paths["go_id"]) - {""}

    pair_to_ev_weight = parse_gaf_subset(GOA_GAF_GZ, uniprots_needed, go_needed)

    # Assign evidence weight; default if missing
    def ev_w(row) -> float:
        key = (row["uniprot"], row["go_id"])
        return float(pair_to_ev_weight.get(key, DEFAULT_EVIDENCE_WEIGHT))

    paths["go_evidence_weight"] = paths.apply(ev_w, axis=1)

    # Weighted path score
    paths["path_weight"] = paths["protein_role_weight"] * paths["go_evidence_weight"]

    # Aggregate per theme
    out = paths.groupby("theme").agg(
        weighted_path_score=("path_weight", "sum"),
        path_count=("theme", "size"),
        avg_protein_role_weight=("protein_role_weight", "mean"),
        avg_go_evidence_weight=("go_evidence_weight", "mean"),
        unique_proteins=("protein", "nunique"),
        unique_go=("go", "nunique"),
    ).reset_index()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)

    print(f"✅ wrote: {OUT} (rows={len(out)})")
    print(out.sort_values("weighted_path_score", ascending=False).head(10).to_string(index=False))


if __name__ == "__main__":
    main()