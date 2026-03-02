#!/usr/bin/env python3
"""
pipelines/03_build_kg.py

Build Pralsetinib KG CSVs from CLEAN inputs.

Reads:
  - data/01_clean/pralsetinib_seed_proteins.csv
  - data/01_clean/pralsetinib_targets_goa.csv
  - data/01_clean/faers_ontology_grouped.csv
  - data/01_clean/go_to_toxicity_theme.csv            (may be empty)
  - OPTIONAL: data/01_clean/go_basic_terms.csv         (GO name/namespace labels)

Writes:
  - data/02_kg/nodes.csv
  - data/02_kg/edges.csv
"""

from __future__ import annotations
import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


# -------------------
# Helpers
# -------------------
def _s(x) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    return str(x).strip()


def norm_go_id(x) -> str:
    """
    Canonical GO id: GO:0006915
    Accepts: GO:..., go:GO:..., go:go:GO:..., embedded strings.
    """
    s = _s(x)
    if not s:
        return ""
    while s.lower().startswith("go:"):
        s = s[3:].strip()
    if s.startswith("GO:"):
        return s
    m = re.search(r"(GO:\d{7})", s)
    return m.group(1) if m else ""


def go_node(goid: str) -> str:
    gid = norm_go_id(goid)
    return f"go:{gid}" if gid else ""


def prot_node(gene: str) -> str:
    return f"protein:{_s(gene).upper()}"


def tox_node(theme: str) -> str:
    return f"tox:{_s(theme)}"


def make_node(node_id: str, node_type: str, label: str, **attrs) -> Dict:
    d = {"node_id": node_id, "node_type": node_type, "label": label}
    d.update(attrs)
    return d


def make_edge(source: str, edge_type: str, target: str, **attrs) -> Dict:
    d = {"source": source, "edge_type": edge_type, "target": target}
    d.update(attrs)
    return d


# -------------------
# Builder
# -------------------
def build_kg(
    drug_name: str,
    seed_proteins_csv: Path,
    targets_goa_csv: Path,
    faers_grouped_csv: Path,
    go_theme_map_csv: Path,
    go_terms_csv: Optional[Path],
    out_nodes: Path,
    out_edges: Path,
    max_go_terms_per_protein: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    drug_node = f"drug:{drug_name}"

    # ---- Load seed proteins
    db = pd.read_csv(seed_proteins_csv)
    if not {"gene_symbol", "uniprot_id"}.issubset(db.columns):
        raise ValueError(f"{seed_proteins_csv} must have gene_symbol,uniprot_id. Found: {list(db.columns)}")
    db["gene_symbol"] = db["gene_symbol"].map(_s).str.upper()
    db["uniprot_id"] = db["uniprot_id"].map(_s)
    if "category" not in db.columns:
        db["category"] = "target"
    db["category"] = db["category"].map(_s).str.lower()
    db = db[(db["gene_symbol"] != "") & (db["uniprot_id"] != "")].drop_duplicates(subset=["uniprot_id", "gene_symbol"])
    uniprot_to_gene = dict(zip(db["uniprot_id"], db["gene_symbol"]))
    db_uniprots = set(db["uniprot_id"])

    # ---- Load GOA subset
    goa = pd.read_csv(targets_goa_csv)
    if not {"uniprot_id", "go_id"}.issubset(goa.columns):
        raise ValueError(f"{targets_goa_csv} must have uniprot_id,go_id. Found: {list(goa.columns)}")
    goa["uniprot_id"] = goa["uniprot_id"].map(_s)
    goa["go_id"] = goa["go_id"].map(norm_go_id)

    # Keep BP only if that filter leaves anything
    if "aspect" in goa.columns:
        goa["aspect"] = goa["aspect"].map(_s).str.upper()
        bp = goa[goa["aspect"].eq("P")].copy()
        if len(bp) > 0:
            goa = bp

    before = len(goa)
    goa = goa[goa["uniprot_id"].isin(db_uniprots)].copy()
    goa = goa[goa["go_id"].astype(str).str.startswith("GO:")].copy()
    after = len(goa)

    # Cap per protein (optional)
    if max_go_terms_per_protein > 0:
        goa = (
            goa.sort_values(["uniprot_id", "go_id"])
               .groupby("uniprot_id", as_index=False)
               .head(max_go_terms_per_protein)
        )

    # ---- Load FAERS grouped
    fa = pd.read_csv(faers_grouped_csv)
    if not {"ontology_group", "count"}.issubset(fa.columns):
        raise ValueError(f"{faers_grouped_csv} must have ontology_group,count. Found: {list(fa.columns)}")
    fa["ontology_group"] = fa["ontology_group"].map(_s)
    fa["count"] = pd.to_numeric(fa["count"], errors="coerce").fillna(0).astype(int)
    fa = fa[fa["ontology_group"] != ""].copy()

    # ---- Load GO->Theme mapping (may be empty)
    gtm = pd.read_csv(go_theme_map_csv)
    if not {"go_id", "toxicity_theme"}.issubset(gtm.columns):
        raise ValueError(f"{go_theme_map_csv} must have go_id,toxicity_theme. Found: {list(gtm.columns)}")
    gtm["go_id"] = gtm["go_id"].map(norm_go_id)
    gtm["toxicity_theme"] = gtm["toxicity_theme"].map(_s)
    gtm = gtm[(gtm["go_id"].astype(str).str.startswith("GO:")) & (gtm["toxicity_theme"] != "")].drop_duplicates()

    # ---- Optional GO names
    go_name: Dict[str, str] = {}
    go_ns: Dict[str, str] = {}
    if go_terms_csv and go_terms_csv.exists():
        gt = pd.read_csv(go_terms_csv)
        if {"go_id", "go_name"}.issubset(gt.columns):
            gt["go_id"] = gt["go_id"].map(norm_go_id)
            gt["go_name"] = gt["go_name"].map(_s)
            go_name = dict(zip(gt["go_id"], gt["go_name"]))
        if "namespace" in gt.columns:
            gt["namespace"] = gt["namespace"].map(_s)
            go_ns = dict(zip(gt["go_id"], gt["namespace"]))

    # --------------------
    # Nodes
    # --------------------
    nodes: List[Dict] = [make_node(drug_node, "Drug", drug_name)]

    # Protein nodes
    for _, r in db.iterrows():
        gene = r["gene_symbol"]
        nodes.append(make_node(
            prot_node(gene), "Protein", gene,
            gene_symbol=gene, uniprot_id=r["uniprot_id"], category=r["category"],
            source="DrugBank_clean"
        ))

    # GO nodes from GOA (THIS is what was missing for you)
    go_used = sorted(set(goa["go_id"].tolist()))
    for gid in go_used:
        label = go_name.get(gid, gid)
        nodes.append(make_node(
            go_node(gid), "GO_Process", label,
            go_id=gid, go_name=go_name.get(gid, ""), namespace=go_ns.get(gid, ""),
            source="GOA/go-basic" if go_name else "GOA_subset"
        ))

    # Theme nodes (union of FAERS + GO mapping themes)
    themes = set(fa["ontology_group"].tolist()) | set(gtm["toxicity_theme"].tolist())
    for t in sorted(themes):
        nodes.append(make_node(tox_node(t), "ToxicityTheme", t, source="FAERS/Mapping"))

    nodes_df = pd.DataFrame(nodes).drop_duplicates(subset=["node_id"]).reset_index(drop=True)

    # --------------------
    # Edges
    # --------------------
    edges: List[Dict] = []

    # Drug -> Protein
    for _, r in db.iterrows():
        edges.append(make_edge(
            drug_node, "binds_to", prot_node(r["gene_symbol"]),
            provenance="DrugBank_clean", interaction_role=r["category"]
        ))

    # Protein -> GO (this should be ~1083 rows minus capping)
    for _, r in goa.iterrows():
        gene = uniprot_to_gene.get(r["uniprot_id"], "")
        if not gene:
            continue
        tgt = go_node(r["go_id"])
        if not tgt:
            continue
        e = make_edge(prot_node(gene), "involved_in", tgt, provenance="GOA_subset")
        if "evidence_code" in goa.columns:
            e["evidence_code"] = _s(r.get("evidence_code", ""))
        edges.append(e)

    # GO -> Theme (may be 0 until mapping works)
    for _, r in gtm.iterrows():
        src = go_node(r["go_id"])
        tgt = tox_node(r["toxicity_theme"])
        if not src or not tgt:
            continue
        edges.append(make_edge(src, "maps_to", tgt, provenance="GO_mapping"))

    # Drug -> Theme (FAERS)
    for _, r in fa.iterrows():
        edges.append(make_edge(
            drug_node, "reported_with", tox_node(r["ontology_group"]),
            provenance="FAERS_grouped", count=int(r["count"])
        ))

    edges_df = pd.DataFrame(edges).drop_duplicates(
        subset=["source", "edge_type", "target", "provenance"]
    ).reset_index(drop=True)

    # Write outputs
    out_nodes.parent.mkdir(parents=True, exist_ok=True)
    out_edges.parent.mkdir(parents=True, exist_ok=True)
    nodes_df.to_csv(out_nodes, index=False)
    edges_df.to_csv(out_edges, index=False)

    # Print sanity (you NEED this)
    print("\n=== KG SANITY ===")
    print(f"seed proteins: {len(db)}")
    print(f"GOA rows before filter: {before}")
    print(f"GOA rows after filter : {after}")
    print(f"GOA rows after cap    : {len(goa)}")
    print(f"unique GO terms       : {len(go_used)}")
    print(f"FAERS themes          : {fa['ontology_group'].nunique()}")
    print(f"GO->Theme rows        : {len(gtm)}")

    return nodes_df, edges_df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--drug_name", default="Pralsetinib")
    ap.add_argument("--seed_proteins", type=Path, default=Path("data/01_clean/pralsetinib_seed_proteins.csv"))
    ap.add_argument("--goa", type=Path, default=Path("data/01_clean/pralsetinib_targets_goa.csv"))
    ap.add_argument("--faers_grouped", type=Path, default=Path("data/01_clean/faers_ontology_grouped.csv"))
    ap.add_argument("--go_theme_map", type=Path, default=Path("data/01_clean/go_to_toxicity_theme.csv"))
    ap.add_argument("--go_terms", type=Path, default=Path("data/01_clean/go_basic_terms.csv"))
    ap.add_argument("--out_nodes", type=Path, default=Path("data/02_kg/nodes.csv"))
    ap.add_argument("--out_edges", type=Path, default=Path("data/02_kg/edges.csv"))
    ap.add_argument("--max_go_terms_per_protein", type=int, default=50)
    args = ap.parse_args()

    nodes_df, edges_df = build_kg(
        drug_name=args.drug_name,
        seed_proteins_csv=args.seed_proteins,
        targets_goa_csv=args.goa,
        faers_grouped_csv=args.faers_grouped,
        go_theme_map_csv=args.go_theme_map,
        go_terms_csv=args.go_terms if args.go_terms.exists() else None,
        out_nodes=args.out_nodes,
        out_edges=args.out_edges,
        max_go_terms_per_protein=args.max_go_terms_per_protein,
    )

    print(f"\n✅ wrote nodes: {args.out_nodes} (n={len(nodes_df)})")
    print(f"✅ wrote edges: {args.out_edges} (n={len(edges_df)})")
    print("\nnode_type counts:")
    print(nodes_df["node_type"].value_counts().to_string())
    print("\nedge_type counts:")
    print(edges_df["edge_type"].value_counts().to_string())


if __name__ == "__main__":
    main()