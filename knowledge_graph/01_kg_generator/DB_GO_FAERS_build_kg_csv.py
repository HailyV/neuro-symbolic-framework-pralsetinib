#!/usr/bin/env python3
"""
Build KG CSVs (DrugBank + GOA + FAERS themes)

KG schema for path-based reasoning:

Drug (Pralsetinib)
  --binds_to--> Protein                     (DrugBank)
Protein
  --involved_in--> GO biological_process    (GOA / GO annotations)
GO biological_process
  --maps_to--> ToxicityTheme                (optional mapping file)
Drug
  --reported_with--> ToxicityTheme          (FAERS grouped)

Outputs:
- kg_nodes_v2.csv
- kg_edges_v2.csv

Typical usage:
  python DB_GO_FAERS_build_kg_csv.py \
    --drugbank data/DrugBank/drugbank_pralsetinib_seed_proteins_complete.csv \
    --goa data/interim/pralsetinib_targets_goa.csv \
    --faers data/interim/faers_ontology_grouped.csv \
    --out_nodes data/interim/kg_nodes_v2.csv \
    --out_edges data/interim/kg_edges_v2.csv

Optional GO->theme mapping:
  --go_theme_map data/interim/go_to_toxicity_theme.csv

IMPORTANT:
- This version assumes your GOA file columns are: uniprot_id, go_id, aspect
  (gene symbols are NOT required in GOA; we map UniProt -> gene using DrugBank)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


# ---------------------------
# Helpers
# ---------------------------

def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


def _normalize_str(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def _detect_col(df: pd.DataFrame, candidates: List[str], required: bool = True) -> Optional[str]:
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    if required:
        raise ValueError(
            f"Could not find any of these columns in input: {candidates}\n"
            f"Available columns: {list(df.columns)}"
        )
    return None


def _make_node(node_id: str, node_type: str, label: str, **attrs) -> Dict:
    d = {"node_id": node_id, "node_type": node_type, "label": label}
    d.update(attrs)
    return d


def _make_edge(source: str, edge_type: str, target: str, **attrs) -> Dict:
    d = {"source": source, "edge_type": edge_type, "target": target}
    d.update(attrs)
    return d


# ---------------------------
# Main builder
# ---------------------------

def build_kg(
    drug_name: str,
    drugbank_path: Path,
    goa_path: Path,
    faers_path: Path,
    go_theme_map_path: Optional[Path],
    out_nodes: Path,
    out_edges: Path,
    max_go_terms_per_protein: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    drug_node_id = f"drug:{drug_name}"

    # ---- DrugBank: drug -> protein
    db = _read_csv(drugbank_path)

    db_gene = _detect_col(db, ["gene_symbol", "gene", "symbol"])
    db_uniprot = _detect_col(db, ["uniprot_id", "uniprot", "accession"])
    db_category = _detect_col(db, ["category", "role", "interaction_role"], required=False)
    db_function = _detect_col(db, ["specific_function", "function"], required=False)

    db["gene_symbol_norm"] = db[db_gene].map(_normalize_str)
    db["uniprot_norm"] = db[db_uniprot].map(_normalize_str)
    db["category_norm"] = db[db_category].map(_normalize_str) if db_category else "target"
    db["function_norm"] = db[db_function].map(_normalize_str) if db_function else ""

    # Clean DB rows
    db = db[(db["gene_symbol_norm"] != "") & (db["uniprot_norm"] != "")]
    db = db.drop_duplicates(subset=["gene_symbol_norm"]).reset_index(drop=True)

    # Mapping UniProt -> gene symbol for GOA join
    uniprot_to_gene = dict(zip(db["uniprot_norm"], db["gene_symbol_norm"]))

    # ---- GOA: protein -> GO term (your file: uniprot_id, go_id, aspect)
    goa = _read_csv(goa_path)

    go_uniprot = _detect_col(goa, ["uniprot_id", "uniprot", "accession"])
    go_id = _detect_col(goa, ["go_id", "go", "go_term_id", "GO_ID"])
    go_aspect = _detect_col(goa, ["aspect", "namespace", "category"], required=False)

    goa["uniprot_norm"] = goa[go_uniprot].map(_normalize_str)
    goa["go_id_norm"] = goa[go_id].map(_normalize_str)
    goa["aspect_norm"] = goa[go_aspect].map(_normalize_str) if go_aspect else ""

    # Filter GOA to proteins in DrugBank only
    db_uniprots = set(db["uniprot_norm"])
    goa = goa[goa["uniprot_norm"].isin(db_uniprots)].copy()
    goa = goa[goa["go_id_norm"] != ""]

    # Keep biological process only when possible (P / BP / biological_process)
    if "aspect_norm" in goa.columns and goa["aspect_norm"].str.len().sum() > 0:
        goa_bp = goa[
            goa["aspect_norm"].str.upper().isin(["P", "BP", "BIOLOGICAL_PROCESS"])
            | goa["aspect_norm"].str.lower().eq("biological_process")
        ]
        if len(goa_bp) > 0:
            goa = goa_bp

    # Limit GO terms per UniProt to reduce noise
    if max_go_terms_per_protein > 0:
        goa = (
            goa.sort_values(["uniprot_norm", "go_id_norm"])
               .groupby("uniprot_norm", as_index=False)
               .head(max_go_terms_per_protein)
        )

    # ---- FAERS grouped: drug -> toxicity theme
    fa = _read_csv(faers_path)

    theme_col = _detect_col(fa, ["ontology_group", "toxicity_theme", "theme", "phenotype", "group"])
    count_col = _detect_col(fa, ["count", "n", "freq", "frequency"], required=False)

    fa["theme_norm"] = fa[theme_col].map(_normalize_str)
    fa = fa[fa["theme_norm"] != ""].copy()
    if count_col:
        fa["count_norm"] = pd.to_numeric(fa[count_col], errors="coerce").fillna(0).astype(int)
    else:
        fa["count_norm"] = 0

    # ---- Optional GO -> theme mapping
    go_theme_map = None
    if go_theme_map_path:
        gtm = _read_csv(go_theme_map_path)
        gtm_go = _detect_col(gtm, ["go_id", "go", "go_term_id", "GO_ID"])
        gtm_theme = _detect_col(gtm, ["toxicity_theme", "theme"])
        gtm["go_id_norm"] = gtm[gtm_go].map(_normalize_str)
        gtm["theme_norm"] = gtm[gtm_theme].map(_normalize_str)
        gtm = gtm[(gtm["go_id_norm"] != "") & (gtm["theme_norm"] != "")]
        go_theme_map = gtm.drop_duplicates(subset=["go_id_norm", "theme_norm"]).reset_index(drop=True)

    # ---------------------------
    # Build Nodes
    # ---------------------------
    nodes: List[Dict] = []

    # Drug node
    nodes.append(_make_node(drug_node_id, "Drug", drug_name))

    # Protein nodes (from DrugBank)
    for _, r in db.iterrows():
        gene = r["gene_symbol_norm"]
        nodes.append(
            _make_node(
                f"protein:{gene}",
                "Protein",
                gene,
                gene_symbol=gene,
                uniprot_id=r["uniprot_norm"],
                category=r["category_norm"],
                specific_function=r["function_norm"],
                source="DrugBank",
            )
        )

    # GO nodes (only those used)
    go_used = goa.drop_duplicates(subset=["go_id_norm"]).reset_index(drop=True)
    for _, r in go_used.iterrows():
        goid = r["go_id_norm"]
        nodes.append(
            _make_node(
                f"go:{goid}",
                "GO_Process",
                goid,  # no GO name in your file, use GO ID as label
                go_id=goid,
                source="GOA",
            )
        )

    # Toxicity theme nodes (from FAERS, and from mapping if present)
    themes = set(fa["theme_norm"].tolist())
    if go_theme_map is not None:
        themes |= set(go_theme_map["theme_norm"].tolist())

    for t in sorted(themes):
        nodes.append(_make_node(f"tox:{t}", "ToxicityTheme", t, source="FAERS/Mapping"))

    nodes_df = pd.DataFrame(nodes).drop_duplicates(subset=["node_id"]).reset_index(drop=True)

    # ---------------------------
    # Build Edges
    # ---------------------------
    edges: List[Dict] = []

    # Drug -> Protein (DrugBank)
    for _, r in db.iterrows():
        gene = r["gene_symbol_norm"]
        edges.append(
            _make_edge(
                drug_node_id,
                "binds_to",
                f"protein:{gene}",
                provenance="DrugBank",
                interaction_role=r["category_norm"],
            )
        )

    # Protein -> GO (GOA, joined via UniProt -> gene)
    for _, r in goa.iterrows():
        unip = r["uniprot_norm"]
        gene = uniprot_to_gene.get(unip)
        if not gene:
            continue
        goid = r["go_id_norm"]
        edges.append(
            _make_edge(
                f"protein:{gene}",
                "involved_in",
                f"go:{goid}",
                provenance="GOA",
            )
        )

    # GO -> Theme (optional mapping)
    if go_theme_map is not None:
        for _, r in go_theme_map.iterrows():
            goid = r["go_id_norm"]
            theme = r["theme_norm"]
            edges.append(
                _make_edge(
                    f"go:{goid}",
                    "maps_to",
                    f"tox:{theme}",
                    provenance="ManualMapping",
                )
            )

    # Drug -> Theme (FAERS)
    for _, r in fa.iterrows():
        theme = r["theme_norm"]
        edges.append(
            _make_edge(
                drug_node_id,
                "reported_with",
                f"tox:{theme}",
                provenance="FAERS",
                count=int(r["count_norm"]),
            )
        )

    edges_df = pd.DataFrame(edges).drop_duplicates(
        subset=["source", "edge_type", "target", "provenance"]
    ).reset_index(drop=True)

    # Write outputs
    out_nodes.parent.mkdir(parents=True, exist_ok=True)
    out_edges.parent.mkdir(parents=True, exist_ok=True)
    nodes_df.to_csv(out_nodes, index=False)
    edges_df.to_csv(out_edges, index=False)

    return nodes_df, edges_df


def main() -> None:
    p = argparse.ArgumentParser(description="Build KG CSVs (DrugBank + GOA + FAERS).")

    p.add_argument("--drug_name", default="Pralsetinib",
                   help="Drug label used in KG node_id (default: Pralsetinib).")

    p.add_argument("--drugbank", required=True, type=Path,
                   help="DrugBank seed proteins CSV "
                        "(e.g., data/DrugBank/drugbank_pralsetinib_seed_proteins_complete.csv)")
    p.add_argument("--goa", required=True, type=Path,
                   help="Protein->GO annotations CSV (must include uniprot_id, go_id, aspect)")
    p.add_argument("--faers", required=True, type=Path,
                   help="FAERS grouped themes CSV (e.g., data/interim/faers_ontology_grouped.csv)")

    p.add_argument("--go_theme_map", default=None, type=Path,
                   help="Optional GO->toxicity theme mapping CSV with columns go_id + theme/toxicity_theme.")

    p.add_argument("--out_nodes", default=Path("data/interim/kg_nodes_v2.csv"), type=Path,
                   help="Output nodes CSV path (default: data/interim/kg_nodes_v2.csv)")
    p.add_argument("--out_edges", default=Path("data/interim/kg_edges_v2.csv"), type=Path,
                   help="Output edges CSV path (default: data/interim/kg_edges_v2.csv)")

    p.add_argument("--max_go_terms_per_protein", default=50, type=int,
                   help="Limit GO terms per UniProt to reduce noise (default: 50). Use 0 for no limit.")

    args = p.parse_args()

    nodes_df, edges_df = build_kg(
        drug_name=args.drug_name,
        drugbank_path=args.drugbank,
        goa_path=args.goa,
        faers_path=args.faers,
        go_theme_map_path=args.go_theme_map if args.go_theme_map else None,
        out_nodes=args.out_nodes,
        out_edges=args.out_edges,
        max_go_terms_per_protein=args.max_go_terms_per_protein,
    )

    print(f"✅ Wrote nodes: {args.out_nodes}  (n={len(nodes_df)})")
    print(f"✅ Wrote edges: {args.out_edges}  (n={len(edges_df)})")

    if args.go_theme_map is None:
        print("\nNext step: add a GO->toxicity mapping file to enable Drug→Protein→GO→Theme path scoring.")
        print("Example columns: go_id,toxicity_theme")


if __name__ == "__main__":
    main()
