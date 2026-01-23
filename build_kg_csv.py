"""Build KG node/edge CSVs for the Pralsetinib project.

Inputs:
  - chembl_pralsetinib_targets_clean.csv: cleaned ChEMBL targets (human single-protein targets)
  - open_targets_target_disease_long.csv: Open Targets target->disease associations (long format)
  - faers_data.xlsx: FAERS extraction (expects a 'Reactions' column)

Outputs:
  - kg_nodes.csv
  - kg_edges.csv

Edit TOP_AE to control the number of FAERS adverse-event nodes kept.
"""

from pathlib import Path
import pandas as pd
import re
from kg_go import build_go_components

CHEMBL_PATH = Path(r"data/chembl/pralsetinib_targets_clean.csv")
OT_PATH = Path(r"data/open_targets_target_disease_long.csv")
FAERS_PATH = Path(r"data/faers_data.xlsx")
OUTPUT_DIR = Path("kg_files")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DRUG_NAME = "Pralsetinib"
DRUG_CHEMBL_ID = "CHEMBL4582651"
DRUG_NODE_ID = f"drug:{DRUG_CHEMBL_ID}"

TOP_AE = 200

def split_reactions(val):
    if pd.isna(val):
        return []
    s = str(val).strip()
    if not s:
        return []
    parts = re.split(r"[;|\n]+", s)
    if len(parts) == 1:
        parts = re.split(r",\s*(?![^()]*\))", s)
    return [p.strip() for p in parts if p.strip()]

def main():
    chembl = pd.read_csv(CHEMBL_PATH, dtype=str)
    ot = pd.read_csv(OT_PATH, dtype={"score": float})
    faers = pd.read_excel(FAERS_PATH)

    # Explicit UniProt → gene symbol mapping
    uniprot_to_symbol = {
        "P07949": "RET",
        "P36888": "FLT3",
        "O60674": "JAK2",
    }
    sym_to_uniprot = {v: k for k, v in uniprot_to_symbol.items()}
    chembl["target_symbol"] = chembl["uniprot_id"].map(uniprot_to_symbol)

    # Build nodes
    nodes = []

    # Drug
    nodes.append({
        "node_id": DRUG_NODE_ID,
        "node_type": "Drug",
        "label": DRUG_NAME,
        "chembl_id": DRUG_CHEMBL_ID
    })

    # Targets
    for _, r in chembl.iterrows():
        nodes.append({
            "node_id": f"target:{r['uniprot_id']}",
            "node_type": "Target",
            "label": r.get("target_symbol") or r.get("target_name", ""),
            "uniprot_id": r["uniprot_id"],
            "chembl_target_id": r.get("chembl_target_id", ""),
        })

    # Disease
    ot_diseases = ot[["disease_id", "disease_name"]].dropna().drop_duplicates()
    for _, r in ot_diseases.iterrows():
        nodes.append({
            "node_id": f"disease:{r['disease_id']}",
            "node_type": "Disease",
            "label": r["disease_name"],
            "ontology_id": r["disease_id"],
        })

    # FAERS
    reactions = []
    for v in faers.get("Reactions", pd.Series([], dtype=object)):
        reactions.extend(split_reactions(v))

    ae_counts = pd.Series(reactions).value_counts().head(TOP_AE)

    for ae, cnt in ae_counts.items():
        nodes.append({
            "node_id": f"ae:{ae}",
            "node_type": "AdverseEvent",
            "label": ae,
            "faers_count": int(cnt),
        })

    nodes_df = (
        pd.DataFrame(nodes)
        .drop_duplicates(subset=["node_id"])
        .reset_index(drop=True)
    )

    # Build edges
    edges = []

    # Drug → Target (ChEMBL)
    for _, r in chembl.iterrows():
        edges.append({
            "source": DRUG_NODE_ID,
            "edge_type": "binds_to",
            "target": f"target:{r['uniprot_id']}",
            "provenance": "ChEMBL",
            "evidence": r.get("chembl_target_id", ""),
        })

    # Target → Disease (Open Targets)
    for _, r in ot.iterrows():
        unip = sym_to_uniprot.get(r.get("target_symbol", ""))
        if not unip:
            continue
        edges.append({
            "source": f"target:{unip}",
            "edge_type": "associated_with",
            "target": f"disease:{r['disease_id']}",
            "provenance": "OpenTargets",
            "score": float(r["score"]) if pd.notna(r["score"]) else None,
        })

    # Drug → Adverse Event (FAERS)
    for ae, cnt in ae_counts.items():
        edges.append({
            "source": DRUG_NODE_ID,
            "edge_type": "reported_with",
            "target": f"ae:{ae}",
            "provenance": "FAERS",
            "count": int(cnt),
        })

    # GO
    GO_PATH = Path("data/raw/goa_human.gaf.gz")
    TARGET_UNIPROTS = list(uniprot_to_symbol.keys())

    go_nodes_df, go_edges_df = build_go_components(
        GO_PATH,
        TARGET_UNIPROTS
    )

    # Merge GO nodes
    nodes_df = pd.concat(
        [nodes_df, go_nodes_df],
        ignore_index=True
    ).drop_duplicates(subset=["node_id"])

    # Merge GO edges
    edges_df = pd.concat(
        [pd.DataFrame(edges), go_edges_df],
        ignore_index=True
    )

    # Write outputs
    out_nodes = OUTPUT_DIR / "kg_nodes.csv"
    out_edges = OUTPUT_DIR / "kg_edges.csv"

    nodes_df.to_csv(out_nodes, index=False)
    edges_df.to_csv(out_edges, index=False)

    print(
        f"Wrote {out_nodes} ({len(nodes_df)} nodes) "
        f"and {out_edges} ({len(edges_df)} edges)."
    )

if __name__ == "__main__":
    main()
