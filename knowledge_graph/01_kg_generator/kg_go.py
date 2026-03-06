from pathlib import Path
import pandas as pd

def build_go_components(go_gaf_path, target_uniprot_ids):
    go = pd.read_csv(
        go_gaf_path,
        sep="\t",
        comment="!",
        header=None,
        compression="gzip",
        low_memory=False
    )

    go.columns = [
        "db", "uniprot_id", "gene_symbol", "qualifier",
        "go_id", "reference", "evidence",
        "with_from", "aspect", "gene_name",
        "synonym", "gene_type", "taxon",
        "date", "assigned_by", "annotation_ext", "gene_product_form"
    ]

    go_sub = go[
        (go["uniprot_id"].isin(target_uniprot_ids)) &
        (go["aspect"] == "P")
    ]

    go_nodes = (
        go_sub[["go_id", "gene_name"]]
        .dropna()
        .drop_duplicates()
        .rename(columns={"gene_name": "label"})
    )

    go_nodes["node_id"] = "go:" + go_nodes["go_id"]
    go_nodes["node_type"] = "BiologicalProcess"

    go_edges = go_sub[["uniprot_id", "go_id", "evidence"]].copy()
    go_edges["source"] = "target:" + go_edges["uniprot_id"]
    go_edges["target"] = "go:" + go_edges["go_id"]
    go_edges["edge_type"] = "involved_in"
    go_edges["provenance"] = "GeneOntology"

    go_edges = go_edges[
        ["source", "edge_type", "target", "provenance", "evidence"]
    ]

    return go_nodes, go_edges
