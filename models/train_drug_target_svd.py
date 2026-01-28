import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from sklearn.decomposition import TruncatedSVD

KG_EDGES = "analysis/kg_files/kg_edges.csv"
DB_EDGES = "data/interim/kg_edges_drug_protein.csv"
DB_NODES = "data/interim/kg_nodes_v2.csv"
GOA_FILE = "data/interim/pralsetinib_targets_goa.csv"



DRUG_QUERY = "drug:CHEMBL4582651"
K_RECS = 20
LATENT_DIM = 16  

def load_edges():
    # --- ChEMBL edges (already in analysis/kg_files/kg_edges.csv) ---
    kg = pd.read_csv(KG_EDGES)

    # Expect kg_edges.csv to have columns: source, target, edge_type OR your v2 schema
    if set(["source", "target", "edge_type"]).issubset(kg.columns):
        chembl_dt = kg[kg["edge_type"].eq("binds_to")].copy()
        chembl_dt = chembl_dt[chembl_dt["source"].str.startswith("drug:") &
                              chembl_dt["target"].str.startswith("target:")]
        chembl_dt = chembl_dt[["source", "target"]].drop_duplicates()
        chembl_dt["provenance"] = "ChEMBL"
    else:
        # v2-like schema
        chembl_dt = kg[kg["edge_type"].eq("binds_to")].copy()
        # keep drug -> target
        chembl_dt = chembl_dt[(chembl_dt["source_type"].str.lower() == "drug") &
                              (chembl_dt["target_type"].str.lower().isin(["target", "protein"]))]

        chembl_dt["source"] = "drug:" + chembl_dt["source_id"].astype(str)
        chembl_dt["target"] = "target:" + chembl_dt["target_id"].astype(str)
        chembl_dt = chembl_dt[["source", "target"]].drop_duplicates()
        chembl_dt["provenance"] = chembl_dt.get("evidence", "ChEMBL")

    # --- DrugBank-derived / combined edges file you showed ---
    db_edges = pd.read_csv(DB_EDGES)

    # keep drug -> protein/target interactions
    db_dt = db_edges[db_edges["edge_type"].eq("interacts_with")].copy()
    db_dt = db_dt[(db_dt["source_type"].str.lower() == "drug") &
                  (db_dt["target_type"].str.lower().isin(["protein", "target"]))]

    db_dt["source"] = "drug:" + db_dt["source_id"].astype(str)
    db_dt["target"] = "target:" + db_dt["target_id"].astype(str)

    db_dt = db_dt[["source", "target"]].drop_duplicates()
    db_dt["provenance"] = db_edges.get("evidence", "DrugBank").iloc[0] if "evidence" in db_edges.columns else "DrugBank"

    # Combine
    all_dt = pd.concat([chembl_dt, db_dt], ignore_index=True).drop_duplicates()
    return all_dt

def build_matrix(dt_edges: pd.DataFrame):
    drugs = sorted(dt_edges["source"].unique())
    targets = sorted(dt_edges["target"].unique())

    drug2i = {d:i for i, d in enumerate(drugs)}
    targ2j = {t:j for j, t in enumerate(targets)}

    rows = dt_edges["source"].map(drug2i).to_numpy()
    cols = dt_edges["target"].map(targ2j).to_numpy()
    data = np.ones(len(dt_edges), dtype=np.float32)

    X = coo_matrix(
        (data, (rows, cols)),
        shape=(len(drugs), len(targets))
    ).tocsr()

    return X, drugs, targets, drug2i, targ2j


def fit_svd(X, k):
    k = min(k, X.shape[0] - 1, X.shape[1] - 1)
    if k < 2:
        raise ValueError(
            f"Not enough drugs/targets for SVD. Need at least 2x2, got {X.shape}."
        )
    svd = TruncatedSVD(n_components=k, random_state=0)
    U = svd.fit_transform(X)              # (n_drugs, k)
    V = svd.components_.T                 # (n_targets, k)
    return U, V

def recommend_for_drug(dt_edges, X, drugs, targets, drug2i, U, V, drug_id, topk=20):
    if drug_id not in drug2i:
        raise KeyError(f"Drug {drug_id} not found. Available: {drugs[:5]} ...")

    i = drug2i[drug_id]
    scores = U[i] @ V.T  # (n_targets,)

    # exclude known targets
    known = set(dt_edges.loc[dt_edges["source"].eq(drug_id), "target"])
    cand = [(targets[j], float(scores[j])) for j in range(len(targets)) if targets[j] not in known]
    cand.sort(key=lambda x: x[1], reverse=True)

    return cand[:topk], known

def load_goa(goa_path: str):
    goa = pd.read_csv(goa_path)

    # Try common column name patterns
    # We want: uniprot_id + go_id/go_term
    cols = {c.lower(): c for c in goa.columns}

    # best guesses
    uniprot_col = cols.get("uniprot_id") or cols.get("uniprot") or cols.get("target_id") or cols.get("db_object_id")
    go_col = cols.get("go_id") or cols.get("go") or cols.get("go_term") or cols.get("go_identifier")

    if uniprot_col is None or go_col is None:
        raise ValueError(f"Can't find UniProt and GO columns in {goa_path}. Columns: {goa.columns.tolist()}")

    goa = goa[[uniprot_col, go_col]].dropna()
    goa = goa.rename(columns={uniprot_col: "uniprot_id", go_col: "go_id"})
    goa["uniprot_id"] = goa["uniprot_id"].astype(str)
    goa["go_id"] = goa["go_id"].astype(str)
    return goa

def rank_targets_by_go_overlap(dt_edges: pd.DataFrame, drug_id: str, goa: pd.DataFrame, topk: int = 20):
    # known targets in "target:UNIPROT" format
    known_targets = set(dt_edges.loc[dt_edges["source"] == drug_id, "target"])
    known_uniprots = {t.split("target:", 1)[1] for t in known_targets if t.startswith("target:")}

    # Build GO sets for each protein
    prot2go = goa.groupby("uniprot_id")["go_id"].apply(set).to_dict()

    # Union GO terms of known targets (the "profile")
    known_go = set()
    for u in known_uniprots:
        known_go |= prot2go.get(u, set())

    if not known_go:
        print("No GO terms found for known targets; can't do GO overlap ranking.")
        return [], known_targets

    # Score every protein by Jaccard overlap with known_go
    candidates = []
    for u, gos in prot2go.items():
        t_id = "target:" + u
        if t_id in known_targets:
            continue
        inter = len(gos & known_go)
        if inter == 0:
            continue
        union = len(gos | known_go)
        score = inter / union
        candidates.append((t_id, score, inter, union))

    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[:topk], known_targets


def main():
    dt_edges = load_edges()
    print(f"Total drug-target edges: {len(dt_edges)}")
    print(f"Unique drugs: {dt_edges['source'].nunique()}, unique targets: {dt_edges['target'].nunique()}")

    n_drugs = dt_edges["source"].nunique()
    n_targets = dt_edges["target"].nunique()

    if n_drugs < 2 or n_targets < 2:
        print("\nNot enough data for SVD (need >=2 drugs and >=2 targets).")
        print("Using GO-overlap baseline instead.")

        goa = load_goa(GOA_FILE)
        recs, known = rank_targets_by_go_overlap(
            dt_edges,
            DRUG_QUERY,
            goa,
            K_RECS
        )

        print("\nKnown targets for", DRUG_QUERY, "=", len(known))
        for t in sorted(list(known))[:15]:
            print("  ", t)

        print("\nTop candidate NEW targets (GO overlap baseline):")
        if not recs:
            print("  (No candidates had GO overlap — try a larger GOA/protein file.)")
        else:
            for t, score, inter, union in recs:
                print(f"  {t:18s}  score={score:.4f}  overlap={inter}/{union}")

        return   # <-- IMPORTANT: stops execution before SVD

    # ---- NORMAL PATH: SVD (used once you have multiple drugs) ----
    X, drugs, targets, drug2i, _ = build_matrix(dt_edges)
    U, V = fit_svd(X, LATENT_DIM)

    recs, known = recommend_for_drug(
        dt_edges,
        X,
        drugs,
        targets,
        drug2i,
        U,
        V,
        DRUG_QUERY,
        K_RECS
    )

    print("\nKnown targets for", DRUG_QUERY, "=", len(known))
    for t in sorted(list(known))[:15]:
        print("  ", t)

    print("\nTop predicted NEW targets:")
    for t, s in recs:
        print(f"  {t:18s}  score={s:.4f}")


if __name__ == "__main__":
    main()
