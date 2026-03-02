# models/build_drug_ae.py
import pandas as pd
import numpy as np

IN_EDGES = "../data/interim/kg_edges_with_maps_to.csv"
OUT_FEATS = "../data/processed/drug_ae_features.csv"

# -----------------------------
# Load KG
# -----------------------------
edges = pd.read_csv(IN_EDGES)
print("Loaded:", IN_EDGES)
print("Columns:", edges.columns.tolist())
print(edges.head())

# Required columns in your KG
required = {"source", "edge_type", "target"}
missing = required - set(edges.columns)
if missing:
    raise ValueError(f"Missing required columns in KG edges: {missing}")

# Optional columns
has_count = "count" in edges.columns
has_target_type = "target_type" in edges.columns
has_source_type = "source_type" in edges.columns

# -----------------------------
# Filter Drug -> AE edges
# -----------------------------
# Preferred: use explicit types if available
if has_source_type and has_target_type:
    drug_ae = edges[
        (edges["source_type"].astype(str).str.lower() == "drug") &
        (edges["target_type"].astype(str).str.lower().isin(["ae", "adverse_event", "adverse event"]))
    ].copy()
else:
    # Fallback: use node prefixes like "drug:" and "ae:" if present
    drug_ae = edges[
        edges["source"].astype(str).str.startswith("drug:") &
        edges["target"].astype(str).str.startswith("ae:")
    ].copy()

# If still empty, fallback to edge_type keyword match for FAERS edges
if drug_ae.empty:
    drug_ae = edges[
        edges["edge_type"].astype(str).str.contains("REPORTED", case=False, na=False) &
        edges["source"].astype(str).str.startswith("drug:")
    ].copy()

if drug_ae.empty:
    # Print edge types to help debug quickly
    print("\n[DEBUG] Unique edge types (top 50):")
    print(edges["edge_type"].value_counts().head(50))
    raise ValueError("No Drug→AE edges found. Check `target_type` values or AE node prefix (e.g., 'ae:').")

print(f"\nFound {len(drug_ae)} Drug→AE edges")

# -----------------------------
# Build FAERS counts per (drug, ae)
# -----------------------------
# Use provided count if present; otherwise treat each row as count=1
if has_count:
    drug_ae["_w"] = pd.to_numeric(drug_ae["count"], errors="coerce").fillna(1.0)
else:
    drug_ae["_w"] = 1.0

drug_ae_counts = (
    drug_ae
    .groupby(["source", "target"], as_index=False)["_w"]
    .sum()
    .rename(columns={"source": "drug_id", "target": "ae", "_w": "faers_count"})
)

# -----------------------------
# Simple initial features (you can enrich later)
# -----------------------------
# These are intentionally minimal so logistic regression can run end-to-end.
drug_ae_counts["path_count"] = drug_ae_counts["faers_count"]
drug_ae_counts["max_path_score"] = np.log1p(drug_ae_counts["faers_count"])
drug_ae_counts["go_overlap"] = 0.0  # placeholder until you add GO-based features
drug_ae_counts["target_faers_score"] = drug_ae_counts["faers_count"]

# Label threshold k
k = 5
drug_ae_counts["label"] = (drug_ae_counts["faers_count"] >= k).astype(int)

# -----------------------------
# Save
# -----------------------------
drug_ae_counts.to_csv(OUT_FEATS, index=False)
print(f"\n[SUCCESS] Wrote features to: {OUT_FEATS}")
print("Rows:", len(drug_ae_counts), " Positives(label=1):", int(drug_ae_counts["label"].sum()))
print(drug_ae_counts.head(10))
