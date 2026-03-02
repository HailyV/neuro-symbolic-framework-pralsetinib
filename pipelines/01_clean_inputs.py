#!/usr/bin/env python3
"""
pipelines/01_clean_inputs.py

Minimal, research-grade cleaning for the Pralsetinib neuro-symbolic project.

What it does (bottom-up):
- Reads *raw* inputs from data/00_raw/
- Writes *cleaned, standardized* outputs to data/01_clean/
- Does NOT touch legacy/ or derived folders
- Safe to re-run: overwrites outputs in 01_clean

It cleans the files you need for KG construction (Step 02):
1) DrugBank seed proteins (targets/enzymes/transporters)
2) GOA human annotations GAF (subset to Pralsetinib proteins)
3) FAERS Pralsetinib XLSX (standardize, plus reaction long/count tables)

Optional extras (cleaned if present):
- ChEMBL targets CSV
- OpenTargets target-disease long CSV
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import gzip
import json
import re
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


# ----------------------------
# Helpers
# ----------------------------

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _norm_col(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s

def _write_df(df: pd.DataFrame, out: Path) -> None:
    _ensure_dir(out.parent)
    df.to_csv(out, index=False)

def _read_csv_auto(path: Path) -> pd.DataFrame:
    """
    Read CSV that might be comma or semicolon delimited.
    """
    # Try comma first; if it's one column with semicolons, re-read with ;
    df = pd.read_csv(path)
    if df.shape[1] == 1 and ";" in df.columns[0]:
        df = pd.read_csv(path, sep=";")
    return df

def _safe_strip(x):
    if pd.isna(x):
        return x
    return str(x).strip()

def _canon_uniprot(x: str) -> str:
    """
    Uniprot IDs typically like: Q9H2X6 or P12345.
    Sometimes files have prefixes (e.g. 'UniProtKB:Q9H2X6').
    """
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    s = str(x).strip()
    s = s.replace("UniProtKB:", "").replace("uniprotkb:", "")
    s = s.split("|")[0].strip()
    return s

def _split_multi(s: str, sep: str = ";") -> list[str]:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return []
    parts = [p.strip() for p in str(s).split(sep)]
    return [p for p in parts if p and p != "-"]


# ----------------------------
# 1) DrugBank seed proteins
# ----------------------------

def clean_drugbank_seed_proteins(raw_csv: Path, out_csv: Path) -> dict:
    df = pd.read_csv(raw_csv)

    # Standardize column names
    df.columns = [_norm_col(c) for c in df.columns]

    # Expected in your repo: drug, gene_symbol, uniprot_id, specific_function, category
    # We'll be tolerant to mild variations.
    rename = {}
    if "gene_symbol" not in df.columns:
        # common variants
        for alt in ["gene", "symbol", "genesymbol"]:
            if alt in df.columns:
                rename[alt] = "gene_symbol"
                break
    if "uniprot_id" not in df.columns:
        for alt in ["uniprot", "uniprotid", "uniprotkb"]:
            if alt in df.columns:
                rename[alt] = "uniprot_id"
                break
    if rename:
        df = df.rename(columns=rename)

    required = {"gene_symbol", "uniprot_id"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"DrugBank seed proteins missing required columns: {missing}. "
                         f"Have: {list(df.columns)}")

    df["gene_symbol"] = df["gene_symbol"].map(_safe_strip).str.upper()
    df["uniprot_id"] = df["uniprot_id"].map(_canon_uniprot)

    # Optional: normalize category/role
    if "category" in df.columns:
        df["category"] = df["category"].map(_safe_strip).str.lower()

    # Drop empties and duplicates
    df = df[df["uniprot_id"].astype(str).str.len() > 0].copy()
    df = df.drop_duplicates(subset=["uniprot_id", "gene_symbol"])

    _write_df(df, out_csv)

    return {
        "rows": int(df.shape[0]),
        "unique_uniprot": int(df["uniprot_id"].nunique()),
        "out": str(out_csv),
    }


# ----------------------------
# 2) GOA human annotations (GAF) -> subset to Pralsetinib proteins
# ----------------------------

GAF_COLS = [
    "db", "db_object_id", "db_object_symbol", "qualifier", "go_id",
    "db_reference", "evidence_code", "with_from", "aspect",
    "db_object_name", "db_object_synonym", "db_object_type", "taxon",
    "date", "assigned_by", "annotation_extension", "gene_product_form_id"
]

def _iter_gaf_rows(gaf_gz: Path) -> Iterable[list[str]]:
    with gzip.open(gaf_gz, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line or line.startswith("!"):
                continue
            parts = line.rstrip("\n").split("\t")
            # Some files can have fewer trailing columns; pad
            if len(parts) < len(GAF_COLS):
                parts = parts + [""] * (len(GAF_COLS) - len(parts))
            yield parts[:len(GAF_COLS)]

def build_goa_subset(
    goa_gaf_gz: Path,
    uniprot_ids: set[str],
    out_csv: Path,
    keep_aspects: Optional[set[str]] = None,
) -> dict:
    """
    Subset GOA GAF to only the proteins of interest.

    Output schema (simple + stable):
    - uniprot_id
    - go_id
    - aspect
    - evidence_code
    - assigned_by
    """
    keep_aspects = keep_aspects or {"P"}  # P = Biological Process

    rows = []
    kept = 0
    scanned = 0

    for parts in _iter_gaf_rows(goa_gaf_gz):
        scanned += 1
        db = parts[0]
        obj_id = parts[1]
        go_id = parts[4]
        evidence = parts[6]
        aspect = parts[8]
        assigned_by = parts[14]

        # We only want UniProtKB rows
        if db not in {"UniProtKB", "UNIPROTKB", "uniprotkb"}:
            continue

        uid = _canon_uniprot(obj_id)
        if uid in uniprot_ids and aspect in keep_aspects and go_id.startswith("GO:"):
            kept += 1
            rows.append([uid, go_id, aspect, evidence, assigned_by])

    df = pd.DataFrame(rows, columns=["uniprot_id", "go_id", "aspect", "evidence_code", "assigned_by"])
    df = df.drop_duplicates()

    _write_df(df, out_csv)

    return {
        "scanned_lines": int(scanned),
        "kept_rows": int(df.shape[0]),
        "unique_uniprot": int(df["uniprot_id"].nunique()) if not df.empty else 0,
        "unique_go": int(df["go_id"].nunique()) if not df.empty else 0,
        "out": str(out_csv),
    }


# ----------------------------
# 3) FAERS Pralsetinib XLSX -> clean + reaction long/count tables
# ----------------------------

def clean_faers_xlsx(raw_xlsx: Path, out_reports_csv: Path, out_reactions_long_csv: Path, out_reaction_counts_csv: Path) -> dict:
    df = pd.read_excel(raw_xlsx)

    # Standardize column names
    df.columns = [_norm_col(c) for c in df.columns]

    # Canonical columns we’ll preserve if present
    # (we don’t enforce all; FAERS extracts vary)
    # Most important for long reactions:
    # - case_id
    # - suspect_product_active_ingredients
    # - reactions
    # - serious
    keep_cols = [c for c in [
        "case_id",
        "suspect_product_names",
        "suspect_product_active_ingredients",
        "reason_for_use",
        "reactions",
        "serious",
        "outcomes",
        "sex",
        "event_date",
        "latest_fda_received_date",
        "patient_age",
        "patient_weight",
        "reporter_type",
        "country_where_event_occurred",
        "initial_fda_received_date",
    ] if c in df.columns]

    df = df[keep_cols].copy()

    # Normalize a few common fields
    if "case_id" in df.columns:
        df["case_id"] = df["case_id"].map(_safe_strip)
    if "suspect_product_active_ingredients" in df.columns:
        df["suspect_product_active_ingredients"] = df["suspect_product_active_ingredients"].map(_safe_strip).str.lower()
    if "serious" in df.columns:
        df["serious"] = df["serious"].map(_safe_strip).str.lower()

    # Save cleaned report-level table
    _write_df(df, out_reports_csv)

    # Build reaction long table
    if "case_id" not in df.columns or "reactions" not in df.columns:
        # Still write empties so downstream doesn’t crash
        long_df = pd.DataFrame(columns=["case_id", "reaction"])
    else:
        rows = []
        for _, r in df[["case_id", "reactions"]].iterrows():
            case_id = _safe_strip(r["case_id"])
            for rxn in _split_multi(r["reactions"], sep=";"):
                # normalize spacing/case minimally
                rxn_clean = re.sub(r"\s+", " ", rxn).strip()
                rows.append([case_id, rxn_clean])
        long_df = pd.DataFrame(rows, columns=["case_id", "reaction"]).dropna()
        long_df = long_df[long_df["reaction"].astype(str).str.len() > 0]
        long_df = long_df.drop_duplicates()

    _write_df(long_df, out_reactions_long_csv)

    # Counts per reaction
    if long_df.empty:
        counts = pd.DataFrame(columns=["reaction", "case_count"])
    else:
        counts = (
            long_df.groupby("reaction")["case_id"]
            .nunique()
            .reset_index()
            .rename(columns={"case_id": "case_count"})
            .sort_values(["case_count", "reaction"], ascending=[False, True])
        )
    _write_df(counts, out_reaction_counts_csv)

    return {
        "report_rows": int(df.shape[0]),
        "unique_cases": int(df["case_id"].nunique()) if "case_id" in df.columns else None,
        "reaction_rows": int(long_df.shape[0]),
        "unique_reactions": int(long_df["reaction"].nunique()) if not long_df.empty else 0,
        "out_reports": str(out_reports_csv),
        "out_reactions_long": str(out_reactions_long_csv),
        "out_reaction_counts": str(out_reaction_counts_csv),
    }


# ----------------------------
# Optional: ChEMBL + OpenTargets cleaning (not required for Step 02 KG)
# ----------------------------

def clean_chembl_targets(raw_csv: Path, out_csv: Path) -> dict:
    df = _read_csv_auto(raw_csv)
    df.columns = [_norm_col(c) for c in df.columns]
    # keep a small useful subset if present
    keep = [c for c in ["chembl_id", "name", "accessions", "type", "organism", "tax_id"] if c in df.columns]
    if keep:
        df = df[keep].copy()
    _write_df(df, out_csv)
    return {"rows": int(df.shape[0]), "out": str(out_csv)}

def clean_opentargets_target_disease(raw_csv: Path, out_csv: Path) -> dict:
    df = pd.read_csv(raw_csv)
    df.columns = [_norm_col(c) for c in df.columns]
    # expected: target_symbol, disease_id, disease_name, score
    _write_df(df, out_csv)
    return {"rows": int(df.shape[0]), "out": str(out_csv)}


def clean_go_basic_obo(raw_obo: Path, out_csv: Path) -> dict:
    """
    Parse go-basic.obo into a compact CSV dictionary for stable downstream use.

    Output schema:
      go_id, go_name

    Notes:
    - Only reads [Term] blocks.
    - Ignores obsolete flags etc. (can be added later if needed).
    """
    rows = []
    go_id = None
    go_name = None
    in_term = False

    with raw_obo.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.rstrip("\n")

            if line == "[Term]":
                in_term = True
                go_id = None
                go_name = None
                continue

            if not in_term:
                continue

            if line.startswith("id: GO:"):
                go_id = line.split("id: ", 1)[1].strip()
            elif line.startswith("name: "):
                go_name = line.split("name: ", 1)[1].strip()
            elif line == "":
                if go_id and go_name:
                    rows.append({"go_id": go_id, "go_name": go_name})
                in_term = False
                go_id = None
                go_name = None

    # last block
    if in_term and go_id and go_name:
        rows.append({"go_id": go_id, "go_name": go_name})

    df = pd.DataFrame(rows).drop_duplicates(subset=["go_id"]).reset_index(drop=True)
    _write_df(df, out_csv)

    return {
        "rows": int(df.shape[0]),
        "unique_go": int(df["go_id"].nunique()) if not df.empty else 0,
        "out": str(out_csv),
    }



# ----------------------------
# Main
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", type=Path, default=Path("data/00_raw"), help="Path to raw inputs")
    ap.add_argument("--out-dir", type=Path, default=Path("data/01_clean"), help="Path to cleaned outputs")
    ap.add_argument("--drug", type=str, default="pralsetinib", help="Drug name used in filenames")
    ap.add_argument("--keep-aspects", type=str, default="P", help="GOA aspects to keep (default P=Biological Process). e.g. 'P,F,C'")
    args = ap.parse_args()

    raw = args.raw_dir
    out = args.out_dir
    _ensure_dir(out)

    summary = {
        "timestamp": dt.datetime.utcnow().isoformat() + "Z",
        "raw_dir": str(raw),
        "out_dir": str(out),
        "drug": args.drug,
        "steps": {},
    }

    # --- DrugBank seed proteins
    # Accept either your old name or the renamed canonical name
    drugbank_candidates = [
        raw / f"drugbank_{args.drug}_seed_proteins.csv",
        raw / f"drugbank_{args.drug}_seed_proteins_complete.csv",
        raw / "drugbank_pralsetinib_seed_proteins_complete.csv",
    ]
    drugbank_in = next((p for p in drugbank_candidates if p.exists()), None)
    if drugbank_in is None:
        raise FileNotFoundError(f"Could not find DrugBank seed proteins CSV in {raw}. "
                                f"Tried: {[str(p) for p in drugbank_candidates]}")
    drugbank_out = out / f"{args.drug}_seed_proteins.csv"
    summary["steps"]["drugbank_seed_proteins"] = clean_drugbank_seed_proteins(drugbank_in, drugbank_out)

    # --- GOA GAF subset
    goa_candidates = [
        raw / "goa_human.gaf.gz",
        raw / "goa_human_annotations.gaf.gz",
        raw / "goa_human.gaf",
        raw / "goa_human_annotations.gaf",
    ]
    goa_in = next((p for p in goa_candidates if p.exists()), None)
    if goa_in is None:
        raise FileNotFoundError(f"Could not find GOA GAF(.gz) in {raw}. Tried: {[str(p) for p in goa_candidates]}")

    uniprot_ids = set(pd.read_csv(drugbank_out)["uniprot_id"].astype(str))
    keep_aspects = set([a.strip() for a in args.keep_aspects.split(",") if a.strip()])
    goa_out = out / f"{args.drug}_targets_goa.csv"
    summary["steps"]["goa_subset"] = build_goa_subset(goa_in, uniprot_ids, goa_out, keep_aspects=keep_aspects)

    # --- FAERS XLSX
    faers_candidates = [
        raw / f"faers_{args.drug}_reports.xlsx",
        raw / "faers_pralsetinib_reports.xlsx",
    ]
    faers_in = next((p for p in faers_candidates if p.exists()), None)
    if faers_in is not None:
        reports_out = out / f"{args.drug}_faers_reports_clean.csv"
        rxn_long_out = out / f"{args.drug}_faers_reactions_long.csv"
        rxn_counts_out = out / f"{args.drug}_faers_reaction_counts.csv"
        summary["steps"]["faers_clean"] = clean_faers_xlsx(faers_in, reports_out, rxn_long_out, rxn_counts_out)
    else:
        summary["steps"]["faers_clean"] = {"skipped": True, "reason": "FAERS XLSX not found in raw-dir"}

    # --- Optional: ChEMBL + OpenTargets
    chembl_in = raw / f"chembl_{args.drug}_targets.csv"
    if chembl_in.exists():
        summary["steps"]["chembl_clean"] = clean_chembl_targets(chembl_in, out / f"{args.drug}_chembl_targets.csv")

    ot_in = raw / "opentargets_target_disease_long.csv"
    if ot_in.exists():
        summary["steps"]["opentargets_clean"] = clean_opentargets_target_disease(ot_in, out / "opentargets_target_disease_long.csv")

    # --- Optional: GO basic ontology dictionary (recommended)
    go_basic_in = raw / "go-basic.obo"
    if go_basic_in.exists():
        summary["steps"]["go_basic_clean"] = clean_go_basic_obo(go_basic_in, out / "go_basic_terms.csv")


    # Write a small manifest for traceability
    manifest_path = out / "_clean_manifest.json"
    _write_df(pd.DataFrame([{"file": k, **v} for k, v in summary["steps"].items()]), out / "_clean_summary.csv")
    _ensure_dir(manifest_path.parent)
    manifest_path.write_text(json.dumps(summary, indent=2))
    print(f"[ok] wrote cleaned outputs to: {out}")
    print(f"[ok] wrote manifest: {manifest_path}")

if __name__ == "__main__":
    main()
