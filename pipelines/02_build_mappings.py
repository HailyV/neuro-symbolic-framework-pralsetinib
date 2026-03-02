#!/usr/bin/env python3
"""
pipelines/02_build_mappings.py

STEP 02: Build mappings needed for KG construction (from CLEAN files only).

A) FAERS reaction -> toxicity_theme
   Input : data/01_clean/pralsetinib_faers_reactions_long.csv   (case_id,reaction)
   Output:
     - data/01_clean/faers_reaction_theme_long.csv              (case_id,reaction,toxicity_theme)
     - data/01_clean/faers_ontology_grouped.csv                 (ontology_group,count)

B) GO -> toxicity_theme using go-basic.obo keyword rules
   Inputs:
     - data/01_clean/pralsetinib_targets_goa.csv                (uniprot_id,go_id,aspect,...)
     - data/00_raw/go-basic.obo
   Outputs:
     - data/01_clean/go_basic_terms.csv                         (go_id,go_name,namespace)
     - data/01_clean/go_to_toxicity_theme_auto.csv              (go_id,toxicity_theme,go_name,namespace,rule)
     - data/01_clean/go_to_toxicity_theme_review.csv            (go_id,go_name,namespace,toxicity_theme,mapped)
     - data/01_clean/go_to_toxicity_theme.csv                   (FINAL for KG)

This file is safe to re-run; it overwrites outputs.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import re
import pandas as pd


# ----------------------------
# Helpers
# ----------------------------

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _s(x) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    return str(x).strip()

def norm_go_id(go: str) -> str:
    """
    Always return canonical GO ID as: GO:0006915
    Accepts: GO:..., go:GO:..., go:go:GO:..., whitespace, etc.
    Returns '' if cannot parse.
    """
    s = _s(go)
    if not s:
        return ""
    # strip any number of leading "go:" prefixes
    while s.lower().startswith("go:"):
        s = s[3:].strip()
    if s.startswith("GO:"):
        return s
    m = re.search(r"(GO:\d{7})", s)
    return m.group(1) if m else ""

def norm_theme(theme: str) -> str:
    return _s(theme)

def split_reactions(val) -> list[str]:
    """
    Your cleaned long file already split reactions into one per row,
    but keep this helper for robustness if someone feeds a semicolon list.
    """
    s = _s(val)
    if not s:
        return []
    parts = re.split(r"[;\n]+", s)
    return [p.strip() for p in parts if p.strip()]


# ----------------------------
# A) FAERS reaction -> theme
# ----------------------------

DEFAULT_RULES: list[tuple[str, str]] = [
    ("Severe Outcome", r"\b(death|fatal|hospitali[sz]ation|life[- ]threatening|icu|intensive care)\b"),
    ("Medication Error", r"\b(product prescribing issue|product dose omission issue|medication error|wrong dose|overdose)\b"),
    ("Immune System", r"\b(infection|sepsis|neutrop|leukopen|white blood cell|immune|inflamm)\b"),
    ("Neurological", r"\b(headache|dizziness|syncope|seizure|neurop|insomnia|anxiety)\b"),
]

COMPILED_RXN_RULES = [(theme, re.compile(pat, flags=re.IGNORECASE)) for theme, pat in DEFAULT_RULES]


def map_reaction_to_theme(rxn: str) -> str:
    s = _s(rxn)
    if not s:
        return "Other / Unmapped"
    for theme, pat in COMPILED_RXN_RULES:
        if pat.search(s):
            return theme
    return "Other / Unmapped"


def build_faers_mappings(faers_rxn_long_csv: Path, out_long: Path, out_grouped: Path) -> None:
    if not faers_rxn_long_csv.exists():
        raise FileNotFoundError(f"Missing {faers_rxn_long_csv} (run 01_clean_inputs.py first)")

    df = pd.read_csv(faers_rxn_long_csv)

    # Expect clean long schema: case_id,reaction
    if not {"case_id", "reaction"}.issubset(df.columns):
        raise ValueError(f"{faers_rxn_long_csv} must have columns case_id,reaction. Found: {list(df.columns)}")

    # Some pipelines might still have semicolon lists; normalize just in case
    rows = []
    for _, r in df.iterrows():
        cid = _s(r["case_id"])
        for rxn in split_reactions(r["reaction"]):
            rows.append((cid, rxn))
    long_df = pd.DataFrame(rows, columns=["case_id", "reaction"]).drop_duplicates()

    long_df["toxicity_theme"] = long_df["reaction"].map(map_reaction_to_theme)

    grouped = (
        long_df.groupby("toxicity_theme")["case_id"]
        .nunique()
        .reset_index()
        .rename(columns={"toxicity_theme": "ontology_group", "case_id": "count"})
        .sort_values("count", ascending=False)
    )

    _ensure_dir(out_long.parent)
    long_df.to_csv(out_long, index=False)
    grouped.to_csv(out_grouped, index=False)

    total = int(long_df["reaction"].nunique())
    unmapped = int((long_df["toxicity_theme"] == "Other / Unmapped").sum())
    print("\n[A] Reaction→Theme mapping complete (fallback substring rules)")
    print(f"✅ wrote: {out_long}")
    print(f"✅ wrote: {out_grouped}")
    print(grouped.to_string(index=False))

    # Show top unmapped reactions by frequency (case-count)
    if unmapped > 0:
        top_unmapped = (
            long_df[long_df["toxicity_theme"] == "Other / Unmapped"]
            .groupby("reaction")["case_id"]
            .nunique()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
            .head(25)
        )
        print(f"\nUnmapped reactions: {unmapped}/{len(long_df)} ({100*unmapped/len(long_df):.1f}%)")
        print("\nTop unmapped reactions to consider adding rules for:")
        print(top_unmapped.to_string(index=False))


# ----------------------------
# B) GO -> theme mapping via go-basic.obo
# ----------------------------

THEME_RULES: list[tuple[str, str]] = [
    ("Immune System",
     r"\b(immune|immun|cytokine|interleukin|chemokine|inflamm|leukocyte|lymphocyte|neutrophil|macrophage|"
     r"antigen|interferon|complement|defense response|response to virus|response to bacterium|innate immune|adaptive immune)\b"),
    ("Cell Death",
     r"\b(apoptosis|apoptotic|cell death|programmed cell death|necrosis|necroptosis|pyroptosis|caspase)\b"),
    ("Oxidative Stress",
     r"\b(oxidative|reactive oxygen|ros\b|hydrogen peroxide|superoxide|redox|glutathione|lipid peroxidation|oxidation-reduction)\b"),
    ("Cell Adhesion",
     r"\b(cell adhesion|integrin|extracellular matrix|cell-cell adhesion|focal adhesion|endotheli|vascul|thrombo|hemostasis|coagulation)\b"),
    ("Neurological",
     r"\b(neuron|neuronal|synapse|synaptic|axon|dendrite|brain|cns\b|central nervous|peripheral nervous|glial|neurotransmitter|dopamine|serotonin|gaba)\b"),
]
COMPILED_GO_RULES = [(theme, re.compile(pat, flags=re.IGNORECASE)) for theme, pat in THEME_RULES]

def parse_go_basic_obo(path: Path) -> pd.DataFrame:
    """
    Robust OBO parser for GO: extracts (go_id, go_name, namespace),
    ignoring obsolete terms.
    """
    rows = []
    go_id = go_name = namespace = None
    is_obsolete = False
    in_term = False

    def flush():
        nonlocal go_id, go_name, namespace, is_obsolete
        if in_term and go_id and go_name and namespace and not is_obsolete:
            rows.append({"go_id": go_id, "go_name": go_name, "namespace": namespace})

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.rstrip("\n")

            if line == "[Term]":
                # flush previous term before starting a new one
                flush()
                in_term = True
                go_id = go_name = namespace = None
                is_obsolete = False
                continue

            if not in_term:
                continue

            if line.startswith("id: GO:"):
                go_id = line.split("id: ", 1)[1].strip()
            elif line.startswith("name: "):
                go_name = line.split("name: ", 1)[1].strip()
            elif line.startswith("namespace: "):
                namespace = line.split("namespace: ", 1)[1].strip()
            elif line.startswith("is_obsolete: "):
                is_obsolete = (line.split("is_obsolete: ", 1)[1].strip().lower() == "true")

            # end-of-stanza marker: blank line
            if line == "":
                flush()
                in_term = False

    # flush last term if file ends without blank line
    flush()

    return pd.DataFrame(rows).drop_duplicates(subset=["go_id"]).reset_index(drop=True)


def theme_for_go_name(go_name: str) -> tuple[str | None, str | None]:
    s = _s(go_name)
    if not s:
        return None, None
    for theme, pat in COMPILED_GO_RULES:
        if pat.search(s):
            return theme, "go_name_keywords"
    return None, None


def build_go_mappings(goa_csv: Path, go_obo: Path, out_terms: Path, out_auto: Path, out_review: Path, out_final: Path) -> None:
    if not goa_csv.exists():
        raise FileNotFoundError(f"Missing {goa_csv} (run 01_clean_inputs.py first)")
    if not go_obo.exists():
        raise FileNotFoundError(f"Missing {go_obo} (put go-basic.obo under data/00_raw/)")

    goa = pd.read_csv(goa_csv)
    if "go_id" not in goa.columns:
        raise ValueError(f"{goa_csv} missing go_id column. Found: {list(goa.columns)}")

    goa["go_id"] = goa["go_id"].map(norm_go_id)
    go_ids_in_project = sorted(set(goa["go_id"].dropna().astype(str)) - {""})

    go_terms = parse_go_basic_obo(go_obo)
    _ensure_dir(out_terms.parent)
    go_terms.to_csv(out_terms, index=False)

    # subset GO terms to those present in your Pralsetinib GOA subset
    used = go_terms[go_terms["go_id"].isin(set(go_ids_in_project))].copy()

    # ALWAYS define these columns (prevents NameError)
    if used.empty:
        used["toxicity_theme"] = pd.Series(dtype="object")
        used["rule"] = pd.Series(dtype="object")
    else:
        # theme_for_go_name returns (theme or None, rule or None)
        tmp = used["go_name"].map(theme_for_go_name).apply(pd.Series)
        tmp.columns = ["toxicity_theme", "rule"]
        used[["toxicity_theme", "rule"]] = tmp

    mapped = used[used["toxicity_theme"].notna()].copy()

    # auto mapping output
    auto_out = mapped[["go_id", "toxicity_theme", "go_name", "namespace", "rule"]].sort_values(
        ["toxicity_theme", "go_id"]
    )
    auto_out.to_csv(out_auto, index=False)

    # review output
    review = used.copy()
    review["mapped"] = review["toxicity_theme"].notna().astype(int)
    review = review[["go_id", "go_name", "namespace", "toxicity_theme", "mapped"]].sort_values(
        ["mapped", "go_id"], ascending=[True, True]
    )
    review.to_csv(out_review, index=False)

    # final mapping used by KG
    final = auto_out[["go_id", "toxicity_theme"]].drop_duplicates()
    final.to_csv(out_final, index=False)

    print("\n[B] GO→Theme mapping complete (using go-basic.obo)")
    print(f"✅ wrote: {out_terms} (rows={len(go_terms)})")
    print(f"✅ wrote: {out_auto} (mapped_rows={len(auto_out)})")
    print(f"✅ wrote: {out_review} (for manual checking; total_go={len(review)})")
    print(f"✅ wrote: {out_final} (FINAL used by KG; rows={len(final)})")

    print("\n-- mapped counts by theme (auto) --")
    if len(auto_out) == 0:
        print("[warn] No GO terms matched your rules. Expand THEME_RULES patterns.")
    else:
        print(auto_out["toxicity_theme"].value_counts().to_string())

    unmapped = int((review["mapped"] == 0).sum())
    print(f"\nUnmapped GO terms (in GOA subset): {unmapped} / {len(review)}")


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean_dir", type=Path, default=Path("data/01_clean"))
    ap.add_argument("--raw_dir", type=Path, default=Path("data/00_raw"))
    args = ap.parse_args()

    clean = args.clean_dir
    raw = args.raw_dir

    # A) FAERS
    build_faers_mappings(
        faers_rxn_long_csv=clean / "pralsetinib_faers_reactions_long.csv",
        out_long=clean / "faers_reaction_theme_long.csv",
        out_grouped=clean / "faers_ontology_grouped.csv",
    )

    # B) GO
    build_go_mappings(
        goa_csv=clean / "pralsetinib_targets_goa.csv",
        go_obo=raw / "go-basic.obo",
        out_terms=clean / "go_basic_terms.csv",
        out_auto=clean / "go_to_toxicity_theme_auto.csv",
        out_review=clean / "go_to_toxicity_theme_review.csv",
        out_final=clean / "go_to_toxicity_theme.csv",
    )


if __name__ == "__main__":
    main()