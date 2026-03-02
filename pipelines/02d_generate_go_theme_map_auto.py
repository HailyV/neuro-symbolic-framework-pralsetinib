#!/usr/bin/env python3
"""
pipelines/02d_generate_go_theme_map_auto.py

Auto-generate GO_ID -> toxicity_theme mapping using GO term names from go-basic.obo
and transparent keyword rules.

Inputs:
  - data/02_kg/nodes.csv
  - data/00_raw/go-basic.obo   (YOU download once and store locally)

Outputs:
  - data/01_clean/go_to_toxicity_theme_auto.csv
  - data/01_clean/go_to_toxicity_theme_auto_review.csv

Notes:
- Keeps mappings only for GO IDs present in YOUR KG (Pralsetinib-specific).
- Uses GO term NAME for mapping (simple + interpretable).
"""

from __future__ import annotations

from pathlib import Path
import re
import pandas as pd

NODES = Path("data/02_kg/nodes.csv")
GO_OBO = Path("data/00_raw/go-basic.obo")

OUT = Path("data/01_clean/go_to_toxicity_theme_auto.csv")
OUT_REVIEW = Path("data/01_clean/go_to_toxicity_theme_auto_review.csv")

THEME_RULES: list[tuple[str, str]] = [
    ("Immune System",
     r"\b(immune|immun|cytokine|interleukin|chemokine|inflamm|inflammatory|"
     r"leukocyte|lymphocyte|neutrophil|macrophage|t cell|b cell|"
     r"antigen|interferon|complement|defense response|response to virus|"
     r"response to bacterium|innate immune|adaptive immune)\b"),

    ("Cell Death",
     r"\b(apoptosis|apoptotic|cell death|programmed cell death|necrosis|"
     r"necroptosis|pyroptosis|autophagy|cell killing|caspase)\b"),

    ("Oxidative Stress",
     r"\b(oxidative|reactive oxygen|ros\b|hydrogen peroxide|superoxide|"
     r"redox|glutathione|lipid peroxidation|oxidation-reduction)\b"),

    ("Cell Adhesion",
     r"\b(cell adhesion|adhesion|integrin|extracellular matrix|"
     r"cell-cell adhesion|focal adhesion|platelet activation|coagulation|"
     r"hemostasis|thrombo|endotheli|vascul)\b"),

    ("Neurological",
     r"\b(neuron|neuronal|synapse|synaptic|axon|dendrite|neuro|"
     r"brain|cns\b|central nervous|peripheral nervous|glial|"
     r"neurotransmitter|dopamine|serotonin|gaba)\b"),
]

COMPILED_RULES = [(theme, re.compile(pat, flags=re.IGNORECASE)) for theme, pat in THEME_RULES]


def parse_go_basic_obo(path: Path) -> pd.DataFrame:
    """
    Minimal OBO parser:
    Extract GO ID and name from [Term] blocks.

    Returns df: go_id (GO:xxxxxxx), go_name
    """
    go_id = None
    go_name = None
    rows = []

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.rstrip("\n")
            if line == "[Term]":
                go_id = None
                go_name = None
                continue

            if line.startswith("id: GO:"):
                go_id = line.split("id: ", 1)[1].strip()
            elif line.startswith("name: "):
                go_name = line.split("name: ", 1)[1].strip()

            # end of block
            if line == "" and go_id and go_name:
                rows.append({"go_id": go_id, "go_name": go_name})
                go_id = None
                go_name = None

    if go_id and go_name:
        rows.append({"go_id": go_id, "go_name": go_name})

    return pd.DataFrame(rows).drop_duplicates(subset=["go_id"]).reset_index(drop=True)


def theme_for_name(go_name: str) -> str | None:
    for theme, pat in COMPILED_RULES:
        if pat.search(go_name):
            return theme
    return None


def main() -> None:
    if not NODES.exists():
        raise FileNotFoundError(f"Missing {NODES}. Build KG nodes first.")

    if not GO_OBO.exists():
        raise FileNotFoundError(
            f"Missing {GO_OBO}.\n"
            "Download go-basic.obo manually and place it at data/00_raw/go-basic.obo\n"
            "Tip: curl -L -o data/00_raw/go-basic.obo https://geneontology.org/ontology/go-basic.obo"
        )

    nodes = pd.read_csv(NODES)
    go_nodes = nodes[nodes["node_type"] == "GO_Process"]["node_id"].astype(str).tolist()

    # node_id looks like "go:GO:0006915"
    go_ids_in_kg = sorted({x.replace("go:", "") for x in go_nodes if x.startswith("go:GO:")})
    go_ids_in_kg_set = set(go_ids_in_kg)

    go = parse_go_basic_obo(GO_OBO)
    go = go[go["go_id"].isin(go_ids_in_kg_set)].copy()

    go["toxicity_theme"] = go["go_name"].map(theme_for_name)
    mapped = go[go["toxicity_theme"].notna()].copy()

    OUT.parent.mkdir(parents=True, exist_ok=True)

    mapped_out = mapped[["go_id", "toxicity_theme", "go_name"]].sort_values(["toxicity_theme", "go_id"])
    mapped_out.to_csv(OUT, index=False)

    review = go.copy()
    review["mapped"] = review["toxicity_theme"].notna().astype(int)
    review = review.sort_values(["mapped", "go_id"], ascending=[True, True])
    review.to_csv(OUT_REVIEW, index=False)

    print(f"✅ wrote: {OUT} (mapped_rows={len(mapped_out)})")
    print(f"✅ wrote: {OUT_REVIEW} (for manual checking)")

    print("\n-- mapped counts by theme --")
    if len(mapped_out) == 0:
        print("[warn] No GO names matched your rules. Expand THEME_RULES patterns.")
    else:
        print(mapped_out["toxicity_theme"].value_counts().to_string())

    unmapped = int((review["mapped"] == 0).sum())
    print(f"\nUnmapped GO terms (in KG): {unmapped} / {len(review)}")


if __name__ == "__main__":
    main()