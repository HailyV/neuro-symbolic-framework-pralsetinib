import os
import gzip
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict, deque

import scipy.stats as stats
from statsmodels.stats.multitest import multipletests

# input files
PATH_DRUGBANK_CLEAN = "../../data/01_clean/drugbank_pralsetinib_proteins_cleaned.csv"
PATH_GO_THEME_MAP   = "../../data/01_clean/go_to_toxicity_theme.csv"
PATH_GOA_GAF_GZ     = "../../data/00_raw/goa_human_annotations.gaf.gz"
PATH_GO_OBO         = "../../data/00_raw/go-basic.obo"

# output files
OUTFIG = "../../figures/mechanistic_enrichment_go_themes.png"

# ---- Parsing GO ontology (OBO) for descendants ----

def parse_go_obo(obo_path: str):
    """
    Parse go-basic.obo to build parent/child relationships via 'is_a'.
    Returns:
        parents: dict(child -> set(parents))
        children: dict(parent -> set(children))
    """
    parents = defaultdict(set)
    children = defaultdict(set)

    term_id = None
    is_obsolete = False

    with open(obo_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()

            if line == "[Term]":
                term_id = None
                is_obsolete = False
                continue

            if line.startswith("id: GO:"):
                term_id = line.split("id:")[1].strip()
                continue

            if line.startswith("is_obsolete:"):
                is_obsolete = (line.split("is_obsolete:")[1].strip().lower() == "true")
                continue

            if term_id and (not is_obsolete) and line.startswith("is_a: GO:"):
                parent = line.split("is_a:")[1].split("!")[0].strip()
                parents[term_id].add(parent)
                children[parent].add(term_id)

    return parents, children


def get_descendants(go_id: str, children_map: dict) -> set[str]:
    """
    Return all descendants of go_id (downward traversal).
    """
    seen = set()
    q = deque([go_id])
    while q:
        cur = q.popleft()
        for ch in children_map.get(cur, ()):
            if ch not in seen:
                seen.add(ch)
                q.append(ch)
    return seen


parents, children = parse_go_obo(PATH_GO_OBO)
print(f"OBO sucessfully parsed: terms_with_parents={len(parents):,}  parents_with_children={len(children):,}")

# ---- Parse GOA (GAF) to build protein → GO terms universe ----

def parse_goa_gaf_gz(gaf_gz_path: str, aspect_filter={"P"}, taxon_filter="taxon:9606"):
    """
    Parse GOA human annotations (GAF) into:
        prot_to_terms: dict(UniProt accession -> set(GO IDs))
        universe: set(UniProt accessions)
    Filters:
        - aspect 'P' (biological_process) by default
        - human taxon:9606
    """
    prot_to_terms = defaultdict(set)
    universe = set()

    with gzip.open(gaf_gz_path, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("!"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 15:
                continue

            uniprot_id = parts[1]     # DB Object ID (UniProt accession)
            go_id      = parts[4]     # GO:xxxxxxx
            aspect     = parts[8]     # P/F/C
            taxon      = parts[12]    # taxon:9606 etc.

            if aspect_filter and aspect not in aspect_filter:
                continue
            if taxon_filter and taxon_filter not in taxon:
                continue

            universe.add(uniprot_id)
            prot_to_terms[uniprot_id].add(go_id)

    return prot_to_terms, universe


prot_to_terms, universe = parse_goa_gaf_gz(PATH_GOA_GAF_GZ)
print(f"GOA sucessfully parsed: universe_size={len(universe):,} proteins (BP + human)")

# ---- Build target set from DrugBank (roles/actions) + build theme termsets (with descendants) ----

# DrugBank targets (cleaned)
db = pd.read_csv(PATH_DRUGBANK_CLEAN)

targets = set(
    db.loc[db["role"].str.lower() == "target", "uniprot_id"]
      .dropna()
      .astype(str)
)

targets = targets & universe  # keep only proteins in the GOA universe
print(f"Targets successfully loaded: n_targets={len(targets)}")

# GO → toxicity theme mapping 
theme_df = pd.read_csv(PATH_GO_THEME_MAP)

def normalize_go_id(x: str) -> str:
    x = str(x).strip()
    if x.lower().startswith("go:"):
        x = x.replace("go:", "")
    return x.upper()

theme_df["go_id_norm"] = theme_df["go_id"].map(normalize_go_id)

# Build: theme -> expanded GO termset (seed + descendants)
theme_termsets = {}
for _, row in theme_df.iterrows():
    theme_name = row["toxicity_theme"]
    seed = row["go_id_norm"]
    expanded = set([seed]) | get_descendants(seed, children)
    theme_termsets[theme_name] = expanded

for k, v in theme_termsets.items():
    print(f"{k}: seed+descendants = {len(v)} GO terms")

# ---- Enrichment test (Fisher exact + BH FDR) ----

def fisher_enrichment(target_set: set, hit_set: set, universe_set: set):
    """
    Fisher's exact test (one-sided, enrichment):
    table = [[a, b],
             [c, d]]
    a = targets that hit
    b = targets that don't hit
    c = non-targets that hit
    d = non-targets that don't hit
    """
    n = len(target_set)
    N = len(universe_set)

    a = len(target_set & hit_set)
    b = n - a
    c = len(hit_set - target_set)
    d = N - (a + b + c)

    table = np.array([[a, b], [c, d]])
    odds, p = stats.fisher_exact(table, alternative="greater")
    return a, n, len(hit_set), N, odds, p


rows = []
theme_hits = {}

for theme_name, termset in theme_termsets.items():
    # proteins that have ANY GO term in this expanded theme set
    hit_set = {prot for prot, terms in prot_to_terms.items() if terms & termset}
    theme_hits[theme_name] = hit_set

    k, n, K, N, odds, p = fisher_enrichment(targets, hit_set, universe)
    rows.append([theme_name, k, n, K, N, odds, p])

res = pd.DataFrame(rows, columns=[
    "theme", "k", "n_targets", "K", "N_universe", "odds_ratio", "p_value"
]).sort_values("p_value")

res["fdr_bh"] = multipletests(res["p_value"].values, method="fdr_bh")[1]
res

# ---- Make the presentation figure (bar chart of −log10(FDR)) + save PNG ----

def plot_mechanistic_enrichment(df: pd.DataFrame, outpath: str):
    df = df.copy()
    df["neglog10_fdr"] = -np.log10(df["fdr_bh"].clip(lower=1e-300))
    df = df.sort_values("neglog10_fdr", ascending=True)  # best at top in barh

    fig, ax = plt.subplots(figsize=(8.2, 4.9))
    ax.barh(df["theme"], df["neglog10_fdr"])

    # FDR=0.05 threshold line
    ax.axvline(-np.log10(0.05), linestyle="--")

    ax.set_xlabel("-log10(FDR)")
    ax.set_title("Pralsetinib target mechanistic enrichment (GO theme descendants)")

    # Annotate each bar
    for i, row in enumerate(df.itertuples(index=False)):
        ax.text(
            row.neglog10_fdr + 0.05,
            i,
            f"k={row.k}/{row.n_targets}, OR={row.odds_ratio:.2g}, FDR={row.fdr_bh:.3g}",
            va="center",
            fontsize=9
        )

    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    return fig


fig = plot_mechanistic_enrichment(res, OUTFIG)
OUTCSV = "../04_model_summaries/mechanistic_enrichment_results.csv"
res.to_csv(OUTCSV, index=False)

# Show which targets contribute to each theme

for theme, hitset in theme_hits.items():
    contributing_targets = targets & hitset
    
    print("\n", theme)
    print("targets contributing:", contributing_targets)