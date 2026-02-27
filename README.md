# Neuro-Symbolic Framework for Off-Target Toxicity Analysis (Pralsetinib)

## Project Overview
This project explores whether **ontology-grounded, neuro-symbolic approaches** can improve the interpretability and reliability of off-target toxicity analysis in drug discovery. We focus on **Pralsetinib**, a recently approved RET kinase inhibitor with emerging post-marketing safety signals.

Rather than predicting toxicity purely from statistical correlations, our goal is to **mechanistically explain observed adverse events** by linking:
- drug–protein interactions,
- biological processes (ontologies),
- and real-world patient outcomes.

---

## Core Research Question
**Do ontologies help ground AI models for drug toxicity analysis by enabling biologically interpretable reasoning from molecular interactions to real-world adverse events?**

---

## Data Sources and Roles

### DrugBank
- Primary biological anchor
- Provides curated drug targets, metabolic enzymes, and transporters
- Used to define *which proteins matter* for Pralsetinib
- DrugBank toxicity annotations used only for **contextual validation**, not as prediction labels

### Gene Ontology (GO / GOA)
- Links proteins to biological processes
- Enables reasoning about *what happens biologically* when a protein is perturbed
- Used as the core ontology layer in the knowledge graph

### FAERS (openFDA)
- Provides real-world post-marketing adverse event signals
- Used as the **primary toxicity signal** and weak supervision target
- Toxicities are grouped into ontology-aligned themes (e.g. hepatic, immune, neurological)
- Final dataset: **200 adverse events, 2,196 reports**

---

## Knowledge Graph

The knowledge graph covers Pralsetinib's three primary targets — **RET, JAK2, and FLT3** — and contains:
- **414 nodes** across four types: Drug, Protein, GO Biological Process, Toxicity Theme
- **524 edges** across four relationship types

### Edge Types
- `binds_to` (Drug → Protein, DrugBank)
- `involved_in` (Protein → GO, GOA)
- `reported_with` (Drug → ToxicityTheme, FAERS)
- `maps_to` (GO → ToxicityTheme, manually curated)

### Reasoning Path
```
Drug → Protein → GO Process → Toxicity Theme
```

This structure enables mechanistic explanations such as:
> "Pralsetinib binds JAK2 → immune signaling pathways → increased infection risk"

---

## Repository Structure

```
data/
├── DrugBank/
│   └── drugbank_pralsetinib_seed_proteins_complete.csv
├── interim/
│   ├── pralsetinib_targets_goa.csv
│   ├── faers_ontology_grouped.csv
│   ├── kg_nodes_v2.csv
│   ├── kg_edges_v2.csv
│   └── go_to_toxicity_theme.csv
scripts/
├── DB_GO_FAERS_build_kg_csv.py
```

---

## Models and Results

### Model 1 — FAERS Baseline
Pure frequency ranking of toxicity themes with no biological reasoning.

**Findings:** "Other" (administrative noise) dominates at 32%, Gastrointestinal ranks 2nd. While Haematological is correctly ranked at #3, the model cannot explain *why*, and buries Immune/Infection at #7 and Renal at #14 despite both having strong mechanistic basis. This establishes the ceiling of frequency-only approaches.

---

### Model 2 — Bayesian KG Path Scoring (Ontology-Grounded)
Combines FAERS frequency with mechanistic path evidence through the knowledge graph. Paths are weighted by protein role (target > enzyme > transporter) and GO relevance.

**Findings:**
- **Haematological** rises to rank #1 (63 paths through JAK2 + FLT3 + RET)
- **Immune/Infection** jumps from #7 → #3, driven by JAK2's cytokine signaling role
- **Renal** rises from #14 → #8, grounded in RET kidney biology
- **Cell Death/Apoptosis** surfaces at rank 12 with only 7 FAERS reports but strong JAK2 apoptosis pathway evidence — completely invisible to frequency analysis

---

### Model 3 — Complementarity Analysis
Tests whether KG path counts and FAERS report counts are measuring the same thing.

**Headline finding:** KG path count and FAERS report count are nearly uncorrelated across themes (**Spearman r ≈ 0.18**), confirming they capture fundamentally different signals — mechanistic potential vs. observed reporting. This justifies the need for both.

**2×2 Quadrant Framework:**

| Quadrant | Themes | Interpretation |
|---|---|---|
| **High KG + High FAERS** | Haematological, Immune/Infection, Neurological, Cardiovascular | Validated mechanistic toxicity — biology and data agree |
| **High KG + Low FAERS** | Cell Death/Apoptosis, Renal, Proliferative | Underreported novel signals — **prospective monitoring warranted** |
| **Low KG + High FAERS** | Pulmonary, Musculoskeletal, Hepatic | Class effects or indirect mechanisms not captured in this KG |
| **Low KG + Low FAERS** | — | Low mechanistic and statistical support |

The **High KG + Low FAERS** quadrant is the most actionable output: these are toxicity themes with strong biological rationale but sparse reporting, suggesting underreporting rather than absence of risk.

---

## Key Conclusions

- Ontologies **do** ground AI models for drug toxicity analysis — the knowledge graph surfaces mechanistically plausible signals that frequency-only approaches miss entirely.
- The low Spearman correlation between KG and FAERS rankings (r ≈ 0.18) is a feature, not a bug: it demonstrates that mechanistic and epidemiological evidence are complementary, not redundant.
- Cell Death/Apoptosis and Renal toxicity are the most notable novel findings — biologically supported but statistically underreported — and represent candidates for prospective monitoring in Pralsetinib safety surveillance.
- The framework produces **explainable mechanistic hypotheses**, not black-box predictions, connecting off-target protein interactions to observed adverse events.

---

## Limitations
- Focused on one drug (Pralsetinib); generalizability untested
- GO → toxicity theme mapping includes manual curation
- FAERS data subject to reporting bias
- KG operates at the **theme level**; individual AE rankings within a theme are not differentiated by the model

---

## Authors
DSC 180B Capstone Project Team  
University of California, San Diego
