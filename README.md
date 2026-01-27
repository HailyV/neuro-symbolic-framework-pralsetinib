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
- Provides curated:
  - drug targets
  - metabolic enzymes
  - transporters
- Used to define *which proteins matter* for Pralsetinib
- DrugBank toxicity annotations are used only for **contextual validation**, not as prediction labels

### Gene Ontology (GO / GOA)
- Links proteins to biological processes
- Enables reasoning about *what happens biologically* when a protein is perturbed
- Used as the core ontology layer in the knowledge graph

### FAERS (openFDA)
- Provides real-world post-marketing adverse event signals
- Used as the **primary toxicity signal** and weak supervision target
- Toxicities are grouped into ontology-aligned themes (e.g. hepatic, immune, neurological)

---

## Knowledge Graph Design

### Node Types
- Drug
- Protein (targets, enzymes, transporters)
- GO Biological Process
- Toxicity Theme

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
> “Pralsetinib binds JAK2 → immune signaling pathways → increased infection risk”

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

## Current Pipeline Status
- ✅ DrugBank protein extraction and curation
- ✅ GO annotation integration (Protein → GO)
- ✅ FAERS toxicity theme grouping
- ✅ Knowledge Graph construction (CSV-based)
- ⏳ GO → ToxicityTheme mapping (in progress)
- ⏳ Path-based scoring and modeling

---

## Modeling Plan

### Baseline 1: FAERS-only
- Rank toxicity themes by FAERS frequency
- No biological reasoning

### Baseline 2: KG Path Scoring (Ontology-Grounded)
- Score Drug → Protein → GO → Toxicity paths
- Weight by:
  - protein role (target > enzyme > transporter)
  - GO relevance
  - FAERS support

### Optional ML Extension
- Convert KG paths into feature vectors
- Train simple models (logistic regression / random forest)
- Compare FAERS-only vs ontology-informed features
- Emphasis on **interpretability**, not black-box prediction

---

## Key Design Choices
- Single-drug deep analysis rather than multi-drug shallow modeling
- Mechanism-level learning instead of population-level prediction
- Ontologies used to structure features, not just annotate outputs

---

## Limitations
- Focused on one drug (Pralsetinib)
- GO → toxicity mapping includes manual curation
- FAERS data subject to reporting bias

---

## Outcome
The final output is not just a toxicity prediction, but an **explainable mechanistic hypothesis** connecting off-target protein interactions to observed adverse events, demonstrating how ontologies can ground AI-driven drug safety analysis.

---

## Authors
DSC180 Capstone Project Team  
University of California, San Diego
