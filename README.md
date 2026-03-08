# Neuro-Symbolic Framework for Off-Target Toxicity Analysis (Pralsetinib)

## Project Overview
This project explores whether **ontology-grounded, neuro-symbolic approaches** can improve the interpretability and reliability of off-target toxicity analysis in drug discovery. We focus on **Pralsetinib**, a RET kinase inhibitor with post-marketing adverse event signals that are difficult to interpret using frequency counts alone.

Rather than predicting toxicity purely from statistical correlations, our goal is to **mechanistically explain observed adverse events** by linking:
- drug–protein interactions,
- biological processes (ontologies),
- and real-world patient outcomes.

By integrating biomedical ontologies with real-world safety data, we construct a **knowledge graph–based reasoning framework** that connects molecular mechanisms to observed toxicity themes.

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

### Node Types
- Drug
- Preotein
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
> "Pralsetinib binds JAK2 → immune signaling pathways → increased infection risk"

---

## Pipeline Overview

```
Raw Biomedical Data
   │
   ├── DrugBank / ChEMBL (drug–protein interactions)
   ├── Gene Ontology (protein → biological process)
   └── FAERS (adverse event reports)
   │
   ▼
Knowledge Graph Construction
(knowledge_graph/01_kg_generator)
   │
   ▼
Mechanistic Knowledge Graph
(nodes: drug, proteins, GO processes, toxicity themes)
   │
   ▼
Modeling Pipeline
   │
   ├── Model 1: FAERS Frequency Baseline
   ├── Model 2: Bayesian Knowledge Graph Scoring
   └── Model 3: Complementarity / Hybrid Analysis
   │
   ▼
Results & Interpretation
   │
   ├── toxicity theme rankings
   ├── KG vs FAERS comparison
   ├── mechanistic enrichment validation
   └── identification of underreported mechanistic signals
```

This pipeline enables reasoning from **molecular interactions → biological pathways → observed clinical adverse events.**

---

# Modeling Methodology


### Model 1 — FAERS Baseline

The first model establishes a **frequency-only baseline** by ranking toxicity themes according to their total FAERS report counts.

Pure frequency ranking of toxicity themes with no biological reasoning.

### Findings

The FAERS baseline ranking shows:

- “Other” administrative events dominate (~32%)
- Gastrointestinal toxicity ranks 2nd
- Haematological toxicity ranks 3rd

However, the model provides **no biological explanation** for these rankings and underestimates mechanistically grounded toxicity themes such as immune/infection and renal effects.

---

### Model 2 — Bayesian KG Path Scoring (Ontology-Grounded)
The second model combines FAERS frequency with **mechanistic pathway evidence** using a Bayesian scoring framework.
Posterior toxicity scores are computed as:

Posterior ∝ Prior(FAERS frequency) × Likelihood(KG pathway evidence)

Knowledge graph evidence is derived from **multi-hop mechanistic paths** connecting the drug to toxicity themes through biological processes.

**Findings:**
- **Haematological toxicity** rises to Rank #1 supported by strong mechanistic connectivity
- **Immune/Infection toxicity** rises from lower FAERS rank due to JAK2 cytokine signaling pathways
- **Renal toxicity** gains prominence through RET kidney development pathways
- **Cell Death / Apoptosis** emerges despite low reporting frequency due to strong pathway support

---

### Model 3 — Complementarity Analysis
The third model evaluates whether **knowledge graph evidence and FAERS reporting frequency capture the same signal**.


Spearman rank correlation between:

- KG mechanistic path counts
- FAERS report counts

Result:

Spearman r ≈ 0.18

This near-zero correlation indicates that mechanistic pathway evidence and reporting frequency capture **largely independent signals of toxicity risk**.

---

# Mechanistic Enrichment Validation

To independently validate biological relevance, a **Gene Ontology enrichment analysis** was performed using Fisher’s Exact Test.

Significant enrichment was observed for several mechanisms:

| Mechanism | Odds Ratio | FDR |
|---|---|---|
Cell Death | 9.1 | 0.036 |
Cell Adhesion | 13 | 0.010 |

These results support the mechanistic pathways identified by the knowledge graph model.

**Interpretation Framework:**

To interpret the relationship between mechanistic evidence and reporting frequency, toxicity themes can be grouped into four categories based on knowledge graph (KG) support and FAERS reporting frequency.


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
- Study focuses on a **single drug (Pralsetinib)**
- GO → toxicity theme mapping involves **manual curation**
- FAERS data is subject to **reporting bias**
- Analysis operates at the **toxicity theme level**, not individual adverse events

---

## Reproducibility

The analysis can be reproduced using the scripts provided in the repository.

## 0. Data Availability

This project integrates data from DrugBank, Gene Ontology / GOA, FAERS (openFDA), and supporting curated intermediate files.

- Most Processed & model ready files are included in the repository under `data/` and `knowledge_graph/`.
- Some raw data sources require access or download if hoping to retreive on your own.
  - DrugBank may require an academic license
  - FAERS / openFDA data can be retrieved from public FDA sources
  - GO / GOA files can be downloaded from Gene Ontology resources

If reproducing the full pipeline from raw data, place downloaded source files in:

`data/00_raw/ `

## 1. Install dependencies

`pip install -r requirements.txt`

Dependencies include:
- pandas
- numpy
- scipy
- scikit-learn
- networkx
- pyvis
- openpyxl


## 2. Build the Knowledge Graph

`cd knowledge_graph/01_kg_generator
python build_kg_csv.py`

Output files will be written to:

`knowledge_graph/00_kg_data/`

These include:
```
kg_nodes.csv
kg_edges.csv
```

## 3. Run the Models

### Model 1 — FAERS Baseline
```
cd models/01_base_model
python model1_baseline.py
```
Output:

`models/04_model_summaries/results_model1_baseline.csv`


### Model 2 — Bayesian Knowledge Graph Scoring
```
cd models/02_secondary_model
python model2_bayes_paths.py
```
Output:

`models/02_secondary_model/results_model2_bayes.csv`


### Model 3 — Hybrid / Complementarity Analysis
```
cd models/03_final_model
python model3_ml_hybrid.py
python loocv_logistic_regression.py
```
Outputs include:
```
models/04_model_summaries/results_model3_theme_analysis.csv
models/04_model_summaries/results_model3_novel_candidates.csv
```

## 4. Generate Summary Figures
```
cd models/05_summaries_generator
python plot_three_models.py
python error_analysis.py
```
Generated figures will appear in:

`figures/`

## Authors
DSC 180B Capstone Project Team
University of California, San Diego

Hannah Lee  
Haily Vuong  
Stephanie Yue  
Zoey He

## Mentors  
Murali Krishnam  
Raju Pusapati  
Justin Eldridge
