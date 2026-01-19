# Neuro-Symbolic AI for Off-Target Effect Analysis
**Case Study: Pralsetinib**

## Project Overview
This project explores whether **ontology-grounded, neuro-symbolic structure** can improve the interpretability and biological plausibility of drug off-target effect analysis compared to using real-world adverse event data alone.

We focus on **Pralsetinib**, a recently approved RET kinase inhibitor with limited long-term safety data and emerging unexpected adverse event signals in post-marketing reports. By integrating **ChEMBL**, **Open Targets**, and **FAERS**, we construct a lightweight **knowledge graph (KG)** that connects drug–target interactions, disease biology, and real-world adverse events.

The goal is not to make definitive clinical claims, but to generate **mechanistically plausible hypotheses** and demonstrate how ontologies help ground AI-driven analysis.

---

## Data Sources

### 1. ChEMBL (Drug–Target Binding Evidence)
- Identifies **human protein targets** of Pralsetinib
- Cleaned to retain only:
  - SINGLE PROTEIN
  - Homo sapiens
  - Valid UniProt accessions
- Final targets:
  - RET (P07949)
  - FLT3 (P36888)
  - JAK2 (O60674)

File:
- `data/chembl_pralsetinib_targets_clean.csv`

---

### 2. Open Targets (Biological & Disease Context)
- Expands protein targets into **disease and phenotype associations**
- Queried using **Ensembl Gene IDs** via the Open Targets GraphQL API
- Associations are ontology-grounded (EFO / MONDO) and include confidence scores

File:
- `data/open_targets_target_disease_long.csv`

---

### 3. FAERS / openFDA (Real-World Adverse Events)
- Captures **post-marketing adverse event signals**
- Used as an outcome and validation layer, not mechanistic ground truth
- Adverse events are preserved as raw reported terms

File:
- `data/faers_data.xlsx`

---

## Project Structure

```
project/
├── data/
│   ├── chembl_pralsetinib_targets_clean.csv
│   ├── chembl_raw_pralsetinib_targets.csv
│   ├── open_targets_target_disease_long.csv
│   └── faers_data.xlsx
├── initial_exploration/
├── build_kg_csv.py
└── README.md
```

---

## Knowledge Graph Construction

### Node Types
- Drug: Pralsetinib
- Target: RET, FLT3, JAK2
- Disease / Phenotype: Open Targets associations
- AdverseEvent: FAERS reaction terms

### Edge Types
- Drug → Target (`binds_to`, ChEMBL)
- Target → Disease (`associated_with`, Open Targets, with score)
- Drug → AdverseEvent (`reported_with`, FAERS)

This structure enables multi-hop reasoning such as:
```
Pralsetinib → JAK2 → immune disease → FAERS infection signals
```

---

## Build the Knowledge Graph

Run:
```bash
python build_kg_csv.py
```

Outputs:
- `kg_nodes.csv`
- `kg_edges.csv`

---

## Design Rationale
FAERS provides real-world signals but no mechanism.  
ChEMBL provides binding evidence but no clinical context.  
Open Targets bridges molecular biology and disease relevance.

Keeping datasets linked rather than merged preserves provenance and enables explainable reasoning.

---

## Current Status
- Feasibility validated
- Targets cleaned and standardized
- Open Targets integration completed
- FAERS linked as validation layer
- Knowledge graph constructed (CSV format)

---

## Disclaimer
This project is for research and educational purposes only.
It does not provide clinical or medical advice.
