"""
kg_shared.py
============
Shared constants, mappings, and KnowledgeGraph loader for the
Neuro-Symbolic Toxicity Framework (Pralsetinib, DSC 180B).

Imported by:
  model1_baseline.py        — FAERS-only frequency ranking
  model2_bayes_paths.py     — Ontology-grounded Bayesian path scoring
  model3_ml_hybrid.py       — Logistic regression on KG-derived features

Why a shared module?
  The GO→AE-theme mapping and AE→theme taxonomy are the same across all three
  models. Keeping them in one place means a single edit propagates everywhere
  and makes it clear these definitions are fixed experimental constants, not
  per-model tuning choices.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# FILE PATHS  (all relative to models/ folder — edit here if layout changes)
# ─────────────────────────────────────────────────────────────────────────────
_HERE           = Path(__file__).parent
DATA_DIR        = _HERE.parent / "analysis" / "kg_files"

NODES_FILE      = DATA_DIR / "kg_nodes.csv"
EDGES_FILE      = DATA_DIR / "kg_edges_with_maps_to.csv"   # full KG
FAERS_ONLY_FILE = DATA_DIR / "kg_edges_stripped.csv"       # reported_with edges only

DRUG_ID = "drug:CHEMBL4582651"   # Pralsetinib


# ─────────────────────────────────────────────────────────────────────────────
# AE THEME TAXONOMY
# 14 clinically meaningful groups that map the 200 FAERS AEs for Pralsetinib.
# "Other" catches anything not explicitly assigned.
# ─────────────────────────────────────────────────────────────────────────────
AE_THEME_MAP: dict[str, str] = {
    # Haematological
    "Anaemia": "Haematological",
    "Neutropenia": "Haematological",
    "Thrombocytopenia": "Haematological",
    "Leukopenia": "Haematological",
    "Lymphopenia": "Haematological",
    "Myelosuppression": "Haematological",
    "Pancytopenia": "Haematological",
    "Febrile Neutropenia": "Haematological",
    "White Blood Cell Count Decreased": "Haematological",
    "Platelet Count Decreased": "Haematological",
    "Platelet Count Increased": "Haematological",
    "Haemoglobin Decreased": "Haematological",
    "Red Blood Cell Count Decreased": "Haematological",
    "Neutrophil Count Decreased": "Haematological",
    "Full Blood Count Abnormal": "Haematological",
    "Blood Test Abnormal": "Haematological",
    # Hepatic
    "Hepatic Function Abnormal": "Hepatic",
    "Hepatic Enzyme Increased": "Hepatic",
    "Transaminases Increased": "Hepatic",
    "Alanine Aminotransferase Increased": "Hepatic",
    "Aspartate Aminotransferase Increased": "Hepatic",
    "Liver Function Test Increased": "Hepatic",
    "Liver Injury": "Hepatic",
    "Liver Disorder": "Hepatic",
    "Blood Alkaline Phosphatase Increased": "Hepatic",
    # Immune/Infection
    "Pneumonia": "Immune/Infection",
    "Infection": "Immune/Infection",
    "Sepsis": "Immune/Infection",
    "Urinary Tract Infection": "Immune/Infection",
    "Covid-19": "Immune/Infection",
    "Herpes Zoster": "Immune/Infection",
    "Nasopharyngitis": "Immune/Infection",
    "Cellulitis": "Immune/Infection",
    "Influenza": "Immune/Infection",
    "Sinusitis": "Immune/Infection",
    "Immunodeficiency": "Immune/Infection",
    "Bronchopulmonary Aspergillosis": "Immune/Infection",
    "Fungal Foot Infection": "Immune/Infection",
    "Pneumonia Fungal": "Immune/Infection",
    "Pneumonia Viral": "Immune/Infection",
    "Pneumocystis Jirovecii Pneumonia": "Immune/Infection",
    # Cardiovascular
    "Hypertension": "Cardiovascular",
    "Blood Pressure Increased": "Cardiovascular",
    "Blood Pressure Abnormal": "Cardiovascular",
    "Labile Blood Pressure": "Cardiovascular",
    "Hypotension": "Cardiovascular",
    "Thrombosis": "Cardiovascular",
    "Pulmonary Embolism": "Cardiovascular",
    "Myocardial Infarction": "Cardiovascular",
    "Myocardial Necrosis Marker Increased": "Cardiovascular",
    "Heart Rate Increased": "Cardiovascular",
    # Pulmonary
    "Dyspnoea": "Pulmonary",
    "Pneumonitis": "Pulmonary",
    "Interstitial Lung Disease": "Pulmonary",
    "Cough": "Pulmonary",
    "Pleural Effusion": "Pulmonary",
    "Pulmonary Oedema": "Pulmonary",
    "Lung Disorder": "Pulmonary",
    "Hypoxia": "Pulmonary",
    "Respiratory Failure": "Pulmonary",
    "Pneumothorax": "Pulmonary",
    "Tachypnoea": "Pulmonary",
    "Haemoptysis": "Pulmonary",
    "Epistaxis": "Pulmonary",
    # Gastrointestinal
    "Diarrhoea": "Gastrointestinal",
    "Nausea": "Gastrointestinal",
    "Constipation": "Gastrointestinal",
    "Vomiting": "Gastrointestinal",
    "Stomatitis": "Gastrointestinal",
    "Dry Mouth": "Gastrointestinal",
    "Dysphagia": "Gastrointestinal",
    "Abdominal Pain": "Gastrointestinal",
    "Abdominal Pain Upper": "Gastrointestinal",
    "Abdominal Discomfort": "Gastrointestinal",
    "Abdominal Distension": "Gastrointestinal",
    "Dyspepsia": "Gastrointestinal",
    "Mouth Ulceration": "Gastrointestinal",
    "Gastrointestinal Disorder": "Gastrointestinal",
    "Taste Disorder": "Gastrointestinal",
    "Ageusia": "Gastrointestinal",
    "Dysgeusia": "Gastrointestinal",
    "Hypogeusia": "Gastrointestinal",
    "Mucosal Inflammation": "Gastrointestinal",
    "Oral Pain": "Gastrointestinal",
    "Hypoaesthesia Oral": "Gastrointestinal",
    "Oropharyngeal Pain": "Gastrointestinal",
    "Dry Throat": "Gastrointestinal",
    "Flatulence": "Gastrointestinal",
    "Gastritis": "Gastrointestinal",
    "Enterocolitis": "Gastrointestinal",
    "Ascites": "Gastrointestinal",
    # Neurological
    "Dizziness": "Neurological",
    "Headache": "Neurological",
    "Neuropathy Peripheral": "Neurological",
    "Confusional State": "Neurological",
    "Paraesthesia": "Neurological",
    "Hypoaesthesia": "Neurological",
    "Balance Disorder": "Neurological",
    "Gait Disturbance": "Neurological",
    "Brain Fog": "Neurological",
    "Syncope": "Neurological",
    "Loss Of Consciousness": "Neurological",
    "Depression": "Neurological",
    "Insomnia": "Neurological",
    "Initial Insomnia": "Neurological",
    "Somnolence": "Neurological",
    "Hypersomnia": "Neurological",
    "Poor Quality Sleep": "Neurological",
    "Anxiety": "Neurological",
    "Metastases To Central Nervous System": "Neurological",
    "Brain Neoplasm": "Neurological",
    # Musculoskeletal
    "Arthralgia": "Musculoskeletal",
    "Myalgia": "Musculoskeletal",
    "Back Pain": "Musculoskeletal",
    "Bone Pain": "Musculoskeletal",
    "Pain In Extremity": "Musculoskeletal",
    "Muscular Weakness": "Musculoskeletal",
    "Muscle Spasms": "Musculoskeletal",
    "Rhabdomyolysis": "Musculoskeletal",
    "Musculoskeletal Chest Pain": "Musculoskeletal",
    "Back Disorder": "Musculoskeletal",
    "Metastases To Bone": "Musculoskeletal",
    "Blood Creatine Phosphokinase Increased": "Musculoskeletal",
    # Renal
    "Renal Impairment": "Renal",
    "Blood Creatinine Increased": "Renal",
    "Renal Function Test Abnormal": "Renal",
    "Dysuria": "Renal",
    # Metabolic/Endocrine
    "Hypothyroidism": "Metabolic/Endocrine",
    "Blood Glucose Increased": "Metabolic/Endocrine",
    "Hyponatraemia": "Metabolic/Endocrine",
    "Blood Sodium Decreased": "Metabolic/Endocrine",
    "Hypoalbuminaemia": "Metabolic/Endocrine",
    "Hyperkalaemia": "Metabolic/Endocrine",
    "Dehydration": "Metabolic/Endocrine",
    "Weight Decreased": "Metabolic/Endocrine",
    "Weight Increased": "Metabolic/Endocrine",
    "Decreased Appetite": "Metabolic/Endocrine",
    "Tumour Lysis Syndrome": "Metabolic/Endocrine",
    # Skin
    "Rash": "Skin",
    "Pruritus": "Skin",
    "Alopecia": "Skin",
    "Rash Macular": "Skin",
    "Hyperhidrosis": "Skin",
    "Palmar-Plantar Erythrodysaesthesia Syndrome": "Skin",
    "Eczema": "Skin",
    # Oedema/Fluid
    "Oedema": "Oedema/Fluid",
    "Oedema Peripheral": "Oedema/Fluid",
    "Generalised Oedema": "Oedema/Fluid",
    "Face Oedema": "Oedema/Fluid",
    "Peripheral Swelling": "Oedema/Fluid",
    "Swelling Face": "Oedema/Fluid",
    "Swelling Of Eyelid": "Oedema/Fluid",
    "Eye Swelling": "Oedema/Fluid",
    "Periorbital Swelling": "Oedema/Fluid",
    "Limb Discomfort": "Oedema/Fluid",
    # Cell Death/Apoptosis
    "Myocardial Necrosis Marker Increased": "Cell Death/Apoptosis",
    # Proliferative
    "Disease Progression": "Proliferative",
    "Malignant Neoplasm Progression": "Proliferative",
    "Neoplasm Malignant": "Proliferative",
    "Neoplasm Progression": "Proliferative",
    "Metastatic Neoplasm": "Proliferative",
    "Drug Resistance": "Proliferative",
    "Carcinoembryonic Antigen Increased": "Proliferative",
}


# ─────────────────────────────────────────────────────────────────────────────
# GO → AE THEME MAPPING
# Covers all 146 GO terms in the KG (3 proteins: JAK2, RET, FLT3).
# The original KG had only 4 maps_to edges; this replaces and extends them.
# Sources: GO annotations + UniProt function text + RET/JAK2/FLT3 literature.
# ─────────────────────────────────────────────────────────────────────────────
GO_THEME_MAP: dict[str, set[str]] = {
    # ── JAK2 (O60674) ──
    "GO:0032760": {"Immune/Infection", "Haematological"},
    "GO:1901731": {"Haematological"},
    "GO:0007167": {"Haematological", "Immune/Infection"},
    "GO:0007259": {"Haematological", "Immune/Infection"},
    "GO:0008285": {"Proliferative", "Haematological"},
    "GO:0010572": {"Haematological"},
    "GO:0030099": {"Haematological"},
    "GO:0030154": {"Haematological", "Immune/Infection"},
    "GO:0034050": {"Immune/Infection"},
    "GO:0035556": {"Immune/Infection", "Haematological"},
    "GO:0036016": {"Haematological"},
    "GO:0043066": {"Cell Death/Apoptosis"},
    "GO:0046677": {"Immune/Infection"},
    "GO:0060391": {"Haematological"},
    "GO:0061180": {"Immune/Infection"},
    "GO:0071222": {"Immune/Infection"},
    "GO:0071549": {"Haematological"},
    "GO:0097191": {"Cell Death/Apoptosis"},
    "GO:2001235": {"Cell Death/Apoptosis"},
    "GO:0001774": {"Immune/Infection", "Haematological"},
    "GO:0045087": {"Immune/Infection"},
    "GO:0002376": {"Immune/Infection"},
    "GO:0045348": {"Haematological"},
    "GO:0045428": {"Immune/Infection"},
    "GO:0045429": {"Immune/Infection"},
    "GO:0045793": {"Haematological"},
    "GO:0045893": {"Haematological", "Immune/Infection"},
    "GO:0045944": {"Proliferative"},
    "GO:0046425": {"Haematological"},
    "GO:0046634": {"Immune/Infection"},
    "GO:0046651": {"Haematological"},
    "GO:0046777": {"Immune/Infection", "Haematological"},
    "GO:0019221": {"Immune/Infection"},
    "GO:0050727": {"Immune/Infection"},
    "GO:0050770": {"Neurological"},
    "GO:0050804": {"Neurological"},
    "GO:0050867": {"Haematological"},
    "GO:0051897": {"Cardiovascular"},
    "GO:0060333": {"Immune/Infection"},
    "GO:0060396": {"Haematological"},
    "GO:0060397": {"Haematological"},
    "GO:0060399": {"Haematological"},
    "GO:0070102": {"Immune/Infection"},
    "GO:0070665": {"Haematological"},
    "GO:0070671": {"Haematological"},
    "GO:0070757": {"Haematological"},
    "GO:0031663": {"Immune/Infection"},
    "GO:0031959": {"Metabolic/Endocrine"},
    "GO:0032024": {"Metabolic/Endocrine"},
    "GO:0032731": {"Immune/Infection"},
    "GO:0033194": {"Haematological"},
    "GO:0033209": {"Proliferative"},
    "GO:0033619": {"Haematological"},
    "GO:0033630": {"Haematological"},
    "GO:0034612": {"Haematological"},
    "GO:0035166": {"Haematological"},
    "GO:0038043": {"Haematological"},
    "GO:0038065": {"Haematological"},
    "GO:0038155": {"Haematological"},
    "GO:0038156": {"Haematological"},
    "GO:0038157": {"Haematological"},
    "GO:0038162": {"Haematological"},
    "GO:0038163": {"Haematological"},
    "GO:0042102": {"Haematological"},
    "GO:0042307": {"Haematological"},
    "GO:0042531": {"Immune/Infection"},
    "GO:0042551": {"Neurological"},
    "GO:0042976": {"Cell Death/Apoptosis"},
    "GO:0042981": {"Cell Death/Apoptosis"},
    "GO:0043065": {"Cell Death/Apoptosis"},
    "GO:0043410": {"Proliferative"},
    "GO:0043524": {"Neurological"},
    "GO:0043687": {"Haematological"},
    "GO:0009755": {"Immune/Infection"},
    "GO:0010628": {"Haematological"},
    "GO:0010667": {"Cell Death/Apoptosis"},
    "GO:0010811": {"Neurological"},
    "GO:0010976": {"Neurological"},
    "GO:1900016": {"Immune/Infection"},
    "GO:1902533": {"Haematological"},
    "GO:1902728": {"Proliferative"},
    "GO:1904037": {"Cell Death/Apoptosis"},
    "GO:1904707": {"Haematological"},
    "GO:1905539": {"Haematological"},
    # ── RET (P07949) ──
    "GO:0000165": {"Neurological"},
    "GO:0001657": {"Renal"},
    "GO:0001755": {"Neurological"},
    "GO:0001838": {"Renal"},
    "GO:0007169": {"Proliferative"},
    "GO:0007204": {"Cardiovascular"},
    "GO:0007399": {"Neurological"},
    "GO:0007411": {"Neurological"},
    "GO:0007497": {"Renal"},
    "GO:0007498": {"Renal"},
    "GO:0008284": {"Proliferative"},
    "GO:0008631": {"Cell Death/Apoptosis"},
    "GO:0030041": {"Musculoskeletal"},
    "GO:0030155": {"Haematological"},
    "GO:0030182": {"Neurological"},
    "GO:0030218": {"Haematological"},
    "GO:0030335": {"Proliferative"},
    "GO:0031103": {"Neurological"},
    "GO:0048008": {"Proliferative"},
    "GO:0048265": {"Neurological"},
    "GO:0048484": {"Neurological"},
    "GO:0061146": {"Renal"},
    "GO:0072300": {"Renal"},
    "GO:0097021": {"Renal"},
    "GO:0120162": {"Metabolic/Endocrine"},
    "GO:0120186": {"Metabolic/Endocrine"},
    "GO:0140546": {"Immune/Infection"},
    "GO:0160144": {"Renal"},
    "GO:2001241": {"Neurological"},
    "GO:0022407": {"Neurological"},
    "GO:0022408": {"Neurological"},
    "GO:0035722": {"Haematological"},
    "GO:0035799": {"Renal"},
    "GO:0035860": {"Neurological"},
    "GO:0043406": {"Proliferative"},
    # ── FLT3 (P36888) ──
    "GO:0030097": {"Haematological"},
    "GO:0001776": {"Haematological"},
    "GO:0002318": {"Haematological"},
    "GO:0002328": {"Haematological"},
    "GO:0007166": {"Haematological"},
    "GO:0010604": {"Proliferative"},
    "GO:0018108": {"Haematological"},
    "GO:0030183": {"Haematological"},
    "GO:0035726": {"Haematological"},
    "GO:0038084": {"Haematological"},
    "GO:0046651": {"Haematological"},
    "GO:0071345": {"Haematological"},
    "GO:0071385": {"Haematological"},
    "GO:0097028": {"Haematological"},
    "GO:0097421": {"Cell Death/Apoptosis"},
    "GO:0016477": {"Proliferative"},
    # ── Shared / chromatin ──
    "GO:0006325": {"Proliferative"},
    "GO:0006338": {"Proliferative"},
    "GO:0006915": {"Cell Death/Apoptosis"},
    "GO:0006979": {"Skin", "Hepatic", "Pulmonary"},
    "GO:0007155": {"Cardiovascular", "Oedema/Fluid"},
    "GO:0007156": {"Cardiovascular"},
    "GO:0007158": {"Cardiovascular"},
    "GO:0007165": {"Haematological", "Proliferative"},
}


# ─────────────────────────────────────────────────────────────────────────────
# KNOWLEDGE GRAPH LOADER
# ─────────────────────────────────────────────────────────────────────────────
class KnowledgeGraph:
    """
    Loads and indexes the Pralsetinib KG from CSV files.

    Indexes built:
      drug_protein     : drug_id → [(protein_id, role)]
      protein_go       : protein_id → {go_node_id}
      go_theme         : go_node_id → {theme_str}   (merged with GO_THEME_MAP)
      drug_ae_count    : drug_id → {ae_node_id: count}
      ae_theme_direct  : ae_node_id → theme_str     (via AE_THEME_MAP)
      go_degree        : go_node_id → int            (for specificity weighting)
    """

    ROLE_WEIGHTS = {
        "target":      1.00,
        "enzyme":      0.70,
        "transporter": 0.70,
        "carrier":     0.65,
        "other":       0.55,
        "":            0.55,
    }

    def __init__(self, nodes_path: Path, edges_path: Path):
        self.nodes = pd.read_csv(nodes_path)
        self.edges = pd.read_csv(edges_path)
        self._build_index()

    def _build_index(self):
        e = self.edges
        self.drug_protein:   dict[str, list]          = defaultdict(list)
        self.protein_go:     dict[str, set[str]]       = defaultdict(set)
        self.go_theme:       dict[str, set[str]]       = defaultdict(set)
        self.drug_ae_count:  dict[str, dict[str,float]]= defaultdict(dict)
        self.go_degree:      dict[str, int]            = defaultdict(int)

        for _, r in e.iterrows():
            src, etype, tgt = str(r["source"]), str(r["edge_type"]), str(r["target"])

            if etype == "binds_to":
                role_raw = str(r.get("evidence", "") or r.get("interaction_role", "") or "").strip().lower()
                role = role_raw if role_raw and role_raw not in ("nan", "none") else "target"
                self.drug_protein[src].append((tgt, role))

            elif etype == "involved_in":
                self.protein_go[src].add(tgt)
                self.go_degree[tgt] += 1

            elif etype == "maps_to":
                self.go_theme[src].add(tgt)

            elif etype == "reported_with":
                cnt = float(r.get("count", 1) or 1)
                prev = self.drug_ae_count[src].get(tgt, 0.0)
                self.drug_ae_count[src][tgt] = prev + cnt

        # Merge comprehensive GO→theme map (replaces sparse 4-edge original)
        go_id_to_node: dict[str, str] = {}
        for _, row in self.nodes[self.nodes["node_type"] == "BiologicalProcess"].iterrows():
            raw = str(row.get("go_id", ""))
            if raw:
                go_id_to_node[raw] = str(row["node_id"])

        for raw_go_id, themes in GO_THEME_MAP.items():
            node_id = go_id_to_node.get(raw_go_id)
            if node_id:
                self.go_theme[node_id] |= themes

        # AE node → theme lookup
        self.ae_theme_direct: dict[str, str] = {}
        for drug_id, ae_dict in self.drug_ae_count.items():
            for ae_node_id in ae_dict:
                label = ae_node_id.replace("ae:", "").strip()
                if label in AE_THEME_MAP:
                    self.ae_theme_direct[ae_node_id] = AE_THEME_MAP[label]

    def role_weight(self, role: str) -> float:
        role = (role or "").strip().lower()
        for key in self.ROLE_WEIGHTS:
            if key and key in role:
                return self.ROLE_WEIGHTS[key]
        return self.ROLE_WEIGHTS[""]

    def go_specificity(self, go_node: str, power: float = 0.8) -> float:
        """Down-weight high-degree (generic) GO terms."""
        deg = float(self.go_degree.get(go_node, 1))
        return 1.0 / (deg ** power)

    def summary(self) -> str:
        n_prot    = len(self.drug_protein.get(DRUG_ID, []))
        n_go      = sum(len(v) for v in self.protein_go.values())
        n_mapped  = sum(len(v) for v in self.go_theme.values())
        n_ae      = len(self.drug_ae_count.get(DRUG_ID, {}))
        return (
            f"  Proteins bound : {n_prot}\n"
            f"  GO terms       : {n_go}\n"
            f"  GO→theme edges : {n_mapped}\n"
            f"  AEs (FAERS)    : {n_ae}"
        )