#!/usr/bin/env python3
"""
model1_baseline.py
==================
Model 1: FAERS-Only Frequency Baseline
DSC 180B Capstone — UC San Diego

What this model does
--------------------
Ranks toxicity themes by raw adverse event report counts from FAERS.
No knowledge graph. No biological reasoning. Just: "what got reported most?"

This is the comparison point for Models 2 and 3. It represents the state of
the art for simple pharmacovigilance: count-based signal detection. The key
question this model answers is whether frequency alone is a reliable guide
to *mechanistically important* toxicities — spoiler: it isn't.

Data source: kg_edges_stripped.csv  (reported_with edges only)
No KG, no GO terms, no protein interactions used.

Run:
    cd models/
    python3 model1_baseline.py
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pandas as pd

from kg_shared import DRUG_ID, FAERS_ONLY_FILE, AE_THEME_MAP

SEP  = "=" * 78
SEP2 = "─" * 78


# ─────────────────────────────────────────────────────────────────────────────
# CORE FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def run_baseline(faers_path: Path, drug_id: str, topk: int = 14) -> list[dict]:
    """
    Read reported_with edges directly from the stripped CSV and rank
    toxicity themes by total FAERS report count.

    Returns a list of dicts with keys:
      theme, faers_count, faers_pct, top_aes, n_ae_types
    """
    df = pd.read_csv(faers_path)
    rw = df[
        (df["edge_type"] == "reported_with") &
        (df["source"] == drug_id)
    ].copy()
    rw["count"] = pd.to_numeric(rw["count"], errors="coerce").fillna(1.0)

    theme_totals: dict[str, float] = defaultdict(float)
    theme_aes:    dict[str, list]  = defaultdict(list)

    for _, row in rw.iterrows():
        ae_label = str(row["target"]).replace("ae:", "").strip()
        cnt      = float(row["count"])
        theme    = AE_THEME_MAP.get(ae_label, "Other")
        theme_totals[theme] += cnt
        theme_aes[theme].append((ae_label, cnt))

    total = sum(theme_totals.values()) or 1.0
    ranked = sorted(theme_totals.items(), key=lambda x: x[1], reverse=True)

    results = []
    for theme, cnt in ranked[:topk]:
        aes = sorted(theme_aes[theme], key=lambda x: x[1], reverse=True)
        results.append({
            "theme":       theme,
            "faers_count": cnt,
            "faers_pct":   cnt / total * 100,
            "top_aes":     aes[:3],
            "n_ae_types":  len(aes),
        })
    return results


# ─────────────────────────────────────────────────────────────────────────────
# FINDINGS COMMENTARY
# ─────────────────────────────────────────────────────────────────────────────

FINDINGS = """
FINDINGS — Model 1: FAERS-Only Baseline
────────────────────────────────────────────────────────────────────────────────

1. DOMINANCE OF NON-BIOLOGICAL NOISE (~32% of reports)
   ─────────────────────────────────────────────────────
   The single largest "theme" is "Other" — a catch-all for administrative events
   like "Product Prescribing Issue" (131 reports), "Off Label Use" (65), and
   "Hospitalisation" (30). These top the frequency chart simply because FAERS
   captures all drug-related reports, not just true adverse reactions.

   Implication: a frequency-only system would flag administrative noise as the
   primary safety signal. This is the core weakness of the baseline.

2. GASTROINTESTINAL IS OVERREPRESENTED (14% of reports, rank 2)
   ─────────────────────────────────────────────────────────────
   GI events (diarrhoea, constipation, nausea) rank second. These are extremely
   common across all oral cancer therapies — they reflect drug class effects and
   patient population rather than Pralsetinib-specific off-target biology.

   In FAERS, high-report conditions are partly driven by reporting culture: GI
   symptoms are easily noticed and reported by patients. This inflates their rank
   relative to more mechanistically significant but less visible toxicities.

3. HAEMATOLOGICAL RANKS 3rd — BIOLOGICALLY CONSISTENT
   ────────────────────────────────────────────────────
   Cytopenias, anaemia and neutropenia appear at rank 3 (9.3%). This is genuinely
   mechanistic — JAK2 (one of Pralsetinib's targets) is critical for haematopoiesis
   and cytokine signaling. The baseline gets this right for the right wrong reasons:
   it's frequent *because* it's mechanistically driven.

   However, the baseline can't tell you *why* — it has no knowledge of JAK2's role
   in myeloid cell differentiation. See Model 2 for that explanation.

4. IMMUNE/INFECTION RANKED 7th — LIKELY UNDERVALUED
   ──────────────────────────────────────────────────
   Pneumonia, sepsis, and opportunistic infections rank 7th by count (5.2%). Given
   that JAK2 inhibition directly suppresses innate immune signaling (IL-6, IFN-γ,
   TNF pathways), this is mechanistically a top-tier concern.

   The baseline's rank-7 placement reflects reporting lag: serious infections often
   develop gradually, may be attributed to the underlying cancer, and are harder for
   patients to self-report. Mechanistic models (Model 2) upgrade this appropriately.

5. RENAL RANKS LAST (1.4%) — BUT HAS STRONG MECHANISTIC BASIS
   ────────────────────────────────────────────────────────────
   Renal impairment appears at the bottom of the frequency chart. However, RET
   (another Pralsetinib target) plays a direct developmental role in kidney
   morphogenesis and ureteric bud formation (GO:0001657, GO:0035799). In adults,
   RET inhibition can impair renal tubular maintenance.

   This is the clearest example of where frequency-only analysis fails: a low
   FAERS count doesn't mean low mechanistic risk. Model 2 ranks Renal at position
   8 despite only 31 reports — purely from path evidence through RET biology.

OVERALL VERDICT
   The baseline is useful as a starting point — it correctly identifies some
   biologically real signals (haematological toxicity). But it cannot distinguish
   mechanistically important low-frequency events from incidentally common
   high-frequency noise. Without the knowledge graph, 32% of "signal" is noise
   and critical risks like Immune/Infection and Renal are structurally depressed.
"""


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{SEP}")
    print(f"  MODEL 1: FAERS-Only Baseline  |  Pralsetinib  |  DSC 180B")
    print(f"  Source: {FAERS_ONLY_FILE.name}  |  No KG  |  No Ontology")
    print(SEP)

    results = run_baseline(FAERS_ONLY_FILE, DRUG_ID, topk=14)

    # Count stats
    total_reports = sum(r["faers_count"] for r in results)
    total_ae_types = sum(r["n_ae_types"] for r in results)
    print(f"\n  Total reports in top-14 themes : {total_reports:.0f}")
    print(f"  Distinct AE types covered      : {total_ae_types}")
    print(f"  Data source                    : FAERS (openFDA, post-marketing)")

    print(f"\n{'Rank':<5} {'Theme':<28} {'Reports':>8} {'%Total':>7} {'#AE Types':>10}  Top 3 AEs")
    print(SEP2)
    for i, r in enumerate(results, 1):
        top_str = ", ".join(f"{a} ({int(c)})" for a, c in r["top_aes"])
        print(f"{i:<5} {r['theme']:<28} {r['faers_count']:>8.0f} "
              f"{r['faers_pct']:>6.1f}% {r['n_ae_types']:>10}  {top_str}")

    # Save results CSV
    out = Path(__file__).parent / "results_model1_baseline.csv"
    pd.DataFrame(results).to_csv(out, index=False)
    print(f"\n  Results saved → {out.name}")

    print(FINDINGS)


if __name__ == "__main__":
    main()