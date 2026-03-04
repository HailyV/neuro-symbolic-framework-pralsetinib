#!/usr/bin/env python3
"""
plot_three_models.py
====================
Publication-quality figure combining all three models for the
Pralsetinib drug safety capstone poster.

Panels:
  A — Model 1 vs Model 2: side-by-side rank comparison (dot + line)
      showing how Bayesian path scoring reorders FAERS-only ranking
  B — Model 3 scatter: KG path count vs FAERS count,
      with LOOCV classification outcome encoded as marker shape/colour
  C — Model 3 LOOCV: predicted probability bar chart, sorted by prob_high,
      coloured by true label, marker for correct/incorrect

Usage:
    Put this file in the same folder as your CSVs and run:
        python3 plot_three_models.py

Requirements:
    pip install matplotlib pandas numpy scipy
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import numpy as np
import pandas as pd
from scipy import stats

# ─────────────────────────────────────────────────────────────────────────────
# FILE PATHS  — edit if needed
# ─────────────────────────────────────────────────────────────────────────────
M1_CSV    = "../04_model_summaries/results_model1_baseline.csv"
M2_CSV    = "../04_model_summaries/results_model2_bayes.csv"
M3_CSV    = "../04_model_summaries/model3_loocv_results.csv"
OUT_PNG   = "../../figures/figure_all_models.png"
OUT_PDF   = "../../figures/figure_all_models.pdf"

# ─────────────────────────────────────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────────────────────────────────────
m1 = pd.read_csv(M1_CSV)
m2 = pd.read_csv(M2_CSV)
m3 = pd.read_csv(M3_CSV)

# ─────────────────────────────────────────────────────────────────────────────
# PREP — build a shared theme ranking table (exclude "Other" — admin noise)
# ─────────────────────────────────────────────────────────────────────────────

m1 = m1[m1["theme"] != "Other"].copy()
m2 = m2[m2["theme"] != "Other"].copy()

# Assign ranks
m1 = m1.sort_values("faers_count", ascending=False).reset_index(drop=True)
m1["rank_m1"] = range(1, len(m1) + 1)

m2 = m2.sort_values("posterior", ascending=False).reset_index(drop=True)
m2["rank_m2"] = range(1, len(m2) + 1)

# Merge on theme for bump chart
bump = m1[["theme","rank_m1","faers_count"]].merge(
    m2[["theme","rank_m2","posterior","n_paths"]], on="theme", how="outer"
)
bump = bump.dropna(subset=["rank_m1","rank_m2"])
bump = bump.sort_values("rank_m1")

# LOOCV  — parse correct flag
m3["is_correct"] = m3["correct"].astype(str).str.strip().isin(["True","✓","1","true"])
m3 = m3[m3["theme"] != "Other"].copy()
m3 = m3.sort_values("prob_high", ascending=False).reset_index(drop=True)

# Spearman rho for scatter annotation
rho, pval = stats.spearmanr(m3["kg_path_count"], m3["faers_count"])

# ─────────────────────────────────────────────────────────────────────────────
# COLOURS & STYLE
# ─────────────────────────────────────────────────────────────────────────────
BG     = "#FAFBFC"
DARK   = "#1C1F2E"
GREY   = "#9098B0"
LGREY  = "#D8DCE8"

C_M1   = "#4A7FBB"   # blue  — FAERS baseline
C_M2   = "#2BAE85"   # teal  — Bayesian
C_UND  = "#D94F4F"   # red   — underreported / low-FAERS misclassified
C_VAL  = "#2BAE85"   # teal  — validated
C_NOISE= "#AAB0C4"   # grey  — noise / low KG

# Themes that are the "story" — underreported candidates
UNDERREPORTED = {"Cell Death/Apoptosis", "Renal", "Proliferative"}
VALIDATED     = {"Haematological", "Immune/Infection", "Neurological"}

plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "axes.linewidth":    0.75,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "xtick.major.size":  3,
    "ytick.major.size":  3,
    "xtick.labelsize":   7.5,
    "ytick.labelsize":   7.5,
    "axes.labelsize":    8.5,
})

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE
# ─────────────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 6.4), dpi=180, facecolor=BG)
fig.patch.set_facecolor(BG)

gs = gridspec.GridSpec(
    1, 3, figure=fig,
    left=0.04, right=0.98,
    top=0.82, bottom=0.12,
    wspace=0.40,
    width_ratios=[1.15, 1.1, 1.0],
)

ax_a = fig.add_subplot(gs[0])
ax_b = fig.add_subplot(gs[1])
ax_c = fig.add_subplot(gs[2])

for ax in [ax_a, ax_b, ax_c]:
    ax.set_facecolor(BG)

# ═════════════════════════════════════════════════════════════════════════════
# PANEL A — Rank Bump Chart: Model 1 vs Model 2
# ═════════════════════════════════════════════════════════════════════════════

n_themes = len(bump)
ax_a.set_xlim(-0.6, 1.6)
ax_a.set_ylim(n_themes + 0.6, 0.4)   # rank 1 at top
ax_a.set_xticks([0, 1])
ax_a.set_xticklabels(["Model 1\nFAERS Baseline", "Model 2\nBayesian Paths"],
                     fontsize=8.5, fontweight="bold")
ax_a.set_yticks(range(1, n_themes + 1))
ax_a.set_yticklabels([])
ax_a.set_ylabel("Rank  (1 = highest risk)", labelpad=4)
ax_a.tick_params(left=False)
ax_a.spines["left"].set_visible(False)
ax_a.set_title("A   Rank Shift: FAERS Baseline -> Bayesian",
               fontweight="bold", loc="left", pad=8, fontsize=9, color=DARK)

for _, row in bump.iterrows():
    t   = row["theme"]
    r1  = row["rank_m1"]
    r2  = row["rank_m2"]
    delta = r1 - r2   # positive = promoted in M2

    # Line colour
    if t in UNDERREPORTED:
        lc = C_UND;  lw = 2.0; alpha = 0.95
    elif t in VALIDATED:
        lc = C_M2;   lw = 1.8; alpha = 0.90
    elif abs(delta) >= 3:
        lc = "#E09030"; lw = 1.4; alpha = 0.85
    else:
        lc = LGREY;  lw = 1.0; alpha = 0.70

    ax_a.plot([0, 1], [r1, r2], color=lc, lw=lw, alpha=alpha,
              solid_capstyle="round", zorder=3)
    ax_a.scatter([0, 1], [r1, r2], s=32, color=lc, zorder=5, alpha=alpha)

    # Left label (Model 1 rank)
    ax_a.text(-0.08, r1, t, ha="right", va="center", fontsize=6.8,
              color=lc if lc != LGREY else GREY,
              fontweight="semibold" if t in UNDERREPORTED | VALIDATED else "normal")

    # Right label with delta arrow
    label_right = t
    if abs(delta) >= 2:
        arrow = f" (+{delta})" if delta > 0 else f" (-{abs(delta)})"
        ax_a.text(1.08, r2, label_right + arrow, ha="left", va="center",
                  fontsize=6.8, color=lc,
                  fontweight="semibold" if t in UNDERREPORTED | VALIDATED else "normal")
    else:
        ax_a.text(1.08, r2, label_right, ha="left", va="center",
                  fontsize=6.8, color=GREY)

# Legend A
leg_a = [
    mlines.Line2D([], [], color=C_M2,    lw=2,   label="Promoted by biology (validated)"),
    mlines.Line2D([], [], color=C_UND,   lw=2,   label="High KG · low FAERS (underreported)"),
    mlines.Line2D([], [], color="#E09030", lw=1.4, label="Notable rank change (|Δ| ≥ 3)"),
    mlines.Line2D([], [], color=LGREY,   lw=1,   label="Minor / no change"),
]
ax_a.legend(handles=leg_a, fontsize=6.5, loc="lower right",
            framealpha=0.85, edgecolor=LGREY, handlelength=1.5)

# ═════════════════════════════════════════════════════════════════════════════
# PANEL B — Model 3 Scatter: KG Paths vs FAERS Count
#           marker shape = LOOCV correct/incorrect
# ═════════════════════════════════════════════════════════════════════════════

med_x = m3["kg_path_count"].median()
med_y = m3["faers_count"].median()

# Shaded quadrants
xlim_b = (m3["kg_path_count"].max() * 1.18)
ylim_b = (m3["faers_count"].max()   * 1.15)

ax_b.fill_between([med_x, xlim_b], med_y, ylim_b,
                  color=C_VAL,  alpha=0.05, zorder=0)
ax_b.fill_between([0, med_x], med_y, ylim_b,
                  color=GREY,   alpha=0.04, zorder=0)
ax_b.fill_between([med_x, xlim_b], 0, med_y,
                  color=C_UND,  alpha=0.06, zorder=0)
ax_b.fill_between([0, med_x], 0, med_y,
                  color=GREY,   alpha=0.03, zorder=0)

ax_b.axhline(med_y, color=LGREY, lw=0.9, ls="--", zorder=1)
ax_b.axvline(med_x, color=LGREY, lw=0.9, ls="--", zorder=1)

# Quadrant annotations
ax_b.text(xlim_b * 0.98, ylim_b * 0.97, "High KG · High FAERS\n(Validated)",
          ha="right", va="top", fontsize=6.5, color=C_VAL, style="italic", alpha=0.8)
ax_b.text(med_x * 0.02,  ylim_b * 0.97, "Low KG · High FAERS\n(Class effects)",
          ha="left",  va="top", fontsize=6.5, color=GREY,  style="italic", alpha=0.8)
ax_b.text(xlim_b * 0.98, ylim_b * 0.04, "High KG · Low FAERS\n[!] Underreported",
          ha="right", va="bottom", fontsize=6.5, color=C_UND, style="italic", alpha=0.85)

for _, row in m3.iterrows():
    t  = row["theme"]
    kp = row["kg_path_count"]
    fc = row["faers_count"]
    correct = row["is_correct"]

    # Colour by quadrant
    if kp > med_x and fc <= med_y:
        c = C_UND
    elif kp > med_x and fc > med_y:
        c = C_VAL
    else:
        c = C_NOISE

    mk  = "o" if correct else "X"
    sz  = 55 + fc * 0.12
    ec  = "white"
    lw  = 0.6 if correct else 0.9

    ax_b.scatter(kp, fc, s=sz, color=c, marker=mk,
                 edgecolors=ec, linewidth=lw, zorder=5, alpha=0.88)

# Labels for key themes only
label_offsets_b = {
    "Haematological":      ( 1.5,  18),
    "Cell Death/Apoptosis":( 1.5, -22),
    "Renal":               ( 1.5,  18),
    "Immune/Infection":    ( 1.5,  18),
    "Neurological":        (-3,   -28),
    "Proliferative":       ( 1.5,  18),
    "Gastrointestinal":    (-5,    18),
}
for _, row in m3.iterrows():
    t = row["theme"]
    if t not in label_offsets_b:
        continue
    kp = row["kg_path_count"]; fc = row["faers_count"]
    ox, oy = label_offsets_b[t]
    c = C_UND if (kp > med_x and fc <= med_y) else (C_VAL if kp > med_x else C_NOISE)
    ax_b.annotate(t, (kp, fc), xytext=(kp + ox, fc + oy),
                  fontsize=6.8, color=c, fontweight="semibold",
                  arrowprops=dict(arrowstyle="-", color=c, lw=0.5))

ax_b.set_xlim(-2, xlim_b)
ax_b.set_ylim(-15, ylim_b)
ax_b.set_xlabel("KG Path Count  (mechanistic evidence)", labelpad=4)
ax_b.set_ylabel("FAERS Report Count", labelpad=4)
ax_b.set_title(
    f"B   Model 3: KG Evidence vs FAERS Frequency\n"
    f"    Spearman ρ = {rho:.2f}  (p = {pval:.2f})",
    fontweight="bold", loc="left", pad=8, fontsize=9, color=DARK,
)

# Marker legend
leg_b = [
    mlines.Line2D([], [], marker="o", color="w", markerfacecolor=C_VAL,
                  markersize=7, label="Validated  (correct)"),
    mlines.Line2D([], [], marker="o", color="w", markerfacecolor=C_UND,
                  markersize=7, label="Underreported  (correct)"),
    mlines.Line2D([], [], marker="X", color="w", markerfacecolor=C_NOISE,
                  markersize=7, label="Misclassified"),
    mlines.Line2D([], [], marker="o", color="w", markerfacecolor=C_NOISE,
                  markersize=7, label="Low KG evidence"),
]
ax_b.legend(handles=leg_b, fontsize=6.5, loc="upper left",
            framealpha=0.85, edgecolor=LGREY, handlelength=1.0,
            bbox_to_anchor=(0.0, 0.72))

# ═════════════════════════════════════════════════════════════════════════════
# PANEL C — Model 3 LOOCV: Predicted Probability Bars
# ═════════════════════════════════════════════════════════════════════════════

m3_sorted = m3.sort_values("prob_high", ascending=True).reset_index(drop=True)

bar_colors = []
edge_colors = []
for _, row in m3_sorted.iterrows():
    t = row["theme"]
    if row["true_label"] == 1 and row["is_correct"]:
        c = C_M2        # High FAERS, correctly flagged
    elif row["true_label"] == 1 and not row["is_correct"]:
        c = "#E09030"   # High FAERS, missed — false negative
    elif row["true_label"] == 0 and not row["is_correct"]:
        c = C_UND       # Low FAERS, predicted high — potential underreported
    else:
        c = C_M1        # Low FAERS, correctly quiet
    bar_colors.append(c)
    edge_colors.append("none")

bars = ax_c.barh(
    m3_sorted["theme"], m3_sorted["prob_high"],
    color=bar_colors, height=0.68,
    linewidth=0, alpha=0.88
)

# Decision boundary
ax_c.axvline(0.5, color=DARK, lw=1.0, ls="--", zorder=10, alpha=0.6)
ax_c.text(0.51, -0.9, "threshold", fontsize=6, color=DARK, alpha=0.55, style="italic")

# Correct/incorrect markers at end of bar
for i, (_, row) in enumerate(m3_sorted.iterrows()):
    sym  = "[ok]" if row["is_correct"] else "[x]"
    col  = DARK if row["is_correct"] else C_UND
    ax_c.text(row["prob_high"] + 0.02, i, sym, va="center",
              fontsize=7.5, color=col, fontweight="bold")

ax_c.set_xlim(0, 1.12)
ax_c.set_xlabel("LOOCV Predicted Probability\n(of High-FAERS label)", labelpad=4)
ax_c.tick_params(axis='y', labelsize=7.5)
ax_c.spines["left"].set_visible(False)
ax_c.tick_params(left=False)

# Accuracy annotation
acc = m3["is_correct"].mean()
ax_c.text(0.98, 0.02, f"Accuracy = {acc:.0%}  (n={len(m3)})",
          transform=ax_c.transAxes, ha="right", va="bottom",
          fontsize=7.5, color=DARK,
          bbox=dict(fc="white", ec=LGREY, boxstyle="round,pad=0.35", alpha=0.9))

ax_c.set_title("C   Model 3: LOOCV Classification",
               fontweight="bold", loc="left", pad=8, fontsize=9, color=DARK)

leg_c = [
    mpatches.Patch(fc=C_M2,      label="True High-FAERS — correctly classified"),
    mpatches.Patch(fc="#E09030", label="True High-FAERS — missed  (false negative)"),
    mpatches.Patch(fc=C_UND,     label="True Low-FAERS — flagged  (potential underreported)"),
    mpatches.Patch(fc=C_M1,      label="True Low-FAERS — correctly quiet"),
]
ax_c.legend(handles=leg_c, fontsize=6.5, loc="lower right",
            framealpha=0.85, edgecolor=LGREY, handlelength=1.2)

# ─────────────────────────────────────────────────────────────────────────────
# TITLE + CAPTION
# ─────────────────────────────────────────────────────────────────────────────
fig.text(
    0.5, 0.96,
    "Neuro-Symbolic Drug Safety Analysis: Pralsetinib (CHEMBL4582651)",
    ha="center", va="top", fontsize=12, fontweight="bold", color=DARK,
)
fig.text(
    0.5, 0.935,
    "(A) Rank shifts from FAERS-only baseline to Bayesian path scoring -- underreported candidates "
    "promoted by biological evidence.  "
    f"(B) Near-zero Spearman ρ = {rho:.2f} between KG path count and FAERS frequency demonstrates "
    "complementarity of signals; LOOCV marker shape encodes classification outcome.  "
    "(C) LOOCV predicted probabilities; Low-FAERS themes misclassified as high-signal "
    "(red) are candidate underreported adverse events.",
    ha="center", va="top", fontsize=7, color=GREY,
)

# ─────────────────────────────────────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────────────────────────────────────
plt.savefig(OUT_PNG, dpi=300, bbox_inches="tight", facecolor=BG)
plt.savefig(OUT_PDF, bbox_inches="tight", facecolor=BG)
print(f"Saved → {OUT_PNG}")
print(f"Saved → {OUT_PDF}")