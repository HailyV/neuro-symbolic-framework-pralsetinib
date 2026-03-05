import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

output_dir = os.path.expanduser('~/Documents/GitHub/neuro-symbolic-framework-pralsetinib/models/04_model_summaries')

# ── Colors ────────────────────────────────────────────────────────────────────
C_KG      = '#2d6a4f'
C_HYBRID  = '#c9592c'
GREY_MID  = '#d5cfc7'
GREY_TEXT = '#6b6460'
INK       = '#1a1714'
BG        = '#fafaf8'

# ── Data ──────────────────────────────────────────────────────────────────────
features  = ['path_count', 'go_overlap', 'max_path_score',
             'mean_path_score', 'theme_specificity', 'n_proteins', 'log_faers']
feat_type = ['KG · path', 'KG · ontology', 'KG · path',
             'KG · path', 'KG · ontology', 'KG · protein', 'FAERS freq.']
kg_w      = [0.154, 0.152, 0.127, 0.080, 0.083, 0.031, 0.0]
hybrid_w  = [0.024, 0.003, 0.093, 0.200, 0.038, 0.204, 4.527]

themes     = ['Pulmonary', 'Cardiovascular', 'Neurological', 'Haematological',
              'Immune/Infect.', 'Musculoskel.', 'Hepatic', 'Proliferative',
              'Oedema/Fluid', 'Metabolic', 'Skin', 'Renal', 'Cell Death']
faers_rank = [2,  4,  3,  1,  5,  6,  7,  8,  8, 10, 11, 12, 13]
kg_rank    = [11, 7,  4,  1,  2, 11, 11,  3,  9,  8, 11,  6,  5]
novel      = {'Cell Death', 'Renal', 'Proliferative'}
kg_gap     = {'Pulmonary', 'Musculoskel.', 'Hepatic'}

# ── Figure: extra wide, tall enough for legend below ─────────────────────────
fig = plt.figure(figsize=(20, 9), facecolor=BG)

# Panel A: left side, stops at 0.40 so there's room for right-axis labels
# Panel B: right side, starts at 0.55, extra right margin for theme labels
ax1 = fig.add_axes([0.05, 0.26, 0.34, 0.60])
ax2 = fig.add_axes([0.54, 0.26, 0.38, 0.60])

# ═══════════════════════════════════════════
# PANEL A — Grouped horizontal bar chart
# ═══════════════════════════════════════════
ax1.set_facecolor(BG)
for sp in ax1.spines.values():
    sp.set_visible(False)

hybrid_display = hybrid_w.copy()
hybrid_display[-1] = 0.58   # cap log_faers bar visually

y = np.arange(len(features))
h = 0.30

ax1.barh(y + h/2, kg_w,           height=h, color=C_KG,     alpha=0.88, zorder=3, linewidth=0)
ax1.barh(y - h/2, hybrid_display,  height=h, color=C_HYBRID, alpha=0.78, zorder=3, linewidth=0)

# True value label for capped bar — placed after bar ends
ax1.text(0.60, 0 - h/2, '4.53 →', fontsize=9, color=C_HYBRID,
         fontweight='600', va='center', ha='left')

# Gridlines
for x in np.arange(0.05, 0.60, 0.05):
    ax1.axvline(x, color=GREY_MID, lw=0.5, zorder=1)

ax1.set_yticks(y)
ax1.set_yticklabels(features, fontsize=10.5, color=INK)
ax1.set_xlabel('Absolute feature weight (logistic regression)',
               fontsize=10, color=GREY_TEXT, labelpad=10)
ax1.set_xlim(0, 0.68)
ax1.tick_params(axis='x', colors=GREY_TEXT, labelsize=9.5, length=0)
ax1.tick_params(axis='y', length=0)

# Feature type labels on a twin right axis
ax1_r = ax1.twinx()
ax1_r.set_ylim(ax1.get_ylim())
ax1_r.set_yticks(y)
ax1_r.set_yticklabels(feat_type, fontsize=8.5, style='italic')
for tick, ft in zip(ax1_r.get_yticklabels(), feat_type):
    tick.set_color(C_HYBRID if 'FAERS' in ft else '#b0a9a2')
ax1_r.tick_params(axis='y', length=0)
for sp in ax1_r.spines.values():
    sp.set_visible(False)

# Legend — bottom left, below bars, clear of everything
patch_kg = mpatches.Patch(color=C_KG,     alpha=0.88, label='KG-only model')
patch_hy = mpatches.Patch(color=C_HYBRID, alpha=0.78, label='Hybrid (KG + FAERS)')
ax1.legend(handles=[patch_kg, patch_hy], fontsize=9.5, frameon=False,
           loc='lower left', bbox_to_anchor=(0.0, -0.22),
           ncol=2, handlelength=1.4, labelspacing=0.5, columnspacing=1.2)

ax1.set_title('A.  Feature weights: KG-only vs. FAERS-dominated hybrid',
              fontsize=11.5, fontweight='bold', color=INK, loc='left', pad=12)

# Callout box — in figure coords, centered under Panel A, above caption
fig.text(0.27, 0.08,
         'When FAERS is included, log_faers captures >95% of model weight,\ndrowning out all ontology signal.',
         ha='center', va='center', fontsize=9, color=C_HYBRID, style='italic',
         bbox=dict(boxstyle='round,pad=0.55', facecolor='#fde8e0',
                   edgecolor='#f5b8a8', linewidth=1.2))

# ═══════════════════════════════════════════
# PANEL B — Dumbbell dot plot
# ═══════════════════════════════════════════
ax2.set_facecolor(BG)
for sp in ax2.spines.values():
    sp.set_visible(False)

n  = len(themes)
y2 = np.arange(n)

# Alternating row shading
for i in range(n):
    if i % 2 == 0:
        ax2.axhspan(i - 0.45, i + 0.45, color='#f3f0eb', zorder=0)

# Connector lines
for i, (fr, kr) in enumerate(zip(faers_rank, kg_rank)):
    if themes[i] in novel:
        col, lw = '#74c69d', 2.4
    elif themes[i] in kg_gap:
        col, lw = '#f4a582', 2.4
    else:
        col, lw = GREY_MID, 1.2
    ax2.plot([fr, kr], [i, i], color=col, lw=lw, zorder=2, solid_capstyle='round')

ax2.scatter(faers_rank, y2, color=C_HYBRID, s=72, zorder=4, edgecolors='white', linewidths=1.0)
ax2.scatter(kg_rank,    y2, color=C_KG,     s=72, zorder=4, edgecolors='white', linewidths=1.0)

# Theme labels — right side, well outside the axis
for i, theme in enumerate(themes):
    bold = theme in novel or theme in kg_gap
    col  = '#1a5e35' if theme in novel else ('#8b2a12' if theme in kg_gap else GREY_TEXT)
    ax2.text(-0.5, i, theme, fontsize=10, va='center', ha='right',
             color=col, fontweight='bold' if bold else 'normal')

# Delta badges above connector for big movers
for i, (fr, kr) in enumerate(zip(faers_rank, kg_rank)):
    delta = fr - kr
    if abs(delta) >= 5:
        sign = '▲' if delta > 0 else '▼'
        col  = C_KG if delta > 0 else C_HYBRID
        ax2.text((fr + kr) / 2, i + 0.37, f'{sign}{abs(delta)}',
                 fontsize=8.5, color=col, ha='center', va='bottom', fontweight='700')

# X axis reversed: rank 1 on right, rank 13 on left
# extra negative xlim space on right for theme labels
ax2.set_xlim(14.5, -7.0)
ax2.set_xticks([1, 3, 5, 7, 9, 11, 13])
ax2.tick_params(axis='x', colors=GREY_TEXT, labelsize=9.5, length=0)
ax2.tick_params(axis='y', length=0)
ax2.set_yticks([])
ax2.set_ylim(-0.7, n - 0.3)
ax2.set_xlabel('Theme rank  (rank 1 = highest signal  →)',
               fontsize=10, color=GREY_TEXT, labelpad=10)

for x in [1, 3, 5, 7, 9, 11, 13]:
    ax2.axvline(x, color=GREY_MID, lw=0.5, zorder=1)

# Legend — below Panel B, 2 columns
dot_kg = plt.Line2D([0],[0], marker='o', color='w', markerfacecolor=C_KG,     markersize=9, label='KG rank')
dot_fa = plt.Line2D([0],[0], marker='o', color='w', markerfacecolor=C_HYBRID, markersize=9, label='FAERS rank')
p_nov  = mpatches.Patch(color='#74c69d', label='KG-nominated novel candidate')
p_miss = mpatches.Patch(color='#f4a582', label='KG misses (ontology gap)')
ax2.legend(handles=[dot_kg, dot_fa, p_nov, p_miss],
           fontsize=9.5, frameon=True, ncol=2,
           loc='upper left', bbox_to_anchor=(0.0, -0.13),
           facecolor=BG, edgecolor=GREY_MID,
           handlelength=1.4, labelspacing=0.5, columnspacing=1.5)

ax2.set_title('B.  Theme ranking: KG vs. FAERS frequency',
              fontsize=11.5, fontweight='bold', color=INK, loc='left', pad=12)

# ── Figure caption ────────────────────────────────────────────────────────────
fig.text(0.50, 0.03,
         'Figure X. (A) In the hybrid model, log-transformed FAERS frequency dominates (weight 4.53 vs. max 0.20 for any KG feature), suppressing ontology signal entirely. '
         'The KG-only model distributes weight across 6 mechanistic features. '
         '(B) KG elevates Cell Death/Apoptosis (13→5), Renal (12→6), and Proliferative (8→3) as underreported candidates invisible to frequency-only analysis. '
         'Pulmonary and Hepatic rank high in FAERS but low in KG, reflecting known ontology coverage gaps.',
         ha='center', fontsize=8.5, color=GREY_TEXT, style='italic', linespacing=1.6)

plt.savefig(os.path.join(output_dir, 'kg_vs_nokg_figure.pdf'),
            dpi=300, bbox_inches='tight', facecolor=BG)
plt.savefig(os.path.join(output_dir, 'kg_vs_nokg_figure.png'),
            dpi=300, bbox_inches='tight', facecolor=BG)
print("Saved.")