"""
Publication-ready Table 1: Sample characteristics by UPF energy quartile.
Uses matplotlib only (no seaborn), OMP-safe.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from scipy import stats
import warnings, os
warnings.filterwarnings('ignore')

OUT = '/home/sandbox/outputs/'
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv('/home/sandbox/nhanes_analysis/nhanes_merged.csv')

# ── Derived variables ──────────────────────────────────────────
df['sex_male']   = (df['RIAGENDR'] == 1).astype(float)
df['educ_high']  = (df['DMDEDUC2'] >= 4).astype(float)
df['race_nhw']   = (df['RIDRETH3'] == 3).astype(float)
df['race_nhb']   = (df['RIDRETH3'] == 4).astype(float)
df['race_hisp']  = df['RIDRETH3'].isin([1,2]).astype(float)
df['race_asian'] = (df['RIDRETH3'] == 6).astype(float)
df['race_other'] = df['RIDRETH3'].isin([5,7]).astype(float)

# UPF quartile
df['upf_q'] = pd.qcut(df['upf_pct'], 4, labels=['Q1 (Lowest)','Q2','Q3','Q4 (Highest)'])

# ── Stats helper ───────────────────────────────────────────────
def fmt_mean_sd(col, mask=None):
    s = df[col] if mask is None else df.loc[mask, col]
    s = s.dropna()
    return f"{s.mean():.1f} ± {s.std():.1f}"

def fmt_pct(col, mask=None):
    s = df[col] if mask is None else df.loc[mask, col]
    s = s.dropna()
    return f"{100*s.mean():.1f}%"

def anova_p(col):
    groups = [df.loc[df['upf_q']==q, col].dropna().values for q in df['upf_q'].cat.categories]
    F, p = stats.f_oneway(*groups)
    return p

def chi2_p(col):
    ct = pd.crosstab(df['upf_q'], df[col])
    chi2, p, _, _ = stats.chi2_contingency(ct)
    return p

def fmt_p(p):
    if p < 0.001: return '<0.001'
    elif p < 0.01: return f'{p:.3f}'
    else: return f'{p:.3f}'

LABELS = ['Q1 (Lowest)', 'Q2', 'Q3', 'Q4 (Highest)']
MASKS  = [df['upf_q'] == q for q in LABELS]

# ── Build table rows ───────────────────────────────────────────
rows = []

# n
rows.append(('n', 'count',
    str(len(df)),
    *[str(m.sum()) for m in MASKS],
    ''))

# Age
rows.append(('Age (years)', 'mean±SD',
    fmt_mean_sd('RIDAGEYR'),
    *[fmt_mean_sd('RIDAGEYR', m) for m in MASKS],
    fmt_p(anova_p('RIDAGEYR'))))

# Sex female
rows.append(('Female sex', '% (n)',
    f"{fmt_pct('sex_male').replace('%','')[:4]}% → {fmt_pct('sex_male')}M, "
    f"{100-float(fmt_pct('sex_male').rstrip('%')):.1f}% F",
    *[f"{100-float(fmt_pct('sex_male',m).rstrip('%')):.1f}%" for m in MASKS],
    fmt_p(chi2_p('RIAGENDR'))))

# Cleaner sex row
pct_f = 100 - 100*df['sex_male'].mean()
rows[-1] = ('Female sex', '%',
    f"{pct_f:.1f}%",
    *[f"{100-100*df.loc[m,'sex_male'].mean():.1f}%" for m in MASKS],
    fmt_p(chi2_p('RIAGENDR')))

# BMI
rows.append(('BMI (kg/m²)', 'mean±SD',
    fmt_mean_sd('BMXBMI'),
    *[fmt_mean_sd('BMXBMI', m) for m in MASKS],
    fmt_p(anova_p('BMXBMI'))))

# Income/poverty
rows.append(('Income/Poverty Ratio', 'mean±SD',
    fmt_mean_sd('INDFMPIR'),
    *[fmt_mean_sd('INDFMPIR', m) for m in MASKS],
    fmt_p(anova_p('INDFMPIR'))))

# Education high
rows.append(('Higher Education (≥college)', '%',
    fmt_pct('educ_high'),
    *[fmt_pct('educ_high', m) for m in MASKS],
    fmt_p(chi2_p('educ_high'))))

# Total energy
rows.append(('Total Energy (kcal/d)', 'mean±SD',
    fmt_mean_sd('DR1TKCAL'),
    *[fmt_mean_sd('DR1TKCAL', m) for m in MASKS],
    fmt_p(anova_p('DR1TKCAL'))))

# UPF %
rows.append(('UPF Energy (%)', 'mean±SD',
    fmt_mean_sd('upf_pct'),
    *[fmt_mean_sd('upf_pct', m) for m in MASKS],
    '<0.001'))

# Race/ethnicity
rows.append(('Race/Ethnicity', '', '', '', '', '', ''))
rows.append(('  Non-Hispanic White', '%',
    fmt_pct('race_nhw'),
    *[fmt_pct('race_nhw', m) for m in MASKS],
    fmt_p(chi2_p('race_nhw'))))
rows.append(('  Non-Hispanic Black', '%',
    fmt_pct('race_nhb'),
    *[fmt_pct('race_nhb', m) for m in MASKS],
    fmt_p(chi2_p('race_nhb'))))
rows.append(('  Hispanic', '%',
    fmt_pct('race_hisp'),
    *[fmt_pct('race_hisp', m) for m in MASKS],
    fmt_p(chi2_p('race_hisp'))))
rows.append(('  Non-Hispanic Asian', '%',
    fmt_pct('race_asian'),
    *[fmt_pct('race_asian', m) for m in MASKS],
    fmt_p(chi2_p('race_asian'))))

# DSST
rows.append(('DSST Score (symbols/90s)', 'mean±SD',
    fmt_mean_sd('CFDDS'),
    *[fmt_mean_sd('CFDDS', m) for m in MASKS],
    fmt_p(anova_p('CFDDS'))))

# Cognitive impairment
rows.append(('Cognitive Impairment', '%',
    fmt_pct('cog_impaired'),
    *[fmt_pct('cog_impaired', m) for m in MASKS],
    fmt_p(chi2_p('cog_impaired'))))

# ── Render as matplotlib table figure ─────────────────────────
COLS = ['Characteristic', 'Statistic',
        f'Total\n(n=2,780)', 'Q1 (Lowest)\n(n≈695)', 'Q2\n(n≈695)',
        'Q3\n(n≈695)', 'Q4 (Highest)\n(n≈695)', 'p-value']

COL_W = [2.9, 0.85, 1.1, 1.1, 1.0, 1.0, 1.2, 0.85]   # inches
ROW_H = 0.32                                              # inches per row
N_ROWS = len(rows) + 1                                   # +1 header
FIG_W  = sum(COL_W) + 0.3
FIG_H  = N_ROWS * ROW_H + 0.6

fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=220)
ax  = fig.add_axes([0, 0, 1, 1])
ax.axis('off')

# ── Background ────────────────────────────────────────────────
fig.patch.set_facecolor('white')

BLUE_HDR  = '#0a2e6e'
BLUE_SUB  = '#dce8f8'
STRIPE    = '#f5f8fd'
WHITE     = 'white'
TEXT_DARK = '#111111'
TEXT_HDR  = 'white'

def draw_cell(ax, x, y, w, h, text, facecolor, textcolor, fontsize=7.8,
              ha='center', va='center', bold=False, italic=False):
    rect = FancyBboxPatch((x, y), w, h,
                          boxstyle='square,pad=0',
                          linewidth=0.4, edgecolor='#aac0d8',
                          facecolor=facecolor)
    ax.add_patch(rect)
    style = {'fontsize': fontsize, 'color': textcolor,
             'ha': ha, 'va': va,
             'fontweight': 'bold' if bold else 'normal',
             'fontstyle': 'italic' if italic else 'normal',
             'wrap': True}
    ax.text(x + (0.08 if ha == 'left' else w/2),
            y + h/2, text, **style)

# Compute cumulative x positions (normalized 0..1)
total_w = sum(COL_W)
x_starts = []
cx = 0
for w in COL_W:
    x_starts.append(cx / total_w)
    cx += w
col_widths_norm = [w / total_w for w in COL_W]
row_h_norm = ROW_H / FIG_H

# Title
ax.text(0.5, 1 - 0.12/FIG_H,
        'Table 1. Analytic Sample Characteristics by UPF Energy Quartile',
        ha='center', va='top', fontsize=10.5, fontweight='bold', color=BLUE_HDR,
        transform=ax.transAxes)
ax.text(0.5, 1 - 0.30/FIG_H,
        'NHANES 2011–2014 · Community-Dwelling Adults ≥60 Years (n=2,780)',
        ha='center', va='top', fontsize=8.5, color='#444', transform=ax.transAxes)

# Header row y
y_top = 1 - 0.48/FIG_H
hdr_h = row_h_norm * 1.25

for ci, (label, xs, wn) in enumerate(zip(COLS, x_starts, col_widths_norm)):
    draw_cell(ax, xs, y_top - hdr_h, wn, hdr_h,
              label, BLUE_HDR, TEXT_HDR,
              fontsize=7.6, bold=True, ha='center')

# Data rows
for ri, row in enumerate(rows):
    y_row = y_top - hdr_h - (ri + 1) * row_h_norm
    bg = '#e8f0fb' if row[0].startswith('Race') and row[1]=='' \
         else (STRIPE if ri % 2 == 0 else WHITE)

    is_sub = row[0].startswith('  ')
    for ci, (val, xs, wn) in enumerate(zip(row, x_starts, col_widths_norm)):
        is_char = ci == 0
        fc = '#d0e2f8' if row[0] == 'Race/Ethnicity' and row[1] == '' and ci == 0 \
             else bg
        txt = val
        draw_cell(ax, xs, y_row, wn, row_h_norm,
                  txt, fc, TEXT_DARK,
                  fontsize=7.2 if is_sub else 7.8,
                  ha='left' if ci == 0 else 'center',
                  bold=(ci == 0 and not is_sub and row[1] != ''))

# Bottom footnote
ax.text(0.01, 0.022,
        'Values: mean ± SD for continuous variables; % for categorical variables.  '
        'p-values: ANOVA (continuous) or chi-square test (categorical) for quartile comparison.  '
        'UPF = ultra-processed food (NOVA Group 4).  DSST = Digit Symbol Substitution Test.',
        ha='left', va='bottom', fontsize=6.5, color='#555', transform=ax.transAxes,
        style='italic')

plt.savefig(OUT + 'table1_publication.png', dpi=220, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("✓ Table 1 saved → outputs/table1_publication.png")

# Also save as CSV for LaTeX table
df_t1 = pd.DataFrame(rows, columns=['Characteristic','Statistic',
                                     'Total','Q1','Q2','Q3','Q4','p_value'])
df_t1.to_csv(OUT + 'table1_data.csv', index=False)
print("✓ Table 1 data → outputs/table1_data.csv")
