"""
Step 2a: Stats + Figures — statsmodels-free version
Uses scipy + sklearn for all inference.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.utils import resample
import warnings, json, os
warnings.filterwarnings('ignore')

DATA_DIR = '/home/sandbox/nhanes_analysis/'
OUT_DIR  = '/home/sandbox/outputs/'
os.makedirs(OUT_DIR, exist_ok=True)

print("Step 2a: Stats + Figures 1 & 2")

df = pd.read_csv(DATA_DIR + 'nhanes_merged.csv')
print(f"Loaded {df.shape[0]:,} participants, {df.shape[1]} vars")

# ── Descriptive stats ─────────────────────────────────────────
rows = []
for q in ['Q1','Q2','Q3','Q4']:
    sub = df[df['upf_quartile']==q]
    rows.append({
        'Quartile': q, 'N': len(sub),
        'Age (mean±SD)': f"{sub['RIDAGEYR'].mean():.1f}±{sub['RIDAGEYR'].std():.1f}",
        'DSST (mean±SD)': f"{sub['CFDDS'].mean():.1f}±{sub['CFDDS'].std():.1f}",
        'BMI (mean±SD)': f"{sub['BMXBMI'].mean():.1f}±{sub['BMXBMI'].std():.1f}",
        'UPF% (mean±SD)': f"{sub['upf_pct'].mean():.1f}±{sub['upf_pct'].std():.1f}",
        'Cog. impaired (%)': f"{sub['cog_impaired'].mean()*100:.1f}%",
        '% Female': f"{(sub['RIAGENDR']==2).mean()*100:.1f}%",
    })
table1 = pd.DataFrame(rows)
table1.to_csv(OUT_DIR + 'table1_descriptives.csv', index=False)
print("\nTable 1:")
print(table1.to_string(index=False))

# ── ANOVA ─────────────────────────────────────────────────────
groups = [df[df['upf_quartile']==q]['CFDDS'].dropna().values
          for q in ['Q1','Q2','Q3','Q4']]
f_stat, p_anova = stats.f_oneway(*groups)
dsst_means = {q: df[df['upf_quartile']==q]['CFDDS'].mean()
              for q in ['Q1','Q2','Q3','Q4']}
print(f"\nANOVA: F={f_stat:.3f}, p={p_anova:.5f}")
print("DSST means:", {k: f"{v:.2f}" for k,v in dsst_means.items()})

# ── Prepare features ──────────────────────────────────────────
df['sex_male']   = (df['RIAGENDR'] == 1).astype(float)
df['educ_high']  = (df['DMDEDUC2'] >= 4).astype(float)
df['race_hisp']  = df['RIDRETH3'].isin([1,2]).astype(float)
df['race_black'] = (df['RIDRETH3'] == 4).astype(float)
df['race_asian'] = (df['RIDRETH3'] == 6).astype(float)
df['race_other'] = df['RIDRETH3'].isin([5,7]).astype(float)
df['upf_q2'] = (df['upf_quartile']=='Q2').astype(float)
df['upf_q3'] = (df['upf_quartile']=='Q3').astype(float)
df['upf_q4'] = (df['upf_quartile']=='Q4').astype(float)

feat_cols = ['upf_q2','upf_q3','upf_q4','RIDAGEYR','sex_male','BMXBMI',
             'INDFMPIR','educ_high','DR1TKCAL',
             'race_hisp','race_black','race_asian','race_other']

df_m = df[feat_cols + ['cog_impaired','upf_pct']].copy()
imp = SimpleImputer(strategy='median')
df_m[feat_cols] = imp.fit_transform(df_m[feat_cols])

X = df_m[feat_cols].values
y = df_m['cog_impaired'].values

# ── Logistic regression + Bootstrap CIs (B=200) ───────────────
print("\nFitting logistic regression with bootstrap CIs (B=200)...")
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X, y)
coef_main = lr.coef_[0]

# Bootstrap
np.random.seed(42)
B = 200
boot_coefs = []
for _ in range(B):
    idx = np.random.choice(len(X), len(X), replace=True)
    lr_b = LogisticRegression(max_iter=500, random_state=0)
    lr_b.fit(X[idx], y[idx])
    boot_coefs.append(lr_b.coef_[0])
boot_coefs = np.array(boot_coefs)

# OR + 95% CI from bootstrap percentiles
OR_vals = np.exp(coef_main)
OR_ci_l = np.exp(np.percentile(boot_coefs, 2.5, axis=0))
OR_ci_h = np.exp(np.percentile(boot_coefs, 97.5, axis=0))

# Wald p-value from bootstrap SE
boot_se = boot_coefs.std(axis=0)
z_scores = coef_main / (boot_se + 1e-10)
p_vals = 2 * (1 - stats.norm.cdf(np.abs(z_scores)))

OR_df = pd.DataFrame({
    'Feature': feat_cols,
    'OR': OR_vals,
    'CI_lower': OR_ci_l,
    'CI_upper': OR_ci_h,
    'p': p_vals,
}).set_index('Feature')

print("\nOR table (UPF quartiles):")
print(OR_df.loc[['upf_q2','upf_q3','upf_q4']].round(3))

or_q4 = float(OR_df.loc['upf_q4','OR'])
ci_q4 = (float(OR_df.loc['upf_q4','CI_lower']), float(OR_df.loc['upf_q4','CI_upper']))
p_q4  = float(OR_df.loc['upf_q4','p'])

OR_df.loc[['upf_q2','upf_q3','upf_q4']].to_csv(OUT_DIR+'logit_OR_upf_quartile.csv')

# ── Trend test (UPF% as continuous) ───────────────────────────
feat_trend = ['upf_pct'] + feat_cols[3:]
df_m2 = df_m[feat_trend + ['cog_impaired']].copy()
df_m2[feat_trend] = imp.fit_transform(df_m2[feat_trend])
lr_t = LogisticRegression(max_iter=1000, random_state=42)
lr_t.fit(df_m2[feat_trend].values, y)
# Trend: OR per 10% increase in UPF
upf_coef = lr_t.coef_[0][0]  # first feature = upf_pct
trend_or10 = float(np.exp(upf_coef * 10))
# Bootstrap SE for trend
boot_t = []
for _ in range(B):
    idx = np.random.choice(len(y), len(y), replace=True)
    lr_bt = LogisticRegression(max_iter=300, random_state=0)
    lr_bt.fit(df_m2[feat_trend].values[idx], y[idx])
    boot_t.append(lr_bt.coef_[0][0])
trend_se = np.std(boot_t)
trend_z  = upf_coef / (trend_se + 1e-10)
trend_p  = float(2 * (1 - stats.norm.cdf(abs(trend_z))))
print(f"\nTrend OR per 10% UPF = {trend_or10:.3f}, p = {trend_p:.4f}")

# ── Figure 1 ─────────────────────────────────────────────────
print("\nFigure 1...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Figure 1. Ultra-Processed Food Consumption and Cognitive Performance\n'
             '(NHANES 2011–2014, Community-Dwelling Adults ≥60 Years)',
             fontsize=11, fontweight='bold', y=1.01)

ax = axes[0]
n_bins = 40
ax.hist(df['upf_pct'], bins=n_bins, color='#2166ac', alpha=0.82, edgecolor='white', linewidth=0.4)
ax.axvline(df['upf_pct'].median(), color='#d6604d', lw=2.2, ls='--',
           label=f"Median = {df['upf_pct'].median():.1f}%")
ax.axvline(df['upf_pct'].mean(), color='#f4a582', lw=1.5, ls=':',
           label=f"Mean = {df['upf_pct'].mean():.1f}%")
ax.set_xlabel('UPF Energy Contribution (%)', fontsize=11)
ax.set_ylabel('Number of Participants', fontsize=11)
ax.set_title('(A) Distribution of UPF Energy %', fontsize=10)
ax.legend(fontsize=9)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

ax2 = axes[1]
pal = {'Q1':'#4dac26','Q2':'#b8e186','Q3':'#f1a340','Q4':'#d7191c'}
sns.boxplot(data=df, x='upf_quartile', y='CFDDS', palette=pal,
            order=['Q1','Q2','Q3','Q4'], width=0.55, linewidth=1.3, ax=ax2,
            fliersize=3, flierprops={'alpha':0.4})
ax2.set_xlabel('UPF Energy Quartile (Q1 = Lowest Consumption)', fontsize=11)
ax2.set_ylabel('DSST Score (Digit Symbol Substitution)', fontsize=11)
ax2.set_title(f'(B) Cognitive Performance by UPF Quartile\nANOVA: F={f_stat:.2f}, p={p_anova:.4f}',
              fontsize=10)
for i, q in enumerate(['Q1','Q2','Q3','Q4']):
    m = dsst_means[q]
    ax2.text(i, m+0.4, f'{m:.1f}', ha='center', fontsize=9,
             fontweight='bold', color='#333333')
ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)

plt.tight_layout()
fig.savefig(OUT_DIR+'fig1_upf_distribution_dsst.png', dpi=200, bbox_inches='tight')
plt.close()
print("  Saved fig1")

# ── Figure 2 (Forest plot) ────────────────────────────────────
print("Figure 2...")
fig, ax = plt.subplots(figsize=(8, 4.5))
labels = ['Q2 vs Q1', 'Q3 vs Q1', 'Q4 vs Q1']
keys   = ['upf_q2', 'upf_q3', 'upf_q4']
ors  = [OR_df.loc[k,'OR']       for k in keys]
ci_l = [OR_df.loc[k,'CI_lower'] for k in keys]
ci_h = [OR_df.loc[k,'CI_upper'] for k in keys]
pv   = [OR_df.loc[k,'p']        for k in keys]
colors_f = ['#4dac26', '#f1a340', '#d7191c']

for i,(lbl,o,l,h,c,pval) in enumerate(zip(labels,ors,ci_l,ci_h,colors_f,pv)):
    y_pos = 2 - i
    ax.errorbar(o, y_pos, xerr=[[o-l],[h-o]], fmt='D', color=c,
                markersize=9, elinewidth=2.5, capsize=6, capthick=2)
    pstr = f"p={pval:.3f}" if pval >= 0.001 else "p<0.001"
    ax.text(h+0.05, y_pos, f"OR={o:.2f} ({l:.2f}–{h:.2f})\n{pstr}",
            va='center', fontsize=9, color='#333333')

ax.axvline(1.0, color='black', ls='--', lw=1.5, alpha=0.6, label='Null (OR=1)')
ax.set_yticks([2,1,0]); ax.set_yticklabels(labels, fontsize=11)
ax.set_xlabel('Adjusted Odds Ratio (95% Bootstrap CI)', fontsize=11)
ax.set_title('Figure 2. Adjusted Odds Ratios for Cognitive Impairment\nby UPF Energy Quartile '
             '(NHANES 2011–2014; ref: Q1)', fontsize=10, fontweight='bold')
ax.set_xlim(0.2, max(ci_h)+1.0); ax.grid(axis='x', alpha=0.3)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.legend(fontsize=9, loc='lower right')

plt.tight_layout()
fig.savefig(OUT_DIR+'fig2_forest_plot_OR.png', dpi=200, bbox_inches='tight')
plt.close()
print("  Saved fig2")

# ── Save partial results ──────────────────────────────────────
res = {
    'n_total': int(len(df)),
    'n_impaired': int(df['cog_impaired'].sum()),
    'pct_impaired': float(df['cog_impaired'].mean()*100),
    'upf_mean_pct': float(df['upf_pct'].mean()),
    'upf_median_pct': float(df['upf_pct'].median()),
    'upf_sd_pct': float(df['upf_pct'].std()),
    'dsst_overall_mean': float(df['CFDDS'].mean()),
    'dsst_overall_sd': float(df['CFDDS'].std()),
    'dsst_Q1_mean': float(dsst_means['Q1']),
    'dsst_Q4_mean': float(dsst_means['Q4']),
    'anova_F': float(f_stat),
    'anova_p': float(p_anova),
    'OR_Q2_vs_Q1': float(OR_df.loc['upf_q2','OR']),
    'OR_Q3_vs_Q1': float(OR_df.loc['upf_q3','OR']),
    'OR_Q4_vs_Q1': float(or_q4),
    'OR_Q2_CI': [float(OR_df.loc['upf_q2','CI_lower']), float(OR_df.loc['upf_q2','CI_upper'])],
    'OR_Q3_CI': [float(OR_df.loc['upf_q3','CI_lower']), float(OR_df.loc['upf_q3','CI_upper'])],
    'OR_Q4_CI': [float(ci_q4[0]), float(ci_q4[1])],
    'OR_Q4_p': float(p_q4),
    'trend_OR_per10pct': float(trend_or10),
    'trend_p': float(trend_p),
}
with open(OUT_DIR+'results_partial.json','w') as f:
    json.dump(res, f, indent=2)

print("\n=== RESULTS ===")
for k,v in res.items():
    print(f"  {k}: {v}")
print("\n✓ Step 2a done")
