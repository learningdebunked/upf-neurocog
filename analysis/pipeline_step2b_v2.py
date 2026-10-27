"""
Step 2b (corrected): Linear regression + XGBoost + Native SHAP + Figures 3-6
Uses XGBoost's built-in pred_contribs=True for SHAP (bypasses shap library version conflict).
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, roc_curve, r2_score
import xgboost as xgb
import warnings, json, os
warnings.filterwarnings('ignore')

DATA_DIR = '/home/sandbox/nhanes_analysis/'
OUT_DIR  = '/home/sandbox/outputs/'
os.makedirs(OUT_DIR, exist_ok=True)

print("Step 2b: Linear reg + XGBoost + Native SHAP + Figures 3-6")

df = pd.read_csv(DATA_DIR + 'nhanes_merged.csv')
with open(OUT_DIR + 'results_partial.json') as f:
    res = json.load(f)
print(f"Loaded {df.shape[0]:,} participants")

FEATURES = ['upf_pct', 'RIDAGEYR', 'sex_male', 'BMXBMI',
            'INDFMPIR', 'educ_high', 'DR1TKCAL',
            'race_hisp', 'race_black', 'race_asian', 'race_other']
FEAT_LABELS = {
    'upf_pct':     'UPF Energy (%)',
    'RIDAGEYR':    'Age (years)',
    'sex_male':    'Sex (male=1)',
    'BMXBMI':      'BMI (kg/m²)',
    'INDFMPIR':    'Income/Poverty Ratio',
    'educ_high':   'Higher Education',
    'DR1TKCAL':    'Total Energy (kcal/d)',
    'race_hisp':   'Hispanic',
    'race_black':  'Non-Hisp. Black',
    'race_asian':  'Non-Hisp. Asian',
    'race_other':  'Other Race/Ethnicity',
}

df['sex_male']   = (df['RIAGENDR'] == 1).astype(float)
df['educ_high']  = (df['DMDEDUC2'] >= 4).astype(float)
df['race_hisp']  = df['RIDRETH3'].isin([1,2]).astype(float)
df['race_black'] = (df['RIDRETH3'] == 4).astype(float)
df['race_asian'] = (df['RIDRETH3'] == 6).astype(float)
df['race_other'] = df['RIDRETH3'].isin([5,7]).astype(float)

df_m = df[FEATURES + ['cog_impaired', 'CFDDS']].copy()
imp = SimpleImputer(strategy='median')
df_m[FEATURES] = imp.fit_transform(df_m[FEATURES])

X   = df_m[FEATURES].values
y_b = df_m['cog_impaired'].values.astype(int)
y_c = df_m['CFDDS'].values

# ── 1. Linear regression (DSST continuous) ────────────────────
print("\n[1] Linear regression DSST ~ UPF + covariates (bootstrap n=300)...")
lm = LinearRegression().fit(X, y_c)
np.random.seed(42)
B = 300
boot_lm = np.array([LinearRegression().fit(X[i:=np.random.choice(len(X), len(X), True)],
                                             y_c[i]).coef_
                    for _ in range(B)])
lm_p  = 2*(1-stats.norm.cdf(np.abs(lm.coef_ / (boot_lm.std(axis=0)+1e-10))))
lm_ci_l = np.percentile(boot_lm, 2.5,  axis=0)
lm_ci_h = np.percentile(boot_lm, 97.5, axis=0)

beta_upf  = float(lm.coef_[0])
p_upf_lm  = float(lm_p[0])
ci_upf_lm = (float(lm_ci_l[0]), float(lm_ci_h[0]))
r2 = float(r2_score(y_c, lm.predict(X)))
print(f"  UPF β={beta_upf:.4f}/1% (per 10%: {beta_upf*10:.2f} pts), "
      f"p={p_upf_lm:.4f}, R²={r2:.4f}")

# ── 2. Logistic regression (weak reg) ─────────────────────────
print("\n[2] Logistic regression (C=1000, bootstrap)...")
lr = LogisticRegression(C=1000, max_iter=2000, random_state=42)
lr.fit(X, y_b)
boot_lr = np.array([LogisticRegression(C=1000, max_iter=500, random_state=0)
                    .fit(X[i:=np.random.choice(len(X), len(X), True)], y_b[i]).coef_[0]
                    for _ in range(B)])
lr_p   = 2*(1-stats.norm.cdf(np.abs(lr.coef_[0] / (boot_lr.std(axis=0)+1e-10))))
lr_ci_l = np.percentile(boot_lr, 2.5,  axis=0)
lr_ci_h = np.percentile(boot_lr, 97.5, axis=0)
upf_idx = FEATURES.index('upf_pct')
or_upf10 = float(np.exp(lr.coef_[0][upf_idx] * 10))
p_or_upf = float(lr_p[upf_idx])
ci_or_upf = (float(np.exp(lr_ci_l[upf_idx]*10)), float(np.exp(lr_ci_h[upf_idx]*10)))
print(f"  UPF OR per 10% = {or_upf10:.3f} (95% CI {ci_or_upf[0]:.3f}–{ci_or_upf[1]:.3f}), "
      f"p={p_or_upf:.4f}")

# ── 3. XGBoost (5-fold CV) ─────────────────────────────────────
print("\n[3] XGBoost (5-fold CV)...")
neg, pos = (y_b==0).sum(), (y_b==1).sum()
clf = xgb.XGBClassifier(n_estimators=150, max_depth=3, learning_rate=0.1,
                         subsample=0.8, colsample_bytree=0.8,
                         scale_pos_weight=neg/pos, eval_metric='logloss',
                         random_state=42, n_jobs=1, tree_method='hist')
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_prob = cross_val_predict(clf, X, y_b, cv=skf, method='predict_proba')[:, 1]
auc = float(roc_auc_score(y_b, y_prob))
fpr, tpr, _ = roc_curve(y_b, y_prob)
clf.fit(X, y_b)
print(f"  5-fold CV AUC = {auc:.4f}")

# ── 4. Native XGBoost SHAP ─────────────────────────────────────
print("\n[4] Computing native XGBoost SHAP values...")
X_f32 = X.astype(np.float32)
dmat  = xgb.DMatrix(X_f32, feature_names=FEATURES)
sv_raw = clf.get_booster().predict(dmat, pred_contribs=True)
shap_vals = sv_raw[:, :-1]   # drop bias column, shape (n, 11)
mean_abs  = np.abs(shap_vals).mean(axis=0)

feat_labels_arr = [FEAT_LABELS[f] for f in FEATURES]
shap_rank = pd.DataFrame({'Feature': feat_labels_arr, 'Mean|SHAP|': mean_abs}
                          ).sort_values('Mean|SHAP|', ascending=False)
print(shap_rank.to_string(index=False))

# ── Figure 3: ROC curve ────────────────────────────────────────
print("\n[5] Figure 3: ROC curve...")
fig, ax = plt.subplots(figsize=(6.5, 6.5))
ax.plot(fpr, tpr, color='#d7191c', lw=2.5, label=f'XGBoost (AUC = {auc:.3f})')
ax.plot([0,1],[0,1],'k--',lw=1.5,alpha=0.55,label='Random (AUC=0.500)')
ax.fill_between(fpr, tpr, alpha=0.07, color='#d7191c')
ax.set_xlabel('False Positive Rate (1 − Specificity)', fontsize=12)
ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12)
ax.set_title('Figure 3. Receiver Operating Characteristic (ROC) Curve\n'
             'XGBoost — Cognitive Impairment Classification\n'
             'NHANES 2011–2014, Adults ≥60 (5-fold Cross-Validation)',
             fontsize=10, fontweight='bold')
ax.legend(fontsize=11, loc='lower right')
ax.grid(alpha=0.22); ax.set_xlim([0,1]); ax.set_ylim([0,1.01])
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
fig.savefig(OUT_DIR+'fig3_roc_curve.png', dpi=200, bbox_inches='tight')
plt.close(); print("  Saved fig3")

# ── Figure 4: SHAP beeswarm (manual) ──────────────────────────
print("\n[6] Figure 4: SHAP beeswarm...")
from matplotlib.cm import RdYlBu_r
from matplotlib.colors import Normalize

n_show = 11
order = shap_rank.index[:n_show].tolist()  # top features
shap_top = shap_vals[:, order]
feat_top  = [feat_labels_arr[i] for i in order]
X_top     = X[:, order]

fig, ax = plt.subplots(figsize=(9, 7))

norm = Normalize(vmin=0, vmax=1)
cmap_used = RdYlBu_r

for row_i, (feat_i, feat_name) in enumerate(zip(range(n_show), reversed(feat_top))):
    col_i = n_show - 1 - row_i  # map reversed display index to data col
    sv_col = shap_top[:, col_i]
    X_col  = X_top[:, col_i]

    # Normalize X values 0..1 for color
    xmin, xmax = X_col.min(), X_col.max()
    if xmax > xmin:
        x_norm = (X_col - xmin) / (xmax - xmin)
    else:
        x_norm = np.full_like(X_col, 0.5)

    # Jitter y position for beeswarm effect
    jitter = np.random.uniform(-0.25, 0.25, len(sv_col))
    y_pos  = np.full(len(sv_col), row_i) + jitter

    colors = cmap_used(norm(x_norm))
    ax.scatter(sv_col, y_pos, c=colors, alpha=0.3, s=6, linewidths=0)

ax.set_yticks(range(n_show))
ax.set_yticklabels(reversed(feat_top), fontsize=10)
ax.axvline(0, color='black', lw=1, alpha=0.4)
ax.set_xlabel('SHAP Value (Impact on Log-Odds of Cognitive Impairment)', fontsize=11)
ax.set_title('Figure 4. SHAP Summary — Feature Impact on XGBoost Predictions\n'
             'Red = High Feature Value, Blue = Low Feature Value',
             fontsize=10, fontweight='bold')
sm = plt.cm.ScalarMappable(cmap=cmap_used, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
cbar.set_label('Feature Value (scaled)', fontsize=9)
ax.grid(axis='x', alpha=0.25)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

plt.tight_layout()
fig.savefig(OUT_DIR+'fig4_shap_summary.png', dpi=200, bbox_inches='tight')
plt.close(); print("  Saved fig4")

# ── Figure 5: SHAP dependence plot ────────────────────────────
print("\n[7] Figure 5: SHAP dependence (UPF%)...")
upf_sv   = shap_vals[:, upf_idx]
age_vals = X[:, FEATURES.index('RIDAGEYR')]

fig, ax = plt.subplots(figsize=(8, 5))
sc = ax.scatter(X[:, upf_idx], upf_sv, c=age_vals, cmap='RdYlBu_r',
                alpha=0.4, s=11, edgecolors='none')
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label('Age (years)', fontsize=10)

# Smoothed trend
sort_i  = np.argsort(X[:, upf_idx])
xs = X[sort_i, upf_idx]; ys = upf_sv[sort_i]
win = max(7, len(xs)//50)
ys_smooth = pd.Series(ys).rolling(win, center=True, min_periods=1).mean().values
ax.plot(xs, ys_smooth, color='#d7191c', lw=2.5, alpha=0.9, label='Smoothed trend')

ax.axhline(0, color='k', ls='--', lw=1, alpha=0.4)
ax.set_xlabel('UPF Energy Contribution (%)', fontsize=12)
ax.set_ylabel('SHAP Value\n(↑ = Higher predicted probability of cognitive impairment)', fontsize=11)
ax.set_title('Figure 5. SHAP Dependence Plot: Effect of UPF Energy %\n'
             'on Cognitive Impairment Prediction (color = age)',
             fontsize=10, fontweight='bold')
ax.legend(fontsize=9); ax.grid(alpha=0.22)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
fig.savefig(OUT_DIR+'fig5_shap_dependence_upf.png', dpi=200, bbox_inches='tight')
plt.close(); print("  Saved fig5")

# ── Figure 6: SHAP bar importance ─────────────────────────────
print("\n[8] Figure 6: SHAP bar importance...")
shap_s = shap_rank.sort_values('Mean|SHAP|', ascending=True)
colors_b = ['#d7191c' if 'UPF' in f else '#4393c3' for f in shap_s['Feature']]

fig, ax = plt.subplots(figsize=(9, 6))
bars = ax.barh(shap_s['Feature'], shap_s['Mean|SHAP|'],
               color=colors_b, alpha=0.85, edgecolor='white', linewidth=0.4)
ax.set_xlabel('Mean |SHAP Value| (Average Absolute Impact on Prediction)', fontsize=11)
ax.set_title('Figure 6. Global Feature Importance via SHAP\n'
             'XGBoost Cognitive Impairment Model, NHANES 2011–2014\n'
             '(Red bar = UPF Energy %; blue = other covariates)',
             fontsize=10, fontweight='bold')
ax.grid(axis='x', alpha=0.3)
for bar, feat in zip(bars, shap_s['Feature']):
    if 'UPF' in feat:
        bar.set_edgecolor('#d7191c'); bar.set_linewidth(2.5)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
fig.savefig(OUT_DIR+'fig6_shap_bar_importance.png', dpi=200, bbox_inches='tight')
plt.close(); print("  Saved fig6")

# ── Save full results ──────────────────────────────────────────
upf_shap_pos = int(shap_rank[shap_rank['Feature']=='UPF Energy (%)'].index[0]) + 1 \
               if not shap_rank[shap_rank['Feature']=='UPF Energy (%)'].empty else 'N/A'

res.update({
    'beta_upf_per1pct': beta_upf,
    'beta_upf_per10pct': float(beta_upf * 10),
    'beta_upf_p': p_upf_lm,
    'beta_upf_CI': list(ci_upf_lm),
    'lm_r2': r2,
    'or_upf_per10pct': or_upf10,
    'or_upf_CI_per10': list(ci_or_upf),
    'or_upf_p': p_or_upf,
    'xgb_cv_auc': auc,
    'top_shap_feature': shap_rank.iloc[0]['Feature'],
    'upf_shap_rank': upf_shap_pos,
    'mean_abs_shap_upf': float(mean_abs[upf_idx]),
})
with open(OUT_DIR+'results_summary.json','w') as f:
    json.dump(res, f, indent=2)

print("\n" + "="*65)
print("FINAL RESULTS SUMMARY")
print("="*65)
print(f"  Sample:  n={res['n_total']:,} adults ≥60 (NHANES 2011–2014)")
print(f"  Impairment: {res['pct_impaired']:.1f}% (n={res['n_impaired']:,})")
print(f"  UPF:     mean {res['upf_mean_pct']:.1f}% ± {res['upf_sd_pct']:.1f}%")
print(f"  DSST:    Q1={res['dsst_Q1_mean']:.1f}, Q4={res['dsst_Q4_mean']:.1f} pts")
print(f"  ANOVA:   F={res['anova_F']:.2f}, p={res['anova_p']:.5f}")
print(f"  Linear β={beta_upf:.4f}/% UPF → per 10% = {beta_upf*10:.2f} DSST pts, p={p_upf_lm:.4f}")
print(f"  Logit:   OR per 10% UPF = {or_upf10:.3f} ({ci_or_upf[0]:.3f}–{ci_or_upf[1]:.3f}), p={p_or_upf:.4f}")
print(f"  XGBoost AUC = {auc:.3f} (5-fold CV)")
print(f"  Top SHAP feature: {shap_rank.iloc[0]['Feature']}")
print(f"  UPF SHAP rank: #{upf_shap_pos} of {len(FEATURES)}")
print("="*65)
print("\n✓ All 6 figures saved to /home/sandbox/outputs/")
