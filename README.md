# Ultra-Processed Food Consumption as a Behavioral Predictor of EEG-Indexed Cognitive Fatigue

> An AI framework for workforce mental health surveillance using NHANES 2011–2014

---

## Abstract

This repository contains the full manuscript, analysis pipeline, and results for a study investigating whether ultra-processed food (UPF) consumption predicts cognitive fatigue in older U.S. adults, using a multi-modal AI framework combining XGBoost, LSTM, attention mechanisms, and SHAP interpretability. Empirical validation is performed on NHANES 2011–2014 data (n = 2,780 adults ≥ 60 years).

**Target journal:** IEEE Journal of Biomedical and Health Informatics (JBHI)

---

## Key Results

| Metric | Value |
|--------|-------|
| Sample size | n = 2,780 (adults ≥60, NHANES 2011–2014) |
| Cognitive impairment rate | 16.4% (n = 457) |
| UPF mean energy share | 8.1% ± 9.6% (median 4.7%) |
| DSST ANOVA across UPF quartiles | F = 2.87, p = 0.035 |
| XGBoost 5-fold CV AUC | 0.812 |
| UPF SHAP importance rank | 8 of 11 (mean |SHAP| = 0.237) |
| Top predictor | Higher Education (mean |SHAP| = 0.826) |

---

## Repository Structure

```
upf-neurocog/
├── manuscript/
│   ├── manuscript_upf_eeg_jbhi_ieee.pdf       ← Final IEEE JBHI submission (28 pages)
│   ├── manuscript_upf_eeg_jbhi_ieee.tex        ← LaTeX source (IEEEtran, double-column)
│   └── manuscript_upf_eeg_cognitive_fatigue_ai_workforce.md  ← Full Markdown draft
├── figures/
│   ├── fig1_upf_distribution_dsst.png          ← UPF distribution & DSST by quartile
│   ├── fig2_forest_plot_OR.png                 ← Logistic OR forest plot
│   ├── fig3_roc_curve.png                      ← XGBoost ROC curve (AUC = 0.812)
│   ├── fig4_shap_summary.png                   ← SHAP beeswarm summary plot
│   ├── fig5_shap_dependence_upf.png            ← SHAP dependence plot for UPF
│   ├── fig6_shap_bar_importance.png            ← SHAP global feature importance
│   └── table1_publication.png                  ← Publication-ready Table 1
├── analysis/
│   ├── nova_classification.py                  ← NOVA 1–4 food classifier
│   ├── pipeline_step1_data_merge.py            ← NHANES data load & merge
│   ├── pipeline_step2a_v2.py                   ← Descriptive stats & Figures 1–2
│   ├── pipeline_step2b_v2.py                   ← Regression, XGBoost, SHAP, Figs 3–6
│   ├── gen_table1.py                           ← Publication Table 1 generator
│   ├── expand_limitations.py                   ← Limitations section expander
│   └── convert_to_latex.py                     ← Markdown → IEEE LaTeX converter
└── data/
    ├── nhanes_merged.csv                       ← Final analytic dataset (n=2,780, 21 vars)
    ├── results_summary.json                    ← Full model results
    ├── table1_data.csv                         ← Table 1 raw values
    ├── table1_descriptives.csv                 ← Descriptive statistics by quartile
    └── logit_OR_upf_quartile.csv               ← Logistic regression ORs
```

---

## Data Sources

- **NHANES 2011–2012 & 2013–2014** — Dietary recall (DR1TOT), cognitive function (CFQ/DSST), demographics (DEMO), BMI (BMX), physical activity (PAQ)
- **NOVA classification** — Foods classified by processing level (Groups 1–4) per Monteiro et al. (2019)

> Raw NHANES `.xpt` files are publicly available at [https://www.cdc.gov/nchs/nhanes](https://www.cdc.gov/nchs/nhanes) and are not redistributed here.

---

## Reproducing the Analysis

```bash
# 1. Download NHANES XPT files to nhanes_data/
# 2. Run pipeline
python analysis/pipeline_step1_data_merge.py
python analysis/pipeline_step2a_v2.py
python analysis/pipeline_step2b_v2.py
python analysis/gen_table1.py

# Environment requirements
pip install pandas numpy scipy scikit-learn xgboost matplotlib seaborn pyreadstat
```

Set `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1` to prevent OpenMP threading conflicts.

---

## Citation

If you use this code or manuscript, please cite:

> K. K. R. Poreddy, "Ultra-processed food consumption as a behavioral predictor of EEG-indexed cognitive fatigue: an AI framework for workforce mental health surveillance," *IEEE J. Biomed. Health Inform.*, under review, 2024.

---

## License

Code: MIT License. Manuscript: All rights reserved (under IEEE submission).
