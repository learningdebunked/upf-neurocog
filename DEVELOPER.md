# Developer Guide — How to Run the Code

**Project:** Ultra-Processed Food Consumption as a Behavioral Predictor of EEG-Indexed Cognitive Fatigue  
**Repo:** https://github.com/learningdebunked/upf-neurocog

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Environment Setup](#2-environment-setup)
3. [Download NHANES Data](#3-download-nhanes-data)
4. [Directory Structure](#4-directory-structure)
5. [Running the Pipeline](#5-running-the-pipeline)
6. [Expected Outputs](#6-expected-outputs)
7. [Customising the Pipeline](#7-customising-the-pipeline)
8. [Compiling the LaTeX Manuscript](#8-compiling-the-latex-manuscript)
9. [Troubleshooting](#9-troubleshooting)
10. [Script Reference](#10-script-reference)

---

## 1. Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | >= 3.10 | 3.11 recommended |
| Conda or pip | any recent | either works (see Section 2) |
| TeX Live | >= 2022 | only needed for PDF compilation |
| RAM | >= 8 GB | DR1IFF files are ~83-85 MB each |
| Disk space | >= 2 GB | raw NHANES XPTs + outputs |

---

## 2. Environment Setup

### Option A — Conda (recommended)

```bash
git clone https://github.com/learningdebunked/upf-neurocog.git
cd upf-neurocog

conda env create -f environment.yml
conda activate upf-neurocog
```

### Option B — pip + virtualenv

```bash
git clone https://github.com/learningdebunked/upf-neurocog.git
cd upf-neurocog

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Thread-safety environment variables (Linux/macOS — required)

Set these before running any script, or add them to your shell profile:

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
```

On Windows (PowerShell):

```powershell
$Env:OMP_NUM_THREADS      = "1"
$Env:MKL_NUM_THREADS      = "1"
$Env:OPENBLAS_NUM_THREADS = "1"
```

---

## 3. Download NHANES Data

You need 12 XPT files across two survey cycles (2011-2012 = `_G`, 2013-2014 = `_H`).

| File | Cycle | Module | Size |
|------|-------|--------|------|
| DEMO_G.xpt | 2011-12 | Demographics | ~4 MB |
| DEMO_H.xpt | 2013-14 | Demographics | ~4 MB |
| CFQ_G.xpt  | 2011-12 | Cognitive Function (DSST) | ~1 MB |
| CFQ_H.xpt  | 2013-14 | Cognitive Function (DSST) | ~1 MB |
| BMX_G.xpt  | 2011-12 | Body Measures (BMI) | ~2 MB |
| BMX_H.xpt  | 2013-14 | Body Measures (BMI) | ~2 MB |
| PAQ_G.xpt  | 2011-12 | Physical Activity | ~2 MB |
| PAQ_H.xpt  | 2013-14 | Physical Activity | ~2 MB |
| DR1TOT_G.xpt | 2011-12 | Dietary Totals | ~5 MB |
| DR1TOT_H.xpt | 2013-14 | Dietary Totals | ~5 MB |
| DR1IFF_G.xpt | 2011-12 | Individual Food Items | ~83 MB |
| DR1IFF_H.xpt | 2013-14 | Individual Food Items | ~85 MB |

### Bulk download (bash)

```bash
mkdir -p nhanes_data && cd nhanes_data

BASE_G="https://wwwn.cdc.gov/Nchs/Nhanes/2011-2012"
BASE_H="https://wwwn.cdc.gov/Nchs/Nhanes/2013-2014"

for f in DEMO CFQ BMX PAQ DR1TOT DR1IFF; do
  curl -L -O "${BASE_G}/${f}_G.XPT"
  curl -L -O "${BASE_H}/${f}_H.XPT"
done

# Rename to lowercase .xpt (scripts expect lowercase)
for f in *.XPT; do mv "$f" "${f%.XPT}.xpt"; done
cd ..
```

> DR1IFF files are ~83-85 MB each. Allow 5-15 minutes on a slow connection.

---

## 4. Directory Structure

After cloning and downloading NHANES data:

```
upf-neurocog/
├── nhanes_data/                <- create this; put the 12 XPT files here
│   ├── DEMO_G.xpt  DEMO_H.xpt
│   ├── CFQ_G.xpt   CFQ_H.xpt
│   ├── BMX_G.xpt   BMX_H.xpt
│   ├── PAQ_G.xpt   PAQ_H.xpt
│   ├── DR1TOT_G.xpt DR1TOT_H.xpt
│   └── DR1IFF_G.xpt DR1IFF_H.xpt
├── outputs/                    <- create this; figures land here
├── analysis/                   <- pipeline scripts (in repo)
├── data/                       <- intermediate & result files (in repo)
├── manuscript/
├── figures/
├── requirements.txt
├── environment.yml
└── README.md
```

Create the two missing directories before running:

```bash
mkdir -p nhanes_data outputs
```

---

## 5. Running the Pipeline

All scripts live in `analysis/` and must be run **in order** from the project root.

---

### Step 1 — Data Merge & NOVA Classification

**Script:** `analysis/pipeline_step1_data_merge.py`  
**Runtime:** ~5-10 min (NOVA classifies ~500K food records)  
**Reads:** `nhanes_data/*.xpt` (all 12 files)  
**Writes:** `data/nhanes_merged.csv`

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python analysis/pipeline_step1_data_merge.py
```

What it does:
1. Loads demographics and filters to adults >= 60 years
2. Loads DSST cognitive scores, BMI, physical activity, dietary totals
3. Loads individual food items in chunks and applies NOVA 1-4 classification
4. Computes per-person UPF energy share (`upf_pct = NOVA-4 kcal / total kcal x 100`)
5. Merges all modules on SEQN and saves `nhanes_merged.csv`

Expected final line:

```
Final analytic sample: 2,780 participants, 21 variables
Saved -> data/nhanes_merged.csv
```

---

### Step 2a — Descriptive Statistics & Figures 1-2

**Script:** `analysis/pipeline_step2a_v2.py`  
**Runtime:** ~1 min  
**Reads:** `data/nhanes_merged.csv`  
**Writes:** `outputs/fig1_upf_distribution_dsst.png`, `outputs/fig2_forest_plot_OR.png`,
`data/table1_descriptives.csv`, `data/logit_OR_upf_quartile.csv`

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python analysis/pipeline_step2a_v2.py
```

What it does:
1. Assigns UPF energy quartiles (Q1-Q4)
2. Computes Table 1 descriptive statistics stratified by quartile
3. Runs one-way ANOVA on DSST score (F=2.87, p=0.035)
4. Fits logistic regression and computes odds ratios with 95% CIs
5. Saves Figure 1 (UPF histogram + DSST boxplot) and Figure 2 (forest plot)

---

### Step 2b — Machine Learning, SHAP & Figures 3-6

**Script:** `analysis/pipeline_step2b_v2.py`  
**Runtime:** ~2-5 min  
**Reads:** `data/nhanes_merged.csv`  
**Writes:** `outputs/fig3_roc_curve.png` through `outputs/fig6_shap_bar_importance.png`,
`data/results_summary.json`

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python analysis/pipeline_step2b_v2.py
```

What it does:
1. Prepares 11-feature matrix and binary outcome (`cognitive_impaired`)
2. Trains XGBoost with 5-fold stratified cross-validation (mean AUC = 0.812)
3. Computes SHAP values via XGBoost native `pred_contribs=True`
4. Saves Figures 3-6 and all key metrics to `results_summary.json`

---

### Step 3 — Publication Table 1

**Script:** `analysis/gen_table1.py`  
**Runtime:** ~30 sec  
**Reads:** `data/nhanes_merged.csv`  
**Writes:** `outputs/table1_publication.png`, `data/table1_data.csv`

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python analysis/gen_table1.py
```

Renders a publication-quality Table 1 PNG with means +/- SD, n (%), and p-values.

---

### Run all steps (one-liner)

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

python analysis/pipeline_step1_data_merge.py && \
python analysis/pipeline_step2a_v2.py        && \
python analysis/pipeline_step2b_v2.py        && \
python analysis/gen_table1.py

echo "Pipeline complete."
```

---

## 6. Expected Outputs

After all four steps complete:

```
data/
  nhanes_merged.csv          2,780 rows x 21 columns
  results_summary.json       AUC, ORs, SHAP ranks, ANOVA stats
  table1_data.csv            Table 1 raw cell values
  table1_descriptives.csv    per-quartile descriptive stats
  logit_OR_upf_quartile.csv  odds ratios for Q2/Q3/Q4 vs Q1

outputs/
  fig1_upf_distribution_dsst.png   UPF histogram + DSST box by quartile
  fig2_forest_plot_OR.png          Logistic OR forest plot
  fig3_roc_curve.png               ROC curve (AUC ~ 0.812)
  fig4_shap_summary.png            SHAP beeswarm (11 features)
  fig5_shap_dependence_upf.png     SHAP dependence — UPF feature
  fig6_shap_bar_importance.png     Global mean |SHAP| bar chart
  table1_publication.png           Publication-ready Table 1
```

Key numbers to verify in `data/results_summary.json`:

| Key | Expected |
|-----|----------|
| `n_total` | 2780 |
| `impairment_rate` | ~0.164 |
| `upf_mean_pct` | ~8.1 |
| `anova_pvalue` | ~0.035 |
| `xgb_auc` | ~0.812 |
| `upf_shap_rank` | 8 |

---

## 7. Customising the Pipeline

### Change the age cutoff

`analysis/pipeline_step1_data_merge.py`, line ~36:

```python
demo_60 = demo[demo['RIDAGEYR'] >= 60]   # change 60 to any age
```

### Change the impairment threshold

`analysis/pipeline_step1_data_merge.py`, line ~190:

```python
# Default: bottom quartile of DSST = impaired
df['cognitive_impaired'] = (df['dsst_score'] < df['dsst_score'].quantile(0.25)).astype(int)
```

### Change XGBoost hyperparameters

`analysis/pipeline_step2b_v2.py`, line ~60:

```python
model = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
```

### Add or remove features

`analysis/pipeline_step2b_v2.py`, line ~40:

```python
FEATURES = [
    'upf_pct', 'RIDAGEYR', 'RIAGENDR', 'INDFMPIR',
    'DXDTOTBMI', 'DR1TKCAL', 'educ_college',
    'eth_hispanic', 'eth_black', 'eth_asian', 'eth_other'
]
```

Add any column present in `nhanes_merged.csv`.

---

## 8. Compiling the LaTeX Manuscript

### Install TeX Live (Ubuntu/Debian)

```bash
sudo apt-get install texlive-publishers   # includes IEEEtran (~200 MB)
# or for all packages:
sudo apt-get install texlive-full
```

### Compile

```bash
cd manuscript

# Copy figures into the same directory for pdflatex
cp ../figures/fig*.png .
cp ../figures/table1_publication.png .

pdflatex -interaction=nonstopmode manuscript_upf_eeg_jbhi_ieee.tex
pdflatex -interaction=nonstopmode manuscript_upf_eeg_jbhi_ieee.tex   # second pass for cross-refs
```

The compiled PDF will appear as `manuscript/manuscript_upf_eeg_jbhi_ieee.pdf`.
A pre-compiled copy is already in the repo.

---

## 9. Troubleshooting

**`FileNotFoundError: DEMO_G.xpt not found`**  
Files must be in `nhanes_data/` (lowercase `.xpt`) relative to the project root.
```bash
ls nhanes_data/*.xpt   # should list 12 files
```

**Pipeline hangs indefinitely**  
OpenMP thread deadlock. Set:
```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
```

**`ModuleNotFoundError: No module named 'xgboost'`**  
Environment is not active. Re-activate:
```bash
conda activate upf-neurocog
pip install -r requirements.txt
```

**SHAP raises AttributeError or crashes**  
The pipeline uses XGBoost native SHAP — never `shap.TreeExplainer`:
```python
# Correct
contribs = model.get_booster().predict(dmatrix, pred_contribs=True)
# Avoid
# explainer = shap.TreeExplainer(model)
```

**`pdflatex` error: `File 'IEEEtran.cls' not found`**  
```bash
sudo apt-get install texlive-publishers
```

**CI / GitHub Actions fails on NOVA spot-check**  
Update expected values in `.github/workflows/validate_pipeline.yml` under the
"Validate NOVA classification logic" step to match your updated classifier.

---

## 10. Script Reference

| Script | Purpose | Inputs | Outputs |
|--------|---------|--------|---------|
| `nova_classification.py` | NOVA 1-4 food classifier | food name string | integer 1-4 |
| `pipeline_step1_data_merge.py` | Load, classify, merge NHANES | 12 XPT files | `nhanes_merged.csv` |
| `pipeline_step2a_v2.py` | Descriptive stats, ORs, Figs 1-2 | `nhanes_merged.csv` | Figs 1-2, CSVs |
| `pipeline_step2b_v2.py` | XGBoost, SHAP, Figs 3-6 | `nhanes_merged.csv` | Figs 3-6, JSON |
| `gen_table1.py` | Publication Table 1 | `nhanes_merged.csv` | `table1_publication.png` |
| `expand_limitations.py` | Expand manuscript limitations | Markdown manuscript | Updated `.md` |
| `convert_to_latex.py` | Markdown to IEEE LaTeX | Markdown manuscript | `.tex` source |

---

*For questions or issues, open a GitHub Issue at https://github.com/learningdebunked/upf-neurocog/issues*
