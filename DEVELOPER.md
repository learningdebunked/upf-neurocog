# DEVELOPER GUIDE — UPF-NeuroCog v2.0

> **End-to-end walkthrough:** from data download → classification → integration → model training → figures

---

## Table of Contents

1. [System Requirements](#1-system-requirements)
2. [Repository Structure](#2-repository-structure)
3. [Environment Setup](#3-environment-setup)
4. [Step 0 — Data Download](#step-0--data-download)
5. [Step 1 — NHANES Pipeline](#step-1--nhanes-pipeline)
6. [Step 2 — BRFSS Pipeline](#step-2--brfss-pipeline)
7. [Step 3 — DEAP EEG Pipeline](#step-3--deap-eeg-pipeline)
8. [Step 4 — SDOH / Equity Layer](#step-4--sdoh--equity-layer)
9. [Step 5 — NOVA LLM Classifier](#step-5--nova-llm-classifier)
10. [Step 6 — Integration Pipeline](#step-6--integration-pipeline)
11. [Step 7 — Model Training & Evaluation](#step-7--model-training--evaluation)
12. [Run Everything End-to-End](#run-everything-end-to-end)
13. [Output Files Reference](#output-files-reference)
14. [Troubleshooting](#troubleshooting)
15. [GitHub CI](#github-ci)

---

## 1. System Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| OS | Ubuntu 20.04 / macOS 12 | Ubuntu 22.04 / macOS 14 |
| Python | 3.10 | 3.11 |
| RAM | 16 GB | 32 GB (BRFSS 2023 is ~400 MB in memory) |
| Disk | 10 GB free | 20 GB free |
| CPU | 4 cores | 8+ cores |
| GPU | Not required | Optional (DEAP EEG faster with CUDA) |

---

## 2. Repository Structure

```
upf-neurocog-v2/
├── analysis/
│   ├── nova_llm_classifier.py      # LLM NOVA 1-4 food classifier
│   ├── nhanes_pipeline.py          # NHANES 2017-2020 ingestion + features
│   ├── brfss_pipeline.py           # BRFSS 2023 ingestion + features
│   ├── deap_eeg_pipeline.py        # DEAP EEG loading + band power + PAC/CFC
│   ├── sdoh_equity.py              # CDC SVI + USDA + PLACES stratification
│   ├── integration_pipeline.py     # Master merge (NHANES+BRFSS+SDOH+DEAP)
│   └── model_training.py           # XGBoost + SHAP + ROC
├── data/
│   ├── download_datasets.sh        # Automated data downloader
│   ├── nhanes/                     # XPT files land here
│   ├── brfss/                      # LLCP2023.XPT lands here
│   ├── sdoh/                       # SVI, PLACES, USDA Atlas CSVs
│   └── deap/                       # s01.dat … s32.dat land here
├── models/                         # Saved model artefacts
├── results/                        # Metrics JSON + SHAP CSV
├── figures/                        # ROC curve + SHAP beeswarm
├── manuscript/
│   ├── manuscript_v2_upf_neurocog_jbhi.tex   # IEEE JBHI LaTeX
│   └── manuscript_v2_upf_neurocog_jbhi.md    # Markdown mirror
├── .github/workflows/ci.yml        # GitHub Actions CI
├── environment.yml                 # Conda environment
├── requirements.txt                # pip install target
├── README.md                       # Project overview
├── DATA_AVAILABILITY.md            # All dataset URLs + citation guidance
└── DEVELOPER.md                    # This file
```

---

## 3. Environment Setup

### Option A — Conda (recommended)

```bash
conda env create -f environment.yml
conda activate upf-neurocog-v2
```

### Option B — pip (virtual environment)

```bash
python3.11 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Verify installation

```bash
python -c "import xgboost, shap, mne, pyreadstat; print('All dependencies OK')"
```

---

## Step 0 — Data Download

```bash
chmod +x data/download_datasets.sh
bash data/download_datasets.sh
```

This script automatically downloads:
- NHANES 2017-2018 (7 XPT files)
- NHANES 2019-2020 (7 XPT files)
- BRFSS 2023 (~400 MB ZIP, auto-extracted)
- USDA Food Desert Atlas 2019 (Excel)
- CDC PLACES 2023 (CSV via API)

**Manual downloads required (see `DATA_AVAILABILITY.md` for full instructions):**

| Dataset | Why Manual | Where to save |
|---------|-----------|---------------|
| CDC SVI 2022 | Form-based download | `data/sdoh/SVI2022_US_county.csv` |
| DEAP EEG | Registration wall | `data/deap/s01.dat` … `data/deap/s32.dat` |

> You can run the full pipeline without DEAP and SVI — those modules fall back gracefully.

---

## Step 1 — NHANES Pipeline

Merges dietary, demographic, clinical, and activity data from NHANES 2017-2020.

```bash
cd analysis
python nhanes_pipeline.py \
    --data_dir ../data/nhanes/ \
    --output   ../data/nhanes_processed.parquet
```

**Key outputs:**
- `data/nhanes_processed.parquet` — ~18K rows × 30 cols
- Columns include: `upf_energy_pct`, `nova4_energy_kcal`, `total_energy_kcal`,
  `sugar_g`, `sodium_mg`, `bmi`, `hba1c_pct`, `systolic_bp`, `age_years`, `sex_male`,
  `pir_ratio`, `race_ethnicity_code`

---

## Step 2 — BRFSS Pipeline

Cleans BRFSS 2023 survey data and computes stress-behaviour score.

```bash
python brfss_pipeline.py \
    --xpt_path ../data/brfss/LLCP2023.XPT \
    --output   ../data/brfss_processed.parquet
```

**Key outputs:**
- `data/brfss_processed.parquet` — ~430K rows
- Columns include: `brfss_poor_mh_days`, `brfss_sleep_insufficient`,
  `brfss_inactive`, `brfss_stress_score`, `age_years`, `sex_male`, `race_code`, `state`

---

## Step 3 — DEAP EEG Pipeline

Loads `.dat` files, extracts band power, PAC/CFC (theta-gamma), and frontal alpha asymmetry.

```bash
python deap_eeg_pipeline.py \
    --data_dir ../data/deap/ \
    --output   ../data/deap_features.csv
```

**Key outputs:**
- `data/deap_features.csv` — 32 subjects × 40 trials, EEG features per trial
- Columns: `subject_id`, `trial_id`, `delta_power`, `theta_power`, `alpha_power`,
  `beta_power`, `gamma_power`, `pac_mi_theta_gamma`, `frontal_alpha_asym`,
  `arousal`, `valence`, `arousal_valence_quad`

**If DEAP data is unavailable:** the pipeline produces a graceful stub CSV and the
integration pipeline maps EEG archetypes from population risk scores.

---

## Step 4 — SDOH / Equity Layer

Loads CDC SVI 2022, USDA Food Desert Atlas, and CDC PLACES; computes stratified risk and fairness diagnostics.

```bash
python sdoh_equity.py \
    --data_dir ../data/sdoh/ \
    --output   ../data/sdoh_processed.csv
```

**Key outputs:**
- `data/sdoh_processed.csv` — county-level SDOH flags
- Columns: `fips`, `svi_percentile`, `food_desert_flag`,
  `places_mental_health_poor_pct`, `equity_risk_stratum`

---

## Step 5 — NOVA LLM Classifier

Classifies food item descriptions into NOVA 1-4. Three backend options:

### A. Rule-based (no API key required)

```bash
python nova_llm_classifier.py \
    --input  ../data/nhanes/DR1IFF_J.XPT \
    --output ../data/nova_classified.csv \
    --backend rule
```

### B. OpenAI GPT-4o (requires OPENAI_API_KEY)

```bash
export OPENAI_API_KEY="sk-..."
python nova_llm_classifier.py \
    --input   ../data/nhanes/DR1IFF_J.XPT \
    --output  ../data/nova_classified.csv \
    --backend openai \
    --model   gpt-4o
```

### C. HuggingFace (local, no API key)

```bash
python nova_llm_classifier.py \
    --input   ../data/nhanes/DR1IFF_J.XPT \
    --output  ../data/nova_classified.csv \
    --backend hf \
    --model   mistralai/Mistral-7B-Instruct-v0.3
```

**Key outputs:**
- `data/nova_classified.csv` — food items with `nova_class` (1-4) column

---

## Step 6 — Integration Pipeline

Merges NHANES + BRFSS + SDOH + DEAP into a single unified cohort.

```bash
python integration_pipeline.py \
    --nhanes_dir  ../data/nhanes/ \
    --brfss_file  ../data/brfss/LLCP2023.XPT \
    --sdoh_dir    ../data/sdoh/ \
    --deap_dir    ../data/deap/ \
    --output_dir  ../data/
```

**Key outputs:**
- `data/integrated_cohort.parquet` — merged population cohort
- `data/eeg_feature_summary.csv` — DEAP archetype summary
- `data/integration_report.json` — QC report (row counts, missing %, label distribution)

---

## Step 7 — Model Training & Evaluation

Trains XGBoost on the integrated cohort; runs SHAP explainability and plots ROC.

```bash
python model_training.py \
    --cohort   ../data/integrated_cohort.parquet \
    --output   ../results/ \
    --models   ../models/ \
    --figures  ../figures/
```

**Key outputs:**
- `models/xgb_model.pkl` — trained XGBoost model
- `models/label_encoder.pkl` — feature encoders
- `results/metrics.json` — AUROC, F1, CV statistics
- `results/shap_summary.csv` — ranked SHAP feature importances
- `figures/shap_beeswarm.png` — SHAP beeswarm plot (top 20 features)
- `figures/roc_curve.png` — ROC curve

### Expected model performance (based on v1.0 preliminary + v2.0 scale-up estimates)

| Metric | v1.0 (n=2,780) | v2.0 target (n~18K NHANES + BRFSS stratified) |
|--------|---------------|----------------------------------------------|
| AUROC  | 0.812         | ≥ 0.840 |
| F1     | —             | ≥ 0.78 |
| SHAP rank #1 | UPF energy share | UPF energy share (expected) |

---

## Run Everything End-to-End

The snippet below runs the full pipeline in the correct dependency order.

```bash
#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/analysis"

echo "=== Step 0: Download datasets ==="
bash ../data/download_datasets.sh

echo "=== Step 1: NHANES pipeline ==="
python nhanes_pipeline.py --data_dir ../data/nhanes/ --output ../data/nhanes_processed.parquet

echo "=== Step 2: BRFSS pipeline ==="
python brfss_pipeline.py --xpt_path ../data/brfss/LLCP2023.XPT --output ../data/brfss_processed.parquet

echo "=== Step 3: DEAP EEG pipeline (requires manual data) ==="
python deap_eeg_pipeline.py --data_dir ../data/deap/ --output ../data/deap_features.csv || \
    echo "DEAP pipeline skipped (data not present — using archetypes fallback)"

echo "=== Step 4: SDOH equity layer ==="
python sdoh_equity.py --data_dir ../data/sdoh/ --output ../data/sdoh_processed.csv

echo "=== Step 5: NOVA LLM classification (rule-based) ==="
python nova_llm_classifier.py --input ../data/nhanes/DR1IFF_J.XPT \
    --output ../data/nova_classified.csv --backend rule

echo "=== Step 6: Integration pipeline ==="
python integration_pipeline.py \
    --nhanes_dir ../data/nhanes/ \
    --brfss_file ../data/brfss/LLCP2023.XPT \
    --sdoh_dir   ../data/sdoh/ \
    --deap_dir   ../data/deap/ \
    --output_dir ../data/

echo "=== Step 7: Model training ==="
python model_training.py \
    --cohort  ../data/integrated_cohort.parquet \
    --output  ../results/ \
    --models  ../models/ \
    --figures ../figures/

echo ""
echo "=== Pipeline complete. Check results/ and figures/ for outputs. ==="
```

Save as `run_all.sh` and execute:

```bash
chmod +x run_all.sh
bash run_all.sh
```

---

## Output Files Reference

| File | Stage | Description |
|------|-------|-------------|
| `data/nhanes_processed.parquet` | Step 1 | NHANES cleaned cohort |
| `data/brfss_processed.parquet` | Step 2 | BRFSS cleaned cohort |
| `data/deap_features.csv` | Step 3 | DEAP EEG features per trial |
| `data/sdoh_processed.csv` | Step 4 | SDOH county-level flags |
| `data/nova_classified.csv` | Step 5 | NHANES food items with NOVA class |
| `data/integrated_cohort.parquet` | Step 6 | Merged population cohort |
| `data/eeg_feature_summary.csv` | Step 6 | DEAP archetype mapping |
| `data/integration_report.json` | Step 6 | QC metrics |
| `models/xgb_model.pkl` | Step 7 | Trained XGBoost |
| `models/label_encoder.pkl` | Step 7 | Feature encoders |
| `results/metrics.json` | Step 7 | AUROC, F1, CV |
| `results/shap_summary.csv` | Step 7 | SHAP feature ranking |
| `figures/shap_beeswarm.png` | Step 7 | SHAP visualisation |
| `figures/roc_curve.png` | Step 7 | ROC curve |

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'pyreadstat'`
```bash
pip install pyreadstat
```

### `FileNotFoundError: LLCP2023.XPT`
BRFSS ZIP auto-extracts as `LLCP2023XPT.XPT` in some platforms. Rename:
```bash
mv data/brfss/LLCP2023XPT.XPT data/brfss/LLCP2023.XPT
```

### DEAP pipeline raises `FileNotFoundError`
DEAP requires manual registration and download. See `DATA_AVAILABILITY.md §3`.
The integration pipeline handles DEAP absence gracefully — it will fill EEG
columns with `NaN` and assign archetypes from population risk scores alone.

### OpenAI NOVA classifier: `RateLimitError`
Reduce batch size:
```bash
python nova_llm_classifier.py --batch_size 20 --backend openai
```

### Out of memory on BRFSS (~400 MB XPT)
Increase system swap or use chunked loading (built into `brfss_pipeline.py`):
```bash
python brfss_pipeline.py --chunk_size 50000
```

### XGBoost training slow on large NHANES+BRFSS combined cohort
Enable histogram tree method with GPU if CUDA is available:
```python
# In model_training.py, add to XGB_PARAMS:
"device": "cuda"
```

---

## GitHub CI

Every push to `main` or `v2` triggers three jobs in `.github/workflows/ci.yml`:

| Job | What it checks |
|-----|---------------|
| `lint` | `flake8` style + `isort` import order |
| `test` | `pytest tests/` unit tests |
| `smoke` | Imports `NOVAClassifier`, runs rule-based classification on 4 foods, writes stub cohort |

To run CI locally before pushing:
```bash
pip install flake8 isort pytest
flake8 analysis/ --max-line-length=120 --ignore=E203,W503
isort --check-only analysis/
pytest tests/ -v
```

---

## Contact & Citation

If you use this framework in research, please cite:

```
[Author et al., 2025]. UPF-NeuroCog v2.0: An LLM-Driven NOVA Food Classification
Framework for Population-Scale Cognitive Fatigue Surveillance Using NHANES, BRFSS,
and EEG Neural Biomarker Validation with Equity-Aware SDOH Stratification.
IEEE Journal of Biomedical and Health Informatics (under review).
```

For issues and contributions: https://github.com/learningdebunked/upf-neurocog
