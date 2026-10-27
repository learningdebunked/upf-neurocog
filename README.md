# UPF-NeuroCog v2.0

> **UPF-NeuroCog: An LLM-Driven NOVA Food Classification Framework for
> Population-Scale Cognitive Fatigue Surveillance Using NHANES, BRFSS,
> and EEG Neural Biomarker Validation with Equity-Aware SDOH Stratification**

[![CI](https://github.com/learningdebunked/upf-neurocog/actions/workflows/ci.yml/badge.svg)](https://github.com/learningdebunked/upf-neurocog/actions)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## What is UPF-NeuroCog v2.0?

This repository implements a full AI framework linking ultra-processed food (UPF)
consumption — classified by the NOVA system using LLMs — to EEG-indexed cognitive
fatigue at population scale. The framework integrates four independent data sources:

| Data Source | N | Role |
|-------------|---|------|
| **NHANES 2017–2020** | ~18,000 | Dietary intake + clinical biomarkers |
| **BRFSS 2023** | ~430,000 | Population-scale mental health + behaviour proxy |
| **DEAP EEG** | 32 subjects × 40 trials | Neural biomarker validation (PAC/CFC) |
| **CDC SDOH (SVI + PLACES + USDA)** | National | Equity stratification |

**Combined addressable population: 800,000+** (NHANES + BRFSS combined)

---

## Motivation — America's Burnout Crisis

> "America's burnout crisis is written in what we eat."

Cognitive fatigue costs the U.S. economy an estimated $411B per year in lost
productivity (Gallup 2023). Ultra-processed foods — now comprising >60% of caloric
intake for many American adults — are linked to neuroinflammation, dopamine
dysregulation, and impaired prefrontal cortex function. Yet no population-scale
AI framework has:

1. Applied LLM-based NOVA classification to real dietary survey data at this scale.
2. Validated dietary-cognitive associations against *real* EEG neural biomarkers
   (cross-frequency coupling, frontal alpha asymmetry).
3. Embedded equity-aware SDOH stratification to surface differential risk in
   food-insecure and high-vulnerability communities.

UPF-NeuroCog v2.0 closes all three gaps.

---

## Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                      UPF-NeuroCog v2.0 Framework                   │
├─────────────────┬──────────────────┬──────────────┬────────────────┤
│  NHANES 2017-20 │   BRFSS 2023     │  DEAP EEG    │  CDC SDOH      │
│  (dietary +     │  (mental health  │  (32 subj,   │  (SVI 2022,    │
│   clinical)     │   behaviour)     │   40 trials) │  PLACES, USDA) │
└────────┬────────┴────────┬─────────┴──────┬───────┴───────┬────────┘
         │                 │                │               │
         ▼                 ▼                ▼               ▼
  ┌─────────────┐  ┌─────────────┐  ┌──────────────┐  ┌──────────┐
  │  NOVA LLM   │  │  Stress-    │  │  Band power  │  │  SDOH    │
  │  Classifier │  │  Behaviour  │  │  PAC/CFC     │  │  Equity  │
  │  (1-4)      │  │  Score      │  │  FAA         │  │  Strata  │
  └──────┬──────┘  └──────┬──────┘  └──────┬───────┘  └────┬─────┘
         │                 │                │               │
         └────────────┬────┘                └───────┬───────┘
                      ▼                             ▼
             ┌──────────────────┐        ┌──────────────────┐
             │ Integration      │        │  EEG Archetype   │
             │ Pipeline         │◄───────│  Mapping         │
             │ (composite risk) │        │  (Q1-Q4)         │
             └────────┬─────────┘        └──────────────────┘
                      │
                      ▼
             ┌──────────────────┐
             │  XGBoost Model   │
             │  + SHAP          │
             │  + Equity Audit  │
             └──────────────────┘
```

---

## Key Innovations vs v1.0

| Feature | v1.0 | v2.0 |
|---------|------|------|
| Sample size | 2,780 | ~18K NHANES + 430K BRFSS |
| Dietary cycle | NHANES 2011-2014 | NHANES 2017-2020 |
| NOVA classification | Rule-based heuristics | LLM (GPT-4o / Mistral / rule fallback) |
| Cognitive outcome | DSST proxy score | BRFSS mental-health days + DEAP real EEG |
| EEG validation | None (proxy only) | DEAP real EEG: PAC/CFC theta-gamma, FAA |
| Mechanistic bridge | None | Cross-frequency coupling (Tort MI) |
| SDOH layer | None | CDC SVI 2022 + USDA Food Desert Atlas + CDC PLACES |
| Population frame | Clinical survey | Workforce burnout at national scale |
| Target journal | IEEE JBHI | IEEE JBHI (primary), npj Digital Medicine |

---

## Quickstart

### 1. Clone and set up environment

```bash
git clone https://github.com/learningdebunked/upf-neurocog.git
cd upf-neurocog
conda env create -f environment.yml
conda activate upf-neurocog-v2
```

### 2. Download datasets

```bash
bash data/download_datasets.sh
```

> DEAP EEG and CDC SVI 2022 require manual download — see `DATA_AVAILABILITY.md`.

### 3. Run the full pipeline

```bash
cd analysis

# Step 1-2: Dietary + behavioural pipelines
python nhanes_pipeline.py   --data_dir ../data/nhanes/ --output ../data/nhanes_processed.parquet
python brfss_pipeline.py    --xpt_path ../data/brfss/LLCP2023.XPT --output ../data/brfss_processed.parquet

# Step 3-4: EEG + SDOH
python deap_eeg_pipeline.py --data_dir ../data/deap/  --output ../data/deap_features.csv
python sdoh_equity.py       --data_dir ../data/sdoh/  --output ../data/sdoh_processed.csv

# Step 5: NOVA classification
python nova_llm_classifier.py --input ../data/nhanes/DR1IFF_J.XPT \
    --output ../data/nova_classified.csv --backend rule

# Step 6: Integration
python integration_pipeline.py \
    --nhanes_dir ../data/nhanes/ --brfss_file ../data/brfss/LLCP2023.XPT \
    --sdoh_dir ../data/sdoh/ --deap_dir ../data/deap/ --output_dir ../data/

# Step 7: Train model
python model_training.py \
    --cohort ../data/integrated_cohort.parquet \
    --output ../results/ --models ../models/ --figures ../figures/
```

For a complete guide including troubleshooting, see **[DEVELOPER.md](DEVELOPER.md)**.

---

## Repository Layout

```
upf-neurocog-v2/
├── analysis/           Python pipeline scripts
├── data/               Raw + processed data (XPTs, CSVs, Parquets)
├── models/             Saved model artefacts
├── results/            Metrics JSON + SHAP CSV
├── figures/            ROC curve + SHAP beeswarm
├── manuscript/         IEEE JBHI LaTeX + Markdown
├── environment.yml     Conda env spec
├── requirements.txt    pip install target
├── DATA_AVAILABILITY.md  All dataset URLs + citation guidance
└── DEVELOPER.md          End-to-end developer guide
```

---

## Preliminary Results (v1.0 — under IEEE JBHI review)

Establishes the prior work baseline:

| Metric | Value |
|--------|-------|
| Sample | n = 2,780 (NHANES 2011-2014) |
| AUROC | 0.812 |
| UPF energy share (mean ± SD) | 8.1% ± 9.6% |
| DSST ANOVA (UPF tertile) | F = 2.87, p = 0.035 |
| SHAP rank of UPF energy share | 8 of 11 features |

v2.0 targets AUROC ≥ 0.840 with equity-stratified validation.

---

## Datasets

See **[DATA_AVAILABILITY.md](DATA_AVAILABILITY.md)** for complete URLs, access
instructions, and citation guidance for all six datasets.

---

## Citation

If you use this framework, please cite:

```bibtex
@article{upfneurocog2025,
  title   = {UPF-NeuroCog: An LLM-Driven NOVA Food Classification Framework
             for Population-Scale Cognitive Fatigue Surveillance Using NHANES,
             BRFSS, and EEG Neural Biomarker Validation with Equity-Aware
             SDOH Stratification},
  journal = {IEEE Journal of Biomedical and Health Informatics},
  year    = {2025},
  note    = {Under review},
  url     = {https://github.com/learningdebunked/upf-neurocog}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.

Dataset licences vary — see [DATA_AVAILABILITY.md](DATA_AVAILABILITY.md).
DEAP EEG is for non-commercial academic use only.
