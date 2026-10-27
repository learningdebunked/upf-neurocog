# Data Availability Statement

## For IEEE JBHI Submission

> **Suggested placement:** End of manuscript, before References (IEEE JBHI author guidelines, Section III-G).

---

### Statement (ready to paste into manuscript)

The analysis code, derived analytic dataset, and all publication figures are
openly available in a public GitHub repository at
[https://github.com/learningdebunked/upf-neurocog](https://github.com/learningdebunked/upf-neurocog)
under the MIT License.

The analytic dataset (`data/nhanes_merged.csv`) was derived entirely from the
**National Health and Nutrition Examination Survey (NHANES) 2011–2012 and
2013–2014** cycles, administered by the U.S. Centers for Disease Control and
Prevention (CDC), National Center for Health Statistics (NCHS). The raw NHANES
public-use data files are freely available at
[https://www.cdc.gov/nchs/nhanes](https://www.cdc.gov/nchs/nhanes) and are not
redistributed in this repository in compliance with the NCHS data use agreement.
No individual-level data were collected by the authors; all data used are
de-identified public-use files exempt from IRB review under 45 CFR 46.101(b).

Food processing classification followed the **NOVA system** (Monteiro et al.,
2019) and was applied programmatically via `analysis/nova_classification.py`.
The classification logic, NHANES merge pipeline, statistical analysis, and
XGBoost/SHAP model training scripts are fully documented and reproducible using
the instructions in `README.md`.

---

### Repository Contents Summary

| Path | Description |
|------|-------------|
| `manuscript/` | IEEE JBHI LaTeX source (`.tex`) and compiled PDF |
| `analysis/` | Full Python pipeline — NOVA classifier, NHANES merge, stats, ML, figures |
| `data/nhanes_merged.csv` | Final analytic sample (n = 2,780; 21 variables) |
| `data/results_summary.json` | All reported model metrics (AUC, ORs, SHAP values, ANOVA) |
| `figures/` | All 6 manuscript figures and publication Table 1 |
| `requirements.txt` | Pinned Python dependencies |
| `environment.yml` | Conda environment for full reproducibility |

---

### Reproducibility

All results reported in this manuscript can be reproduced by:

1. Downloading the NHANES 2011–2014 public-use `.xpt` files from the CDC website.
2. Placing them in a local `nhanes_data/` directory.
3. Running the pipeline in sequence as described in `README.md`.

Execution requires Python 3.11 and the packages listed in `requirements.txt`.
Total runtime on a standard laptop (8-core, 16 GB RAM): approximately 5–10 minutes.

---

### Competing Interests

The authors declare no competing financial or non-financial interests in relation
to the data sources or datasets used in this work.
