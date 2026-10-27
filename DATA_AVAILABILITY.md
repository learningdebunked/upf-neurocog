# Data Availability Statement — UPF-NeuroCog v2.0

This document lists every dataset used in UPF-NeuroCog v2.0, its access status,
exact download URL, and any terms or registration requirements.

---

## 1. NHANES 2017–2020 (National Health and Nutrition Examination Survey)

| Attribute | Details |
|-----------|---------|
| Provider | U.S. Centers for Disease Control and Prevention (CDC) / NCHS |
| Access | Public — no registration required |
| License | Public domain (U.S. government data) |
| Coverage | 2017-2018 (cycle J) and 2019-2020 (cycle P) |
| Approximate N | ~9,000 per cycle (dietary sub-sample) |

### File URLs

#### Cycle J (2017-2018)
| File | URL |
|------|-----|
| Dietary totals (Day 1) | https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/DR1TOT_J.XPT |
| Individual food items | https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/DR1IFF_J.XPT |
| Demographics | https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/DEMO_J.XPT |
| Body measures (BMI) | https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/BMX_J.XPT |
| Physical activity | https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/PAQ_J.XPT |
| Glycohemoglobin (HbA1c) | https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/GHB_J.XPT |
| Blood pressure | https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/BPX_J.XPT |

#### Cycle P (2019-2020)
| File | URL |
|------|-----|
| Dietary totals (Day 1) | https://wwwn.cdc.gov/Nchs/Nhanes/2019-2020/DR1TOT_P.XPT |
| Individual food items | https://wwwn.cdc.gov/Nchs/Nhanes/2019-2020/DR1IFF_P.XPT |
| Demographics | https://wwwn.cdc.gov/Nchs/Nhanes/2019-2020/DEMO_P.XPT |
| Body measures (BMI) | https://wwwn.cdc.gov/Nchs/Nhanes/2019-2020/BMX_P.XPT |
| Physical activity | https://wwwn.cdc.gov/Nchs/Nhanes/2019-2020/PAQ_P.XPT |
| Glycohemoglobin (HbA1c) | https://wwwn.cdc.gov/Nchs/Nhanes/2019-2020/GHB_P.XPT |
| Blood pressure (oscillometric) | https://wwwn.cdc.gov/Nchs/Nhanes/2019-2020/BPXO_P.XPT |

---

## 2. BRFSS 2023 (Behavioral Risk Factor Surveillance System)

| Attribute | Details |
|-----------|---------|
| Provider | U.S. CDC |
| Access | Public — no registration required |
| License | Public domain |
| Coverage | All 50 states + DC + territories, 2023 |
| Approximate N | ~430,000 respondents |
| File format | SAS XPT (zipped) |

| File | URL |
|------|-----|
| LLCP2023XPT.zip | https://www.cdc.gov/brfss/annual_data/2023/files/LLCP2023XPT.zip |

---

## 3. DEAP EEG Dataset (Database for Emotion Analysis using Physiological Signals)

| Attribute | Details |
|-----------|---------|
| Provider | Queen Mary University of London |
| Access | **Registration required** (free academic license) |
| Registration URL | https://www.eecs.qmul.ac.uk/mmv/datasets/deap/download.html |
| License | Non-commercial academic use only |
| Coverage | 32 participants × 40 music video trials, 32-channel EEG at 512 Hz |
| File | data_preprocessed_python.zip (~2.9 GB) |

> **Note:** DEAP data cannot be redistributed. Researchers must register individually
> at the URL above. The `deap_eeg_pipeline.py` script expects `.dat` files placed in
> `data/deap/` (e.g., `s01.dat` … `s32.dat`).

---

## 4. CDC Social Vulnerability Index (SVI) 2022

| Attribute | Details |
|-----------|---------|
| Provider | U.S. CDC / ATSDR |
| Access | **Manual download via form** (free, no registration) |
| Download page | https://www.atsdr.cdc.gov/placeandhealth/svi/data_documentation_download.html |
| License | Public domain |
| Coverage | U.S. census tracts and counties, 2022 |

**Download instructions:**
1. Go to the download page above.
2. Select Geography: **US**, Year: **2022**, File Type: **CSV**.
3. Save `SVI2022_US_county.csv` to `data/sdoh/`.

---

## 5. USDA Economic Research Service — Food Access Research Atlas (2019)

| Attribute | Details |
|-----------|---------|
| Provider | U.S. Department of Agriculture (USDA ERS) |
| Access | Public — direct download link |
| License | Public domain |
| Coverage | U.S. census tracts, 2019 |
| File | FoodAccessResearchAtlasData2019.xlsx |

| File | URL |
|------|-----|
| Food Access Research Atlas 2019 | https://www.ers.usda.gov/webdocs/DataFiles/80591/FoodAccessResearchAtlasData2019.xlsx |

---

## 6. CDC PLACES 2023 (Population-Level Analysis and Community Estimates)

| Attribute | Details |
|-----------|---------|
| Provider | U.S. CDC |
| Access | Public — direct CSV download via CDC Open Data API |
| License | Public domain |
| Coverage | U.S. counties and census tracts, 2023 release |

| File | URL |
|------|-----|
| PLACES 2023 CSV | https://data.cdc.gov/api/views/cwsq-ngmh/rows.csv?accessType=DOWNLOAD |

---

## Automated Download

Run the provided shell script to download all publicly accessible datasets:

```bash
bash data/download_datasets.sh
```

CDC SVI 2022 and DEAP EEG require manual steps as noted above.

---

## Citation Guidance

When publishing work based on this framework, please cite each dataset appropriately:

- **NHANES:** National Center for Health Statistics. National Health and Nutrition Examination Survey. Hyattsville, MD: CDC, NCHS. https://www.cdc.gov/nchs/nhanes/
- **BRFSS:** Centers for Disease Control and Prevention. Behavioral Risk Factor Surveillance System Survey Data, 2023. Atlanta: U.S. Department of Health and Human Services.
- **DEAP:** Koelstra S, et al. "DEAP: A Database for Emotion Analysis Using Physiological Signals." *IEEE Trans Affective Comput*, 2012. DOI: 10.1109/T-AFFC.2011.15
- **CDC SVI:** Agency for Toxic Substances and Disease Registry. CDC/ATSDR Social Vulnerability Index 2022. https://www.atsdr.cdc.gov/placeandhealth/svi/
- **USDA Food Atlas:** Economic Research Service, U.S. Department of Agriculture. Food Access Research Atlas, 2019. https://www.ers.usda.gov/data-products/food-access-research-atlas/
- **CDC PLACES:** Centers for Disease Control and Prevention. PLACES: Local Data for Better Health, 2023. https://www.cdc.gov/places/
