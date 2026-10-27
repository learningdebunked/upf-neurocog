"""
sdoh_equity.py  —  UPF-NeuroCog v2.0
======================================
Loads the CDC Social Vulnerability Index (SVI) 2022,
merges it with NHANES/BRFSS by state FIPS, stratifies
risk scores by race/income/food-desert, and runs
algorithmic fairness diagnostics.

Real dataset URLs:
  CDC SVI 2022 (county-level CSV):
    https://www.atsdr.cdc.gov/placeandhealth/svi/data_documentation_download.html
  USDA Food Access Research Atlas (food deserts):
    https://www.ers.usda.gov/data-products/food-access-research-atlas/download-the-data/
  CDC PLACES 2023 (health outcomes at census-tract level):
    https://data.cdc.gov/api/views/cwsq-ngmh/rows.csv?accessType=DOWNLOAD
  AHRQ SDOH Database 2021:
    https://www.ahrq.gov/sdoh/data-analytics/sdoh-data.html

Usage
-----
  python sdoh_equity.py --svi_csv data/sdoh/SVI2022_US_county.csv \
                         --cohort  data/nhanes/nhanes_2017_2020_merged.csv \
                         --outdir  data/sdoh_equity
"""

import argparse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# ─────────────────────────────────────────────────────────────
# SVI download helper
# ─────────────────────────────────────────────────────────────
SVI_URL = (
    "https://www.atsdr.cdc.gov/placeandhealth/svi/data_documentation_download.html"
    # NOTE: Direct CSV download requires manual acceptance at the URL above.
    # After downloading, place SVI2022_US_county.csv in data/sdoh/
)

FOOD_DESERT_URL = (
    "https://www.ers.usda.gov/webdocs/DataFiles/80591/"
    "FoodAccessResearchAtlasData2019.xlsx?v=8368.7"
)

CDC_PLACES_URL = (
    "https://data.cdc.gov/api/views/cwsq-ngmh/rows.csv?accessType=DOWNLOAD"
)


def download_cdc_places(dest_dir: Path) -> Path:
    """Download CDC PLACES 2023 health outcomes (auto-downloadable)."""
    out = dest_dir / "cdc_places_2023.csv"
    if out.exists():
        print(f"  [cache] {out}")
        return out
    print("  [download] CDC PLACES 2023 …")
    urllib.request.urlretrieve(CDC_PLACES_URL, out)
    return out


# ─────────────────────────────────────────────────────────────
# SVI variable definitions (CDC SVI 2022 codebook)
# ─────────────────────────────────────────────────────────────
SVI_THEMES = {
    "RPL_THEME1": "Socioeconomic Status",
    "RPL_THEME2": "Household Characteristics",
    "RPL_THEME3": "Racial & Ethnic Minority Status",
    "RPL_THEME4": "Housing Type & Transportation",
    "RPL_THEMES": "Overall SVI (composite)",
}

SVI_KEY_VARS = [
    "FIPS", "STATE", "ST_ABBR", "COUNTY",
    "RPL_THEME1", "RPL_THEME2", "RPL_THEME3", "RPL_THEME4", "RPL_THEMES",
    "EP_POV150",   # % persons below 150% poverty line
    "EP_UNEMP",    # % civilian unemployed
    "EP_NOHSDP",   # % no high school diploma (≥25 y)
    "EP_UNINSUR",  # % uninsured
    "EP_MINRTY",   # % racial/ethnic minority
    "EP_LIMENG",   # % speaking English <well
    "EP_MOBILE",   # % mobile homes
    "EP_CROWD",    # % housing ≥1 person/room
    "E_TOTPOP",    # Total population
]


def load_svi(svi_csv: str) -> pd.DataFrame:
    """Load CDC SVI county CSV, keep key variables."""
    df = pd.read_csv(svi_csv, dtype={"FIPS": str})
    present = [c for c in SVI_KEY_VARS if c in df.columns]
    df = df[present].copy()
    # Replace -999 (missing) with NaN
    df.replace(-999, np.nan, inplace=True)
    print(f"  [SVI] Loaded {len(df):,} counties × {df.shape[1]} vars")
    return df


# ─────────────────────────────────────────────────────────────
# Food desert index (USDA)
# ─────────────────────────────────────────────────────────────

def load_food_desert(excel_path: str) -> pd.DataFrame:
    """
    Load USDA Food Access Research Atlas.
    Key variable: LILATracts_1And10 (1 = food desert tract)
    """
    df = pd.read_excel(excel_path, sheet_name="Food Access Research Atlas",
                       dtype={"CensusTract": str})
    df["state_fips"] = df["CensusTract"].str[:2]
    df["county_fips"] = df["CensusTract"].str[:5]
    food_desert_by_county = (
        df.groupby("county_fips")["LILATracts_1And10"]
        .mean()
        .reset_index()
        .rename(columns={"LILATracts_1And10": "food_desert_pct", "county_fips": "FIPS"})
    )
    return food_desert_by_county


# ─────────────────────────────────────────────────────────────
# Equity stratification
# ─────────────────────────────────────────────────────────────

def stratify_risk(cohort_df: pd.DataFrame,
                  risk_col: str,
                  strata_col: str,
                  label_col: str = None) -> pd.DataFrame:
    """
    Compute mean risk score (and optionally AUC) by stratum.

    Parameters
    ----------
    cohort_df : DataFrame with risk scores and stratification variable
    risk_col  : column name for the continuous risk score
    strata_col: column to stratify by (e.g. 'race_eth_label', 'income_group')
    label_col : optional binary outcome column for per-stratum AUC
    """
    groups = cohort_df.groupby(strata_col)
    rows   = []
    for name, grp in groups:
        row = {
            strata_col:          name,
            "n":                 len(grp),
            f"mean_{risk_col}":  grp[risk_col].mean(),
            f"std_{risk_col}":   grp[risk_col].std(),
            f"p75_{risk_col}":   grp[risk_col].quantile(0.75),
        }
        if label_col and label_col in grp and grp[label_col].nunique() == 2:
            try:
                row["auc"] = roc_auc_score(grp[label_col], grp[risk_col])
            except Exception:
                row["auc"] = np.nan
        rows.append(row)
    result = pd.DataFrame(rows).sort_values(f"mean_{risk_col}", ascending=False)
    return result


# ─────────────────────────────────────────────────────────────
# Algorithmic fairness metrics
# ─────────────────────────────────────────────────────────────

def fairness_diagnostics(cohort_df: pd.DataFrame,
                          risk_col: str,
                          group_col: str,
                          threshold: float = None,
                          reference_group: str = "NH-White") -> pd.DataFrame:
    """
    Compute fairness metrics across demographic groups.

    Metrics:
      - Demographic parity difference  : |P(high-risk|group) - P(high-risk|ref)|
      - Mean risk ratio                : mean_risk(group) / mean_risk(ref)
      - Equalized odds proxy           : ratio of false-positive rates (if label available)
    """
    if threshold is None:
        threshold = cohort_df[risk_col].quantile(0.75)  # top quartile = high risk

    ref_rate = (cohort_df.loc[cohort_df[group_col] == reference_group, risk_col] >= threshold).mean()
    ref_mean = cohort_df.loc[cohort_df[group_col] == reference_group, risk_col].mean()

    rows = []
    for grp, sub in cohort_df.groupby(group_col):
        grp_rate = (sub[risk_col] >= threshold).mean()
        grp_mean = sub[risk_col].mean()
        rows.append({
            "group":                     grp,
            "n":                         len(sub),
            "high_risk_rate":            grp_rate,
            "demographic_parity_diff":   abs(grp_rate - ref_rate),
            "mean_risk":                 grp_mean,
            "mean_risk_ratio":           grp_mean / ref_mean if ref_mean > 0 else np.nan,
        })
    return pd.DataFrame(rows).sort_values("mean_risk", ascending=False)


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SDOH Equity Analysis — UPF-NeuroCog v2.0")
    parser.add_argument("--svi_csv",     required=True, help="Path to SVI2022_US_county.csv")
    parser.add_argument("--cohort",      required=True, help="NHANES or BRFSS merged CSV with risk_score column")
    parser.add_argument("--food_desert", default=None,  help="USDA Food Access Atlas Excel (optional)")
    parser.add_argument("--outdir",      default="data/sdoh_equity")
    parser.add_argument("--risk_col",    default="stress_score")
    parser.add_argument("--group_col",   default="race_eth_label")
    args = parser.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    svi    = load_svi(args.svi_csv)
    cohort = pd.read_csv(args.cohort)
    print(f"  [cohort] n={len(cohort):,}")

    # Merge SVI by state if county FIPS unavailable
    if "FIPS" in cohort.columns and "FIPS" in svi.columns:
        cohort = cohort.merge(svi, on="FIPS", how="left")
    elif "_STATE" in cohort.columns:
        state_svi = svi.groupby("STATE")[["RPL_THEMES", "EP_POV150", "EP_MINRTY"]].mean().reset_index()
        cohort = cohort.merge(state_svi, left_on="_STATE", right_on="STATE", how="left")

    if args.food_desert:
        fd = load_food_desert(args.food_desert)
        if "FIPS" in cohort.columns:
            cohort = cohort.merge(fd, on="FIPS", how="left")

    # Stratified risk analysis
    if args.risk_col in cohort.columns and args.group_col in cohort.columns:
        strat_df = stratify_risk(cohort, args.risk_col, args.group_col)
        fair_df  = fairness_diagnostics(cohort, args.risk_col, args.group_col)

        strat_path = out / f"risk_by_{args.group_col}.csv"
        fair_path  = out / f"fairness_{args.group_col}.csv"
        strat_df.to_csv(strat_path, index=False)
        fair_df.to_csv(fair_path,   index=False)

        print(f"\n[sdoh_equity] Stratified risk saved → {strat_path}")
        print(strat_df.to_string(index=False))
        print(f"\n[sdoh_equity] Fairness diagnostics → {fair_path}")
        print(fair_df.to_string(index=False))
    else:
        print(f"Risk column '{args.risk_col}' or group column '{args.group_col}' not found.")

    cohort.to_csv(out / "cohort_with_sdoh.csv", index=False)
    print(f"\n[sdoh_equity] Cohort + SVI merged → {out / 'cohort_with_sdoh.csv'}")


if __name__ == "__main__":
    main()
