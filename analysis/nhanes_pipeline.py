"""
nhanes_pipeline.py  —  UPF-NeuroCog v2.0
==========================================
Downloads NHANES 2017-2018 and 2019-March 2020 XPT files,
merges dietary recall, demographics, BMI, physical activity,
lab biomarkers, and applies NOVA classification.

Real dataset URLs (all public CDC NHANES):
  2017-2018 Dietary Total   : https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/DR1TOT_J.XPT
  2017-2018 Individual Foods: https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/DR1IFF_J.XPT
  2017-2018 Demographics    : https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/DEMO_J.XPT
  2017-2018 BMI             : https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/BMX_J.XPT
  2017-2018 Physical Act    : https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/PAQ_J.XPT
  2017-2018 HbA1c           : https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/GHB_J.XPT
  2017-2018 Blood Pressure  : https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/BPX_J.XPT
  2019-2020 Dietary Total   : https://wwwn.cdc.gov/Nchs/Nhanes/2019-2020/DR1TOT_P.XPT
  2019-2020 Individual Foods: https://wwwn.cdc.gov/Nchs/Nhanes/2019-2020/DR1IFF_P.XPT
  2019-2020 Demographics    : https://wwwn.cdc.gov/Nchs/Nhanes/2019-2020/DEMO_P.XPT
  2019-2020 BMI             : https://wwwn.cdc.gov/Nchs/Nhanes/2019-2020/BMX_P.XPT
  2019-2020 Physical Act    : https://wwwn.cdc.gov/Nchs/Nhanes/2019-2020/PAQ_P.XPT
  2019-2020 HbA1c           : https://wwwn.cdc.gov/Nchs/Nhanes/2019-2020/GHB_P.XPT
  2019-2020 Blood Pressure  : https://wwwn.cdc.gov/Nchs/Nhanes/2019-2020/BPXO_P.XPT
  USDA FNDDS food codes     : https://www.ars.usda.gov/ARSUserFiles/80400530/apps/2017-2018%20FNDDS%20At%20A%20Glance.xlsx

Usage
-----
  python nhanes_pipeline.py --outdir data/nhanes --cycles 2017 2019
"""

import argparse
import os
from pathlib import Path

import pandas as pd
import pyreadstat

# ─────────────────────────────────────────────────────────────
# Dataset manifests per cycle
# ─────────────────────────────────────────────────────────────
NHANES_URLS = {
    "2017": {
        "dietary_total":    "https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/DR1TOT_J.XPT",
        "dietary_indiv":    "https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/DR1IFF_J.XPT",
        "demographics":     "https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/DEMO_J.XPT",
        "bmi":              "https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/BMX_J.XPT",
        "physical_activity":"https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/PAQ_J.XPT",
        "hba1c":            "https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/GHB_J.XPT",
        "blood_pressure":   "https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/BPX_J.XPT",
    },
    "2019": {
        "dietary_total":    "https://wwwn.cdc.gov/Nchs/Nhanes/2019-2020/DR1TOT_P.XPT",
        "dietary_indiv":    "https://wwwn.cdc.gov/Nchs/Nhanes/2019-2020/DR1IFF_P.XPT",
        "demographics":     "https://wwwn.cdc.gov/Nchs/Nhanes/2019-2020/DEMO_P.XPT",
        "bmi":              "https://wwwn.cdc.gov/Nchs/Nhanes/2019-2020/BMX_P.XPT",
        "physical_activity":"https://wwwn.cdc.gov/Nchs/Nhanes/2019-2020/PAQ_P.XPT",
        "hba1c":            "https://wwwn.cdc.gov/Nchs/Nhanes/2019-2020/GHB_P.XPT",
        "blood_pressure":   "https://wwwn.cdc.gov/Nchs/Nhanes/2019-2020/BPXO_P.XPT",
    },
}

# Key variable mappings
DIETARY_VARS = ["SEQN", "DR1TKCAL", "DR1TPROT", "DR1TCARB",
                "DR1TSUGR", "DR1TFIBE", "DR1TTFAT", "DR1TSODI"]
DEMO_VARS    = ["SEQN", "RIDAGEYR", "RIAGENDR", "RIDRETH3",
                "DMDEDUC2", "INDFMPIR", "WTDRD1"]
BMI_VARS     = ["SEQN", "BMXBMI", "BMXWT", "BMXHT"]
PAQ_VARS     = ["SEQN", "PAQ605", "PAQ620", "PAD660"]
HBA1C_VARS   = ["SEQN", "LBXGH"]
BP_VARS      = ["SEQN", "BPXSY1", "BPXDI1"]

# ─────────────────────────────────────────────────────────────
# Download helpers
# ─────────────────────────────────────────────────────────────

def download_xpt(url: str, dest_path: Path) -> Path:
    """Download a CDC NHANES XPT file if not already cached."""
    if dest_path.exists():
        print(f"  [cache] {dest_path.name}")
        return dest_path
    print(f"  [download] {url}")
    import urllib.request
    urllib.request.urlretrieve(url, dest_path)
    assert dest_path.stat().st_size > 0, f"Empty file: {dest_path}"
    return dest_path


def load_xpt(path: Path, keep_cols: list = None) -> pd.DataFrame:
    """Read SAS XPT file; optionally keep only specified columns."""
    df, _ = pyreadstat.read_xport(str(path))
    if keep_cols:
        present = [c for c in keep_cols if c in df.columns]
        df = df[present]
    return df


# ─────────────────────────────────────────────────────────────
# Per-cycle pipeline
# ─────────────────────────────────────────────────────────────

def process_cycle(cycle_year: str, raw_dir: Path) -> pd.DataFrame:
    """Download and merge all NHANES modules for one cycle."""
    urls = NHANES_URLS[cycle_year]
    cycle_dir = raw_dir / f"nhanes_{cycle_year}"
    cycle_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== NHANES {cycle_year}-{int(cycle_year)+1} ===")

    # Download all modules
    paths = {}
    for module, url in urls.items():
        fname = Path(url).name
        paths[module] = download_xpt(url, cycle_dir / fname)

    # Load and subset
    dietary = load_xpt(paths["dietary_total"],    DIETARY_VARS)
    demo    = load_xpt(paths["demographics"],     DEMO_VARS)
    bmi     = load_xpt(paths["bmi"],              BMI_VARS)
    paq     = load_xpt(paths["physical_activity"],PAQ_VARS)
    hba1c   = load_xpt(paths["hba1c"],            HBA1C_VARS)
    bp      = load_xpt(paths["blood_pressure"],   BP_VARS)

    # Merge on SEQN (participant ID)
    df = (dietary
          .merge(demo,   on="SEQN", how="left")
          .merge(bmi,    on="SEQN", how="left")
          .merge(paq,    on="SEQN", how="left")
          .merge(hba1c,  on="SEQN", how="left")
          .merge(bp,     on="SEQN", how="left"))

    df["cycle"] = cycle_year
    print(f"  Merged n={len(df):,} participants, {df.shape[1]} variables")
    return df


# ─────────────────────────────────────────────────────────────
# NOVA energy share computation
# ─────────────────────────────────────────────────────────────

def compute_upf_energy_share(individual_foods_path: Path,
                              nova_classified_csv: str) -> pd.DataFrame:
    """
    Compute each participant's UPF energy share (%).

    Requires the individual food file merged with NOVA classifications.
    nova_classified_csv must have columns: DR1IFDCD, nova_group
    """
    indiv, _ = pyreadstat.read_xport(str(individual_foods_path))
    nova_df   = pd.read_csv(nova_classified_csv)[["DR1IFDCD", "nova_group"]]

    indiv = indiv.merge(nova_df, on="DR1IFDCD", how="left")
    indiv["is_upf"] = indiv["nova_group"] == 4

    per_person = indiv.groupby("SEQN").apply(
        lambda x: pd.Series({
            "total_kcal_foods":  x["DR1IKCAL"].sum(),
            "upf_kcal":          x.loc[x["is_upf"], "DR1IKCAL"].sum(),
        })
    ).reset_index()

    per_person["upf_energy_pct"] = (
        100 * per_person["upf_kcal"] / per_person["total_kcal_foods"].replace(0, pd.NA)
    )
    return per_person[["SEQN", "upf_energy_pct", "upf_kcal", "total_kcal_foods"]]


# ─────────────────────────────────────────────────────────────
# Feature engineering
# ─────────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create analysis-ready features from raw NHANES variables.

    Variable rename map (NHANES codebook → readable names):
      RIDAGEYR  → age
      RIAGENDR  → sex (1=Male, 2=Female)
      RIDRETH3  → race_eth (1=Mex-Am, 2=Hisp, 3=NH-White, 4=NH-Black, 6=NH-Asian, 7=Multi)
      DMDEDUC2  → education (1-5 scale)
      INDFMPIR  → poverty_ratio (family income / poverty threshold)
      BMXBMI    → bmi
      LBXGH     → hba1c_pct
      DR1TKCAL  → total_energy_kcal
      DR1TSODI  → sodium_mg
      DR1TSUGR  → sugar_g
      DR1TPROT  → protein_g
      DR1TFIBE  → fiber_g
      PAQ605    → vigorous_activity (1=Yes, 2=No)
      BPXSY1    → sbp (systolic BP mmHg)
    """
    rename = {
        "RIDAGEYR": "age", "RIAGENDR": "sex", "RIDRETH3": "race_eth",
        "DMDEDUC2": "education", "INDFMPIR": "poverty_ratio",
        "BMXBMI":   "bmi",  "LBXGH": "hba1c_pct",
        "DR1TKCAL": "total_energy_kcal", "DR1TSODI": "sodium_mg",
        "DR1TSUGR": "sugar_g", "DR1TPROT": "protein_g",
        "DR1TFIBE": "fiber_g", "PAQ605": "vigorous_activity",
        "BPXSY1":   "sbp",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    # Derived features
    df["is_female"]          = (df["sex"] == 2).astype(int)
    df["low_income"]         = (df["poverty_ratio"] < 1.85).astype(int)
    df["obese"]              = (df["bmi"] >= 30).astype(int)
    df["hyperglycemia"]      = (df["hba1c_pct"] >= 5.7).astype(int)
    df["sodium_high"]        = (df["sodium_mg"] > 2300).astype(int)
    df["fiber_low"]          = (df["fiber_g"] < 25).astype(int)
    df["age_group"]          = pd.cut(
        df["age"], bins=[17, 34, 49, 64, 120],
        labels=["18-34", "35-49", "50-64", "65+"]
    )

    return df


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="NHANES 2017-2020 Pipeline — UPF-NeuroCog v2.0")
    parser.add_argument("--outdir",  default="data/nhanes",  help="Output directory for raw + merged data")
    parser.add_argument("--cycles",  nargs="+", default=["2017", "2019"], choices=["2017", "2019"])
    parser.add_argument("--nova_csv", default=None,
                        help="Pre-computed NOVA CSV (DR1IFDCD, nova_group) to add UPF energy share")
    args = parser.parse_args()

    raw_dir = Path(args.outdir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for cycle in args.cycles:
        frames.append(process_cycle(cycle, raw_dir))

    combined = pd.concat(frames, ignore_index=True)
    combined = engineer_features(combined)

    if args.nova_csv:
        # Merge UPF energy share for first cycle (extend as needed)
        indiv_path = raw_dir / "nhanes_2017" / "DR1IFF_J.XPT"
        upf_share  = compute_upf_energy_share(indiv_path, args.nova_csv)
        combined   = combined.merge(upf_share, on="SEQN", how="left")
        print(f"\nUPF energy share stats:\n{combined['upf_energy_pct'].describe().round(2)}")

    out_path = raw_dir / "nhanes_2017_2020_merged.csv"
    combined.to_csv(out_path, index=False)
    print(f"\n[nhanes_pipeline] Saved merged dataset → {out_path}")
    print(f"  Total participants : {len(combined):,}")
    print(f"  Variables          : {combined.shape[1]}")
    print(combined.dtypes.value_counts().to_string())


if __name__ == "__main__":
    main()
