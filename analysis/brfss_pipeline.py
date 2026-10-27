"""
brfss_pipeline.py  —  UPF-NeuroCog v2.0
=========================================
Downloads BRFSS 2023 (n≈433,000), extracts mental health,
stress, dietary behaviour, and SDOH covariates, then
produces a stress-behaviour feature matrix.

Real dataset URL:
  BRFSS 2023 combined XPT : https://www.cdc.gov/brfss/annual_data/2023/files/LLCP2023XPT.zip
  BRFSS 2023 codebook     : https://www.cdc.gov/brfss/annual_data/2023/pdf/codebook23_llcp-v2-508.pdf
  BRFSS 2023 SAS format   : https://www.cdc.gov/brfss/annual_data/2023/files/LLCP2023SAS.zip

Usage
-----
  python brfss_pipeline.py --outdir data/brfss
"""

import argparse
import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadstat

# ─────────────────────────────────────────────────────────────
# Download
# ─────────────────────────────────────────────────────────────
BRFSS_2023_ZIP = "https://www.cdc.gov/brfss/annual_data/2023/files/LLCP2023XPT.zip"
BRFSS_2023_XPT = "LLCP2023.XPT"


def download_brfss(dest_dir: Path) -> Path:
    """Download and unzip BRFSS 2023 XPT if not already cached."""
    xpt_path = dest_dir / BRFSS_2023_XPT
    if xpt_path.exists():
        print(f"  [cache] {xpt_path}")
        return xpt_path

    zip_path = dest_dir / "LLCP2023XPT.zip"
    print(f"  [download] BRFSS 2023 — this may take a few minutes (~1 GB compressed)…")
    urllib.request.urlretrieve(BRFSS_2023_ZIP, zip_path)
    print(f"  [unzip] {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)
    assert xpt_path.exists(), f"Expected {BRFSS_2023_XPT} after unzip"
    return xpt_path


# ─────────────────────────────────────────────────────────────
# Variable manifest  (BRFSS 2023 codebook)
# ─────────────────────────────────────────────────────────────
# Mental health & stress
MENTAL_VARS = [
    "MENTHLTH",   # Days mental health not good in past 30 days (1-30, 88=none, 77=DK, 99=refused)
    "POORHLTH",   # Days poor physical or mental health (same scale)
    "PHYSHLTH",   # Days physical health not good
    "DECIDE",     # Difficulty concentrating / remembering (1=Yes, 2=No)
    "DIFFALON",   # Difficulty doing errands alone (1=Yes, 2=No)
]

# Dietary behavior
DIETARY_VARS = [
    "FRUIT2",     # Fruit consumption times per day/week/month
    "FVGREEN",    # Green vegetables times per day/week/month
    "FVORANG",    # Orange vegetables times per day/week/month
    "VEGETAB2",   # Vegetable consumption times per day/week/month
    "_FRTLT1A",   # Consumed fruit < 1/day (calculated, 1=Yes, 2=No)
    "_VEGLT1A",   # Consumed vegetables < 1/day (calculated)
    "_FRUTSUM",   # Total fruit computed daily intake
    "_VEGESUM",   # Total vegetable computed daily intake
]

# Substance / behavioural
BEHAV_VARS = [
    "_RFBING5",   # Binge drinking indicator (1=No, 2=Yes)
    "SMOKE100",   # Smoked ≥100 cigarettes lifetime (1=Yes, 2=No)
    "_RFSMOK3",   # Current smoker (1=No, 2=Yes)
    "EXERANY2",   # Exercise in past 30 days (1=Yes, 2=No)
    "_TOTINDA",   # No leisure-time physical activity (1=Had activity, 2=No activity)
]

# Sociodemographic
DEMO_VARS = [
    "_STATE",     # State FIPS code
    "_AGE_G",     # Age group (1=18-24 … 6=65+)
    "SEX1",       # Sex (1=Male, 2=Female)
    "_RACE",      # Race/ethnicity (1=White, 2=Black, 3=AI/AN, 4=Asian, 5=PI, 6=Other, 7=Multi, 8=Hisp, 9=DK)
    "_EDUCAG",    # Education group (1=<HS, 2=HS grad, 3=Some college, 4=College grad)
    "_INCOMG1",   # Income group (1=<15K … 8=≥75K)
    "_BMI5CAT",   # BMI category (1=Under, 2=Normal, 3=Overweight, 4=Obese)
    "EMPLOY1",    # Employment status (1=Employed, 2=Self-employed, 3=Out of work, 4=Homemaker, 5=Student, 6=Retired, 7=Unable, 9=Refused)
    "HLTHPLN1",   # Health plan coverage (1=Yes, 2=No)
    "_LLCPWT",    # Final design weight
]

ALL_VARS = list(set(MENTAL_VARS + DIETARY_VARS + BEHAV_VARS + DEMO_VARS + ["_PSU", "_STSTR"]))


# ─────────────────────────────────────────────────────────────
# Load & clean
# ─────────────────────────────────────────────────────────────

def load_brfss(xpt_path: Path) -> pd.DataFrame:
    """Load BRFSS XPT and keep only analysis variables."""
    print(f"  [load] Reading {xpt_path} (n≈433,000) …")
    df, meta = pyreadstat.read_xport(str(xpt_path))
    present  = [c for c in ALL_VARS if c in df.columns]
    df = df[present].copy()
    print(f"  Loaded {len(df):,} rows × {df.shape[1]} cols")
    return df


def clean_brfss(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recode BRFSS variables to analysis-ready numeric fields.

    MENTHLTH: 1-30 days → keep; 88 → 0; 77/99 → NaN
    POORHLTH: same
    DECIDE, DIFFALON: 1=Yes→1, 2=No→0; others NaN
    _RFBING5: 1→0 (no binge), 2→1 (binge); 9→NaN
    EXERANY2: 1→1 (exercised), 2→0; 7/9→NaN
    """
    out = df.copy()

    for col in ["MENTHLTH", "POORHLTH", "PHYSHLTH"]:
        if col in out:
            out[col] = pd.to_numeric(out[col], errors="coerce")
            out[col] = out[col].replace(88, 0)
            out[col] = out[col].where(out[col].between(0, 30))

    for col in ["DECIDE", "DIFFALON"]:
        if col in out:
            out[col] = pd.to_numeric(out[col], errors="coerce")
            out[col] = out[col].map({1: 1, 2: 0})

    if "_RFBING5" in out:
        out["binge_drink"] = pd.to_numeric(out["_RFBING5"], errors="coerce").map({1: 0, 2: 1})

    if "EXERANY2" in out:
        out["any_exercise"] = pd.to_numeric(out["EXERANY2"], errors="coerce").map({1: 1, 2: 0})

    if "SMOKE100" in out:
        out["ever_smoked"] = pd.to_numeric(out["SMOKE100"], errors="coerce").map({1: 1, 2: 0})

    if "SEX1" in out:
        out["is_female"] = pd.to_numeric(out["SEX1"], errors="coerce").map({2: 1, 1: 0})

    if "EMPLOY1" in out:
        out["employed"] = pd.to_numeric(out["EMPLOY1"], errors="coerce").isin([1, 2]).astype(float)

    return out


# ─────────────────────────────────────────────────────────────
# Stress-behaviour composite score
# ─────────────────────────────────────────────────────────────

def compute_stress_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Composite stress-behaviour score (0–100) from validated proxies.

    Components (each normalised 0-1, then averaged × 100):
      1. Psychological distress  : MENTHLTH / 30
      2. Poor health days ratio  : POORHLTH / 30
      3. Cognitive difficulty    : DECIDE
      4. Physical inactivity     : 1 - any_exercise
      5. Binge drinking          : binge_drink
      6. Diet quality (inverse)  : 1 - fruit_veg_adequate
    """
    components = {}

    if "MENTHLTH" in df:
        components["psych_distress"]  = df["MENTHLTH"].clip(0, 30) / 30
    if "POORHLTH" in df:
        components["poor_health"]     = df["POORHLTH"].clip(0, 30) / 30
    if "DECIDE" in df:
        components["cognitive_diff"]  = df["DECIDE"].fillna(0)
    if "any_exercise" in df:
        components["inactivity"]      = 1 - df["any_exercise"].fillna(0.5)
    if "binge_drink" in df:
        components["binge_drink"]     = df["binge_drink"].fillna(0)

    # Diet adequacy proxy: consuming ≥1 fruit AND ≥1 veg per day
    if "_FRTLT1A" in df and "_VEGLT1A" in df:
        frt = pd.to_numeric(df["_FRTLT1A"], errors="coerce").map({1: 1, 2: 0}).fillna(0.5)
        veg = pd.to_numeric(df["_VEGLT1A"], errors="coerce").map({1: 1, 2: 0}).fillna(0.5)
        components["poor_diet"]       = ((frt + veg) / 2)

    if not components:
        raise ValueError("No stress-score components found in dataframe.")

    comp_df = pd.DataFrame(components)
    df["stress_score"] = comp_df.mean(axis=1) * 100
    df["stress_score_n_components"] = comp_df.notna().sum(axis=1)

    return df


# ─────────────────────────────────────────────────────────────
# Stratification helpers for SDOH integration
# ─────────────────────────────────────────────────────────────

def add_sdoh_strata(df: pd.DataFrame) -> pd.DataFrame:
    """Add categorical SDOH strata columns for equity analysis."""
    if "_RACE" in df:
        race_map = {1: "NH-White", 2: "NH-Black", 3: "AI-AN", 4: "Asian",
                    5: "Pacific-Islander", 6: "Other", 7: "Multiracial", 8: "Hispanic"}
        df["race_eth_label"] = pd.to_numeric(df["_RACE"], errors="coerce").map(race_map)

    if "_INCOMG1" in df:
        df["income_group"] = pd.to_numeric(df["_INCOMG1"], errors="coerce")
        df["low_income"]   = (df["income_group"] <= 3).astype(float)

    if "_EDUCAG" in df:
        df["edu_group"] = pd.to_numeric(df["_EDUCAG"], errors="coerce")
        df["low_edu"]   = (df["edu_group"] <= 2).astype(float)

    if "_BMI5CAT" in df:
        df["obese"] = (pd.to_numeric(df["_BMI5CAT"], errors="coerce") == 4).astype(float)

    return df


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="BRFSS 2023 Pipeline — UPF-NeuroCog v2.0")
    parser.add_argument("--outdir", default="data/brfss")
    parser.add_argument("--skip_download", action="store_true",
                        help="Skip download (use if XPT already present)")
    args = parser.parse_args()

    dest = Path(args.outdir)
    dest.mkdir(parents=True, exist_ok=True)

    if not args.skip_download:
        xpt_path = download_brfss(dest)
    else:
        xpt_path = dest / BRFSS_2023_XPT
        assert xpt_path.exists(), f"XPT not found at {xpt_path}"

    df = load_brfss(xpt_path)
    df = clean_brfss(df)
    df = compute_stress_score(df)
    df = add_sdoh_strata(df)

    out_path = dest / "brfss_2023_stress_features.csv"
    df.to_csv(out_path, index=False)
    print(f"\n[brfss_pipeline] Saved → {out_path}")
    print(f"  n = {len(df):,}")
    print(f"\nStress score summary:\n{df['stress_score'].describe().round(2)}")
    print(f"\nRace distribution:\n{df['race_eth_label'].value_counts().head(10)}")


if __name__ == "__main__":
    main()
