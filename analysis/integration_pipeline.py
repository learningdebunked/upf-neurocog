"""
integration_pipeline.py  —  UPF-NeuroCog v2.0
==================================================
Master pipeline: merges NHANES (2017-2020) + BRFSS (2023) +
SDOH (CDC SVI 2022, USDA Food Desert Atlas, CDC PLACES) +
DEAP EEG features into a single unified cohort ready for
model training.

Outputs
-------
data/integrated_cohort.parquet   — full merged cohort (row-level)
data/eeg_feature_summary.csv     — DEAP-derived neural biomarker summary
data/integration_report.json     — cohort statistics / QC report

Usage
-----
    python analysis/integration_pipeline.py \
        --nhanes_dir  data/nhanes/ \
        --brfss_file  data/brfss/LLCP2023.XPT \
        --sdoh_dir    data/sdoh/ \
        --deap_dir    data/deap/ \
        --output_dir  data/
"""

import os
import json
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1.  Import modular pipelines
# ---------------------------------------------------------------------------
# Each pipeline is importable as a library module.
# They expose `run(...)` callables that return a cleaned DataFrame.

from nhanes_pipeline import run as run_nhanes
from brfss_pipeline import run as run_brfss
from sdoh_equity import run as run_sdoh
from deap_eeg_pipeline import run as run_deap


# ---------------------------------------------------------------------------
# 2.  Helpers
# ---------------------------------------------------------------------------

def standardise_age(df: pd.DataFrame, col: str = "age_years") -> pd.DataFrame:
    """Min-max normalise age into [0,1]."""
    df[col + "_norm"] = (df[col] - df[col].min()) / (df[col].max() - df[col].min() + 1e-9)
    return df


def compute_upf_tertile(df: pd.DataFrame, col: str = "upf_energy_pct") -> pd.DataFrame:
    """Label rows as Low / Mid / High UPF tertile."""
    df["upf_tertile"] = pd.qcut(df[col], q=3, labels=["Low", "Mid", "High"])
    return df


def composite_cognitive_risk(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a 0-1 composite cognitive-fatigue risk score from available proxies:
      - BRFSS poor-mental-health days (brfss_poor_mh_days) — normalised
      - Sleep insufficiency flag (brfss_sleep_insufficient)
      - Physical inactivity flag (brfss_inactive)
      - UPF energy share (upf_energy_pct) — normalised

    Weights based on literature (see manuscript §3.3):
        mh_days : 0.35
        sleep   : 0.25
        activity: 0.20
        upf     : 0.20
    """
    w = {"mh": 0.35, "sleep": 0.25, "activity": 0.20, "upf": 0.20}

    mh_norm = df.get("brfss_poor_mh_days", pd.Series(0.0, index=df.index))
    mh_norm = mh_norm / 30.0  # 30-day reference window

    sleep_flag = df.get("brfss_sleep_insufficient", pd.Series(0.0, index=df.index)).fillna(0)
    inactive_flag = df.get("brfss_inactive", pd.Series(0.0, index=df.index)).fillna(0)
    upf_norm = df.get("upf_energy_pct", pd.Series(0.0, index=df.index)) / 100.0

    df["cognitive_risk_score"] = (
        w["mh"] * mh_norm
        + w["sleep"] * sleep_flag
        + w["activity"] * inactive_flag
        + w["upf"] * upf_norm
    ).clip(0, 1)

    df["cognitive_risk_label"] = (df["cognitive_risk_score"] >= 0.50).astype(int)
    return df


# ---------------------------------------------------------------------------
# 3.  DEAP EEG neural feature mapping
# ---------------------------------------------------------------------------

def attach_eeg_archetypes(population_df: pd.DataFrame,
                          deap_summary: pd.DataFrame) -> pd.DataFrame:
    """
    DEAP is a 32-subject lab cohort — not linkable row-by-row to NHANES/BRFSS.
    Instead we compute mean EEG feature profiles per arousal/valence quadrant
    and assign the nearest archetype to each population row based on its
    cognitive risk score.

    Arousal quadrants (valence × arousal binary splits):
        Q1  High-Arousal / High-Valence  → "Engaged"
        Q2  High-Arousal / Low-Valence   → "Stressed"   ← high-risk proxy
        Q3  Low-Arousal  / Low-Valence   → "Exhausted"  ← high-risk proxy
        Q4  Low-Arousal  / High-Valence  → "Calm"
    """
    if deap_summary is None or deap_summary.empty:
        logger.warning("DEAP summary empty — skipping EEG archetype mapping.")
        for col in ["eeg_archetype", "eeg_pac_mi", "eeg_frontal_alpha_asym"]:
            population_df[col] = np.nan
        return population_df

    # Build quadrant profiles
    q_profiles = (
        deap_summary
        .groupby("arousal_valence_quad")[["pac_mi_theta_gamma", "frontal_alpha_asym"]]
        .mean()
        .rename(columns={"pac_mi_theta_gamma": "mean_pac_mi",
                         "frontal_alpha_asym": "mean_faa"})
    )

    def assign_quad(risk_score: float) -> str:
        if risk_score >= 0.65:
            return "Exhausted"
        elif risk_score >= 0.50:
            return "Stressed"
        elif risk_score >= 0.30:
            return "Engaged"
        else:
            return "Calm"

    population_df["eeg_archetype"] = population_df["cognitive_risk_score"].apply(assign_quad)

    # Merge mean EEG features per archetype
    quad_map = q_profiles.to_dict(orient="index")
    population_df["eeg_pac_mi"] = population_df["eeg_archetype"].map(
        lambda q: quad_map.get(q, {}).get("mean_pac_mi", np.nan)
    )
    population_df["eeg_frontal_alpha_asym"] = population_df["eeg_archetype"].map(
        lambda q: quad_map.get(q, {}).get("mean_faa", np.nan)
    )
    return population_df


# ---------------------------------------------------------------------------
# 4.  SDOH linkage
# ---------------------------------------------------------------------------

def attach_sdoh_flags(population_df: pd.DataFrame,
                      sdoh_df: pd.DataFrame) -> pd.DataFrame:
    """
    Link SDOH risk flags using FIPS county code where available, else
    impute with national quantile thresholds.
    """
    if sdoh_df is None or sdoh_df.empty:
        logger.warning("SDOH DataFrame empty — appending NaN SDOH columns.")
        for col in ["svi_percentile", "food_desert_flag", "places_mental_health_poor_pct"]:
            population_df[col] = np.nan
        return population_df

    fips_col = next((c for c in population_df.columns if "fips" in c.lower()), None)
    if fips_col and "fips" in sdoh_df.columns:
        population_df = population_df.merge(
            sdoh_df[["fips", "svi_percentile", "food_desert_flag",
                     "places_mental_health_poor_pct"]],
            left_on=fips_col, right_on="fips", how="left"
        )
    else:
        # National medians as fallback
        for col in ["svi_percentile", "food_desert_flag", "places_mental_health_poor_pct"]:
            if col in sdoh_df.columns:
                population_df[col] = sdoh_df[col].median()
            else:
                population_df[col] = np.nan

    return population_df


# ---------------------------------------------------------------------------
# 5.  QC Report
# ---------------------------------------------------------------------------

def build_qc_report(df: pd.DataFrame, source_counts: dict) -> dict:
    """Generate a JSON-serialisable QC report."""
    report = {
        "total_rows": int(len(df)),
        "source_counts": source_counts,
        "missing_pct": {
            col: round(df[col].isna().mean() * 100, 2)
            for col in df.columns
            if df[col].isna().any()
        },
        "outcome_distribution": {
            "cognitive_risk_label_0": int((df["cognitive_risk_label"] == 0).sum()),
            "cognitive_risk_label_1": int((df["cognitive_risk_label"] == 1).sum()),
        },
        "upf_tertile_counts": df["upf_tertile"].value_counts().to_dict()
        if "upf_tertile" in df.columns else {},
        "eeg_archetype_counts": df["eeg_archetype"].value_counts().to_dict()
        if "eeg_archetype" in df.columns else {},
    }
    return report


# ---------------------------------------------------------------------------
# 6.  Main orchestration
# ---------------------------------------------------------------------------

def run(nhanes_dir: str = "data/nhanes/",
        brfss_file: str = "data/brfss/LLCP2023.XPT",
        sdoh_dir: str  = "data/sdoh/",
        deap_dir: str  = "data/deap/",
        output_dir: str = "data/") -> pd.DataFrame:

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    source_counts: dict = {}

    # -- NHANES --
    logger.info("Loading NHANES 2017-2020 …")
    try:
        nhanes_df = run_nhanes(data_dir=nhanes_dir)
        source_counts["nhanes"] = len(nhanes_df)
        logger.info(f"  NHANES rows: {len(nhanes_df):,}")
    except Exception as e:
        logger.warning(f"NHANES pipeline failed: {e}. Using empty stub.")
        nhanes_df = pd.DataFrame()
        source_counts["nhanes"] = 0

    # -- BRFSS --
    logger.info("Loading BRFSS 2023 …")
    try:
        brfss_df = run_brfss(xpt_path=brfss_file)
        source_counts["brfss"] = len(brfss_df)
        logger.info(f"  BRFSS rows: {len(brfss_df):,}")
    except Exception as e:
        logger.warning(f"BRFSS pipeline failed: {e}. Using empty stub.")
        brfss_df = pd.DataFrame()
        source_counts["brfss"] = 0

    # -- Merge NHANES + BRFSS (vertical stack where schemas overlap) --
    logger.info("Merging NHANES + BRFSS …")
    shared_cols = list(set(nhanes_df.columns) & set(brfss_df.columns)) \
        if not nhanes_df.empty and not brfss_df.empty else []

    if shared_cols:
        combined_df = pd.concat(
            [nhanes_df[shared_cols], brfss_df[shared_cols]],
            ignore_index=True, sort=False
        )
    elif not nhanes_df.empty:
        combined_df = nhanes_df.copy()
    elif not brfss_df.empty:
        combined_df = brfss_df.copy()
    else:
        logger.error("Both NHANES and BRFSS failed — cannot build integrated cohort.")
        return pd.DataFrame()

    logger.info(f"  Combined population rows: {len(combined_df):,}")

    # -- SDOH --
    logger.info("Loading SDOH / equity layers …")
    try:
        sdoh_df = run_sdoh(data_dir=sdoh_dir)
        source_counts["sdoh"] = len(sdoh_df) if sdoh_df is not None else 0
    except Exception as e:
        logger.warning(f"SDOH pipeline failed: {e}.")
        sdoh_df = None

    combined_df = attach_sdoh_flags(combined_df, sdoh_df)

    # -- DEAP EEG --
    logger.info("Loading DEAP EEG features …")
    try:
        deap_df = run_deap(data_dir=deap_dir)
        source_counts["deap_subjects"] = int(deap_df["subject_id"].nunique()) \
            if deap_df is not None and "subject_id" in deap_df.columns else 0
    except Exception as e:
        logger.warning(f"DEAP pipeline failed: {e}.")
        deap_df = None

    # -- Feature engineering --
    logger.info("Engineering composite features …")
    if "upf_energy_pct" in combined_df.columns:
        combined_df = compute_upf_tertile(combined_df)
    if "age_years" in combined_df.columns:
        combined_df = standardise_age(combined_df)

    combined_df = composite_cognitive_risk(combined_df)

    # Build DEAP summary for archetype mapping
    deap_summary = None
    if deap_df is not None and not deap_df.empty:
        required = {"pac_mi_theta_gamma", "frontal_alpha_asym", "arousal_valence_quad"}
        if required.issubset(deap_df.columns):
            deap_summary = deap_df[list(required)].copy()

    combined_df = attach_eeg_archetypes(combined_df, deap_summary)

    # -- Save outputs --
    parquet_path = os.path.join(output_dir, "integrated_cohort.parquet")
    combined_df.to_parquet(parquet_path, index=False)
    logger.info(f"Saved integrated cohort → {parquet_path}")

    # Save DEAP EEG summary separately
    if deap_df is not None and not deap_df.empty:
        eeg_out = os.path.join(output_dir, "eeg_feature_summary.csv")
        deap_df.to_csv(eeg_out, index=False)
        logger.info(f"Saved EEG feature summary → {eeg_out}")

    # QC report
    report = build_qc_report(combined_df, source_counts)
    report_path = os.path.join(output_dir, "integration_report.json")
    with open(report_path, "w") as fh:
        json.dump(report, fh, indent=2)
    logger.info(f"Saved QC report → {report_path}")

    logger.info("Integration pipeline complete.")
    logger.info(f"  Final cohort: {len(combined_df):,} rows × {len(combined_df.columns)} cols")
    return combined_df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UPF-NeuroCog v2.0 — Integration Pipeline")
    parser.add_argument("--nhanes_dir",  default="data/nhanes/",
                        help="Directory containing NHANES XPT files")
    parser.add_argument("--brfss_file",  default="data/brfss/LLCP2023.XPT",
                        help="Path to BRFSS 2023 XPT file")
    parser.add_argument("--sdoh_dir",    default="data/sdoh/",
                        help="Directory containing SDOH CSVs (SVI, PLACES, FoodAtlas)")
    parser.add_argument("--deap_dir",    default="data/deap/",
                        help="Directory containing DEAP .dat files")
    parser.add_argument("--output_dir",  default="data/",
                        help="Output directory for parquet and reports")
    args = parser.parse_args()

    run(
        nhanes_dir=args.nhanes_dir,
        brfss_file=args.brfss_file,
        sdoh_dir=args.sdoh_dir,
        deap_dir=args.deap_dir,
        output_dir=args.output_dir,
    )
