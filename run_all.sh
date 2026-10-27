#!/usr/bin/env bash
# =============================================================================
# run_all.sh  —  UPF-NeuroCog v2.0
# Runs the complete pipeline end-to-end.
# Usage: bash run_all.sh
# =============================================================================

set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT/analysis"

echo "================================================================"
echo "  UPF-NeuroCog v2.0 — Full Pipeline"
echo "  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "================================================================"

echo ""
echo "[0/7] Downloading datasets …"
bash ../data/download_datasets.sh

echo ""
echo "[1/7] NHANES 2017-2020 pipeline …"
python nhanes_pipeline.py \
    --data_dir ../data/nhanes/ \
    --output   ../data/nhanes_processed.parquet

echo ""
echo "[2/7] BRFSS 2023 pipeline …"
python brfss_pipeline.py \
    --xpt_path ../data/brfss/LLCP2023.XPT \
    --output   ../data/brfss_processed.parquet

echo ""
echo "[3/7] DEAP EEG pipeline (skips gracefully if data absent) …"
python deap_eeg_pipeline.py \
    --data_dir ../data/deap/ \
    --output   ../data/deap_features.csv || \
    echo "  [warn] DEAP pipeline skipped — manual download required (see DATA_AVAILABILITY.md)"

echo ""
echo "[4/7] SDOH equity layer …"
python sdoh_equity.py \
    --data_dir ../data/sdoh/ \
    --output   ../data/sdoh_processed.csv

echo ""
echo "[5/7] NOVA LLM classification (rule-based backend) …"
python nova_llm_classifier.py \
    --input    ../data/nhanes/DR1IFF_J.XPT \
    --output   ../data/nova_classified.csv \
    --backend  rule

echo ""
echo "[6/7] Integration pipeline …"
python integration_pipeline.py \
    --nhanes_dir  ../data/nhanes/ \
    --brfss_file  ../data/brfss/LLCP2023.XPT \
    --sdoh_dir    ../data/sdoh/ \
    --deap_dir    ../data/deap/ \
    --output_dir  ../data/

echo ""
echo "[7/7] Model training …"
python model_training.py \
    --cohort  ../data/integrated_cohort.parquet \
    --output  ../results/ \
    --models  ../models/ \
    --figures ../figures/

echo ""
echo "================================================================"
echo "  Pipeline complete!"
echo "  Results:  $ROOT/results/metrics.json"
echo "  Figures:  $ROOT/figures/"
echo "  Model:    $ROOT/models/xgb_model.pkl"
echo "================================================================"
