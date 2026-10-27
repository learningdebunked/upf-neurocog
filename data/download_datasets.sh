#!/usr/bin/env bash
# =============================================================================
# download_datasets.sh  —  UPF-NeuroCog v2.0
# =============================================================================
# Downloads all publicly available datasets required for the project.
#
# Datasets requiring manual registration are flagged with [MANUAL] and
# provide exact download URLs / instructions.
#
# Usage:
#   chmod +x data/download_datasets.sh
#   cd /path/to/upf-neurocog-v2
#   bash data/download_datasets.sh
# =============================================================================

set -euo pipefail

NHANES_DIR="data/nhanes"
BRFSS_DIR="data/brfss"
SDOH_DIR="data/sdoh"
DEAP_DIR="data/deap"

mkdir -p "$NHANES_DIR" "$BRFSS_DIR" "$SDOH_DIR" "$DEAP_DIR"

echo "================================================================"
echo "  UPF-NeuroCog v2.0 — Dataset Downloader"
echo "  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "================================================================"

# ---------------------------------------------------------------------------
# A.  NHANES 2017–2018  (cycle J)
# ---------------------------------------------------------------------------
echo ""
echo "[1/5] Downloading NHANES 2017-2018 …"

declare -A NHANES_J=(
  [DR1TOT_J.XPT]="https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/DR1TOT_J.XPT"
  [DR1IFF_J.XPT]="https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/DR1IFF_J.XPT"
  [DEMO_J.XPT]="https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/DEMO_J.XPT"
  [BMX_J.XPT]="https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/BMX_J.XPT"
  [PAQ_J.XPT]="https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/PAQ_J.XPT"
  [GHB_J.XPT]="https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/GHB_J.XPT"
  [BPX_J.XPT]="https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/BPX_J.XPT"
)

for fname in "${!NHANES_J[@]}"; do
  target="$NHANES_DIR/$fname"
  if [ -f "$target" ]; then
    echo "  [skip] $fname already present"
  else
    echo "  Downloading $fname …"
    curl -fL --retry 3 --retry-delay 5 \
      "${NHANES_J[$fname]}" -o "$target"
    echo "  OK: $fname ($(du -sh "$target" | cut -f1))"
  fi
done

# ---------------------------------------------------------------------------
# B.  NHANES 2019–2020  (cycle P — pre-pandemic)
# ---------------------------------------------------------------------------
echo ""
echo "[2/5] Downloading NHANES 2019-2020 …"

declare -A NHANES_P=(
  [DR1TOT_P.XPT]="https://wwwn.cdc.gov/Nchs/Nhanes/2019-2020/DR1TOT_P.XPT"
  [DR1IFF_P.XPT]="https://wwwn.cdc.gov/Nchs/Nhanes/2019-2020/DR1IFF_P.XPT"
  [DEMO_P.XPT]="https://wwwn.cdc.gov/Nchs/Nhanes/2019-2020/DEMO_P.XPT"
  [BMX_P.XPT]="https://wwwn.cdc.gov/Nchs/Nhanes/2019-2020/BMX_P.XPT"
  [PAQ_P.XPT]="https://wwwn.cdc.gov/Nchs/Nhanes/2019-2020/PAQ_P.XPT"
  [GHB_P.XPT]="https://wwwn.cdc.gov/Nchs/Nhanes/2019-2020/GHB_P.XPT"
  [BPXO_P.XPT]="https://wwwn.cdc.gov/Nchs/Nhanes/2019-2020/BPXO_P.XPT"
)

for fname in "${!NHANES_P[@]}"; do
  target="$NHANES_DIR/$fname"
  if [ -f "$target" ]; then
    echo "  [skip] $fname already present"
  else
    echo "  Downloading $fname …"
    curl -fL --retry 3 --retry-delay 5 \
      "${NHANES_P[$fname]}" -o "$target"
    echo "  OK: $fname ($(du -sh "$target" | cut -f1))"
  fi
done

# ---------------------------------------------------------------------------
# C.  BRFSS 2023
# ---------------------------------------------------------------------------
echo ""
echo "[3/5] Downloading BRFSS 2023 …"

BRFSS_URL="https://www.cdc.gov/brfss/annual_data/2023/files/LLCP2023XPT.zip"
BRFSS_ZIP="$BRFSS_DIR/LLCP2023XPT.zip"

if [ -f "$BRFSS_DIR/LLCP2023.XPT" ]; then
  echo "  [skip] LLCP2023.XPT already present"
else
  echo "  Downloading BRFSS 2023 ZIP (~400 MB) …"
  curl -fL --retry 3 --retry-delay 10 \
    "$BRFSS_URL" -o "$BRFSS_ZIP"
  echo "  Extracting …"
  unzip -o "$BRFSS_ZIP" -d "$BRFSS_DIR"
  rm -f "$BRFSS_ZIP"
  echo "  OK: BRFSS 2023 extracted to $BRFSS_DIR"
fi

# ---------------------------------------------------------------------------
# D.  SDOH — CDC PLACES 2023
# ---------------------------------------------------------------------------
echo ""
echo "[4/5] Downloading CDC PLACES 2023 …"

PLACES_CSV="$SDOH_DIR/PLACES_2023.csv"
if [ -f "$PLACES_CSV" ]; then
  echo "  [skip] PLACES_2023.csv already present"
else
  echo "  Downloading CDC PLACES 2023 (~250 MB) …"
  curl -fL --retry 3 --retry-delay 5 \
    "https://data.cdc.gov/api/views/cwsq-ngmh/rows.csv?accessType=DOWNLOAD" \
    -o "$PLACES_CSV"
  echo "  OK: PLACES_2023.csv ($(du -sh "$PLACES_CSV" | cut -f1))"
fi

# USDA Food Desert Atlas 2019
USDA_XLSX="$SDOH_DIR/FoodAccessResearchAtlasData2019.xlsx"
if [ -f "$USDA_XLSX" ]; then
  echo "  [skip] USDA Food Desert Atlas already present"
else
  echo "  Downloading USDA Food Desert Atlas …"
  curl -fL --retry 3 --retry-delay 5 \
    "https://www.ers.usda.gov/webdocs/DataFiles/80591/FoodAccessResearchAtlasData2019.xlsx" \
    -o "$USDA_XLSX"
  echo "  OK: FoodAccessResearchAtlasData2019.xlsx ($(du -sh "$USDA_XLSX" | cut -f1))"
fi

# ---------------------------------------------------------------------------
# E.  CDC SVI 2022  [MANUAL — requires browser download]
# ---------------------------------------------------------------------------
echo ""
echo "[5a/5] CDC Social Vulnerability Index 2022 — MANUAL DOWNLOAD"
echo "  URL: https://www.atsdr.cdc.gov/placeandhealth/svi/data_documentation_download.html"
echo "  Instructions:"
echo "    1. Navigate to the URL above."
echo "    2. Select Geography: US, Year: 2022, File Type: CSV."
echo "    3. Download 'SVI2022_US_county.csv' to $SDOH_DIR/"
echo "  (Automated download is unavailable due to form-based access.)"

# ---------------------------------------------------------------------------
# F.  DEAP EEG  [MANUAL — requires registration]
# ---------------------------------------------------------------------------
echo ""
echo "[5b/5] DEAP EEG Dataset — MANUAL DOWNLOAD (registration required)"
echo "  URL: https://www.eecs.qmul.ac.uk/mmv/datasets/deap/download.html"
echo "  Instructions:"
echo "    1. Register at the URL above (free academic access)."
echo "    2. Download 'data_preprocessed_python.zip' (~2.9 GB)."
echo "    3. Extract to $DEAP_DIR/"
echo "    Expected structure after extraction:"
echo "      $DEAP_DIR/s01.dat  $DEAP_DIR/s02.dat  …  $DEAP_DIR/s32.dat"
echo "  (Automated download is unavailable due to registration wall.)"

echo ""
echo "================================================================"
echo "  Download complete."
echo "  Automated datasets: NHANES 2017-2020, BRFSS 2023, CDC PLACES,"
echo "                      USDA Food Desert Atlas"
echo "  Manual datasets:    CDC SVI 2022, DEAP EEG"
echo "================================================================"
