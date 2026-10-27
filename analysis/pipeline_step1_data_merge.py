"""
Step 1: Load, merge, and classify NHANES 2011-2014 data.
Outputs merged DataFrame saved to /home/sandbox/nhanes_analysis/nhanes_merged.csv
"""

import sys
sys.path.insert(0, '/home/sandbox/nhanes_analysis')

import pandas as pd
import numpy as np
from nova_classification import classify_nova
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = '/home/sandbox/nhanes_data/'
OUT_DIR  = '/home/sandbox/nhanes_analysis/'

print("=" * 60)
print("NHANES 2011-2014 Data Pipeline — Step 1: Merge & Classify")
print("=" * 60)

# ── 1. Load demographics ───────────────────────────────────────
print("\n[1] Loading demographics...")
demo_g = pd.read_sas(DATA_DIR + 'DEMO_G.xpt', format='xport', encoding='latin-1')
demo_h = pd.read_sas(DATA_DIR + 'DEMO_H.xpt', format='xport', encoding='latin-1')
demo = pd.concat([demo_g, demo_h], ignore_index=True)

# Keep relevant columns
demo_cols = ['SEQN', 'RIDAGEYR', 'RIAGENDR', 'RIDRETH3', 'DMDEDUC2',
             'INDFMPIR', 'SDMVPSU', 'SDMVSTRA', 'WTMEC2YR']
demo = demo[demo_cols].copy()
demo['SEQN'] = demo['SEQN'].astype(int)
print(f"   Demographics: {demo.shape[0]:,} participants")

# ── 2. Filter to adults ≥60 ────────────────────────────────────
demo_60 = demo[demo['RIDAGEYR'] >= 60].copy()
print(f"   Adults ≥60:   {demo_60.shape[0]:,} participants")

# ── 3. Load cognitive function (CFQ) ──────────────────────────
print("\n[2] Loading cognitive function data...")
cfq_g = pd.read_sas(DATA_DIR + 'CFQ_G.xpt', format='xport', encoding='latin-1')
cfq_h = pd.read_sas(DATA_DIR + 'CFQ_H.xpt', format='xport', encoding='latin-1')
cfq = pd.concat([cfq_g, cfq_h], ignore_index=True)

# DSST = Digit Symbol Substitution Test score
cfq_cols = ['SEQN', 'CFDDS', 'CFANIM', 'CFDCST2']
available_cfq = [c for c in cfq_cols if c in cfq.columns]
cfq = cfq[available_cfq].copy()
cfq['SEQN'] = cfq['SEQN'].astype(int)
print(f"   CFQ records:  {cfq.shape[0]:,}")
print(f"   Available cognitive cols: {available_cfq}")

# ── 4. Load BMI ────────────────────────────────────────────────
print("\n[3] Loading body measures (BMI)...")
bmx_g = pd.read_sas(DATA_DIR + 'BMX_G.xpt', format='xport', encoding='latin-1')
bmx_h = pd.read_sas(DATA_DIR + 'BMX_H.xpt', format='xport', encoding='latin-1')
bmx = pd.concat([bmx_g, bmx_h], ignore_index=True)
bmx = bmx[['SEQN', 'BMXBMI']].copy()
bmx['SEQN'] = bmx['SEQN'].astype(int)
print(f"   BMX records:  {bmx.shape[0]:,}")

# ── 5. Load physical activity ──────────────────────────────────
print("\n[4] Loading physical activity...")
paq_g = pd.read_sas(DATA_DIR + 'PAQ_G.xpt', format='xport', encoding='latin-1')
paq_h = pd.read_sas(DATA_DIR + 'PAQ_H.xpt', format='xport', encoding='latin-1')
paq = pd.concat([paq_g, paq_h], ignore_index=True)

# PAD680 = sedentary minutes/day; PAQ650 = vigorous activity
paq_cols_want = ['SEQN', 'PAD680', 'PAQ650', 'PAQ665']
paq_cols_avail = [c for c in paq_cols_want if c in paq.columns]
paq = paq[paq_cols_avail].copy()
paq['SEQN'] = paq['SEQN'].astype(int)
print(f"   PAQ records:  {paq.shape[0]:,}, cols: {paq_cols_avail}")

# ── 6. Load dietary totals (for total energy) ─────────────────
print("\n[5] Loading dietary totals...")
drtot_g = pd.read_sas(DATA_DIR + 'DR1TOT_G.xpt', format='xport', encoding='latin-1')
drtot_h = pd.read_sas(DATA_DIR + 'DR1TOT_H.xpt', format='xport', encoding='latin-1')
drtot = pd.concat([drtot_g, drtot_h], ignore_index=True)
drtot = drtot[['SEQN', 'DR1TKCAL']].copy()
drtot['SEQN'] = drtot['SEQN'].astype(int)
print(f"   Dietary total records: {drtot.shape[0]:,}")

# ── 7. Load individual food items & apply NOVA ────────────────
print("\n[6] Loading individual food items and applying NOVA classification...")
print("    (This may take ~1-2 min for 83MB + 85MB files...)")

chunks_g = []
for chunk in pd.read_sas(DATA_DIR + 'DR1IFF_G.xpt', format='xport',
                          encoding='latin-1', chunksize=50000):
    chunks_g.append(chunk[['SEQN', 'DR1IFDCD', 'DR1IKCAL', 'DR1IGRMS',
                             'DRDDESC'] if 'DRDDESC' in chunk.columns
                            else ['SEQN', 'DR1IFDCD', 'DR1IKCAL', 'DR1IGRMS']])
dr1iff_g = pd.concat(chunks_g, ignore_index=True)

chunks_h = []
for chunk in pd.read_sas(DATA_DIR + 'DR1IFF_H.xpt', format='xport',
                          encoding='latin-1', chunksize=50000):
    chunks_h.append(chunk[['SEQN', 'DR1IFDCD', 'DR1IKCAL', 'DR1IGRMS',
                             'DRDDESC'] if 'DRDDESC' in chunk.columns
                            else ['SEQN', 'DR1IFDCD', 'DR1IKCAL', 'DR1IGRMS']])
dr1iff_h = pd.concat(chunks_h, ignore_index=True)

dr1iff = pd.concat([dr1iff_g, dr1iff_h], ignore_index=True)
dr1iff['SEQN'] = dr1iff['SEQN'].astype(int)
print(f"   Total food item records: {dr1iff.shape[0]:,}")
print(f"   Columns: {list(dr1iff.columns)}")

# Apply NOVA classification to each food item
print("\n[7] Applying NOVA classification to each food item...")
desc_col = 'DRDDESC' if 'DRDDESC' in dr1iff.columns else None

if desc_col:
    dr1iff['NOVA'] = [
        classify_nova(int(fc) if not pd.isna(fc) else 0,
                      str(desc) if not pd.isna(desc) else '')
        for fc, desc in zip(dr1iff['DR1IFDCD'], dr1iff[desc_col])
    ]
else:
    dr1iff['NOVA'] = [
        classify_nova(int(fc) if not pd.isna(fc) else 0, '')
        for fc in dr1iff['DR1IFDCD']
    ]

nova_dist = dr1iff['NOVA'].value_counts().sort_index()
print(f"   NOVA distribution (food items):")
for g, n in nova_dist.items():
    print(f"     NOVA {g}: {n:,} items ({n/len(dr1iff)*100:.1f}%)")

# ── 8. Compute per-person UPF energy % ────────────────────────
print("\n[8] Computing per-person UPF energy %...")

# Total kcal per person from ALL items
total_kcal = dr1iff.groupby('SEQN')['DR1IKCAL'].sum().reset_index()
total_kcal.columns = ['SEQN', 'item_total_kcal']

# NOVA 4 kcal per person
nova4_kcal = (dr1iff[dr1iff['NOVA'] == 4]
              .groupby('SEQN')['DR1IKCAL'].sum().reset_index())
nova4_kcal.columns = ['SEQN', 'upf_kcal']

upf_df = total_kcal.merge(nova4_kcal, on='SEQN', how='left')
upf_df['upf_kcal'] = upf_df['upf_kcal'].fillna(0)
upf_df['upf_pct'] = np.where(
    upf_df['item_total_kcal'] > 0,
    upf_df['upf_kcal'] / upf_df['item_total_kcal'] * 100,
    np.nan
)

print(f"   Participants with dietary data: {upf_df.shape[0]:,}")
print(f"   UPF % — mean: {upf_df['upf_pct'].mean():.1f}%, "
      f"median: {upf_df['upf_pct'].median():.1f}%, "
      f"std: {upf_df['upf_pct'].std():.1f}%")

# ── 9. Merge all datasets ──────────────────────────────────────
print("\n[9] Merging all datasets for adults ≥60...")

df = (demo_60
      .merge(cfq,    on='SEQN', how='inner')
      .merge(upf_df[['SEQN', 'upf_kcal', 'upf_pct']], on='SEQN', how='left')
      .merge(drtot,  on='SEQN', how='left')
      .merge(bmx,    on='SEQN', how='left')
      .merge(paq,    on='SEQN', how='left'))

print(f"   After inner merge (demo ≥60 + CFQ): {df.shape[0]:,} participants")

# ── 10. Filter complete DSST cases ────────────────────────────
print("\n[10] Filtering complete DSST cases (CFDDS not missing)...")
df = df.dropna(subset=['CFDDS'])
print(f"    With valid DSST score: {df.shape[0]:,} participants")

# ── 11. Handle NHANES missing value codes ─────────────────────
print("\n[11] Recoding NHANES refusal/unknown codes to NaN...")

# NHANES uses 7=Refused, 9=Don't Know, 77/99/777/999 etc.
def recode_nhanes_missing(series):
    """Replace NHANES refused/unknown codes with NaN."""
    s = series.copy()
    # Large sentinel values indicating missing
    if s.dtype in [float, np.float64]:
        s = s.replace([7.0, 9.0, 77.0, 99.0, 777.0, 999.0,
                        7777.0, 9999.0], np.nan)
    return s

for col in ['DMDEDUC2', 'INDFMPIR', 'BMXBMI']:
    if col in df.columns:
        df[col] = recode_nhanes_missing(df[col])

# Physical activity: 9999 = missing
if 'PAD680' in df.columns:
    df['PAD680'] = df['PAD680'].replace([9999.0, 7777.0], np.nan)
if 'PAQ650' in df.columns:
    df['PAQ650'] = df['PAQ650'].replace([9.0, 7.0], np.nan)

# ── 12. Create binary cognitive impairment outcome ────────────
print("\n[12] Creating cognitive impairment binary outcome...")
# Low DSST defined as <1 SD below age-sex adjusted mean
# Age-stratified: 60-69 vs 70+
df['age_group'] = np.where(df['RIDAGEYR'] < 70, '60-69', '70+')

def compute_impairment(group):
    mean = group['CFDDS'].mean()
    sd   = group['CFDDS'].std()
    # Cognitive impairment = score < (mean - 1 SD)
    group = group.copy()
    group['cog_impaired'] = (group['CFDDS'] < (mean - sd)).astype(int)
    return group

df = df.groupby('age_group', group_keys=False).apply(compute_impairment)

imp_pct = df['cog_impaired'].mean() * 100
print(f"   Cognitive impairment prevalence: {imp_pct:.1f}% "
      f"(n={df['cog_impaired'].sum():,} of {len(df):,})")

# ── 13. UPF quartiles ─────────────────────────────────────────
print("\n[13] Creating UPF energy% quartiles...")
df = df.dropna(subset=['upf_pct'])
df['upf_quartile'] = pd.qcut(df['upf_pct'], q=4,
                               labels=['Q1', 'Q2', 'Q3', 'Q4'])
q_summary = df.groupby('upf_quartile', observed=True)['upf_pct'].agg(['min','max','mean'])
print(q_summary)

# ── 14. Save merged dataset ───────────────────────────────────
out_path = OUT_DIR + 'nhanes_merged.csv'
df.to_csv(out_path, index=False)
print(f"\n[SAVED] {out_path}")
print(f"   Final analytic sample: {df.shape[0]:,} participants, "
      f"{df.shape[1]} variables")

# Summary of missing values
print("\n[15] Missing value summary for key variables:")
key_vars = ['CFDDS', 'upf_pct', 'RIDAGEYR', 'RIAGENDR', 'RIDRETH3',
            'DMDEDUC2', 'INDFMPIR', 'BMXBMI', 'DR1TKCAL']
for v in key_vars:
    if v in df.columns:
        n_miss = df[v].isna().sum()
        pct_miss = n_miss / len(df) * 100
        print(f"   {v:12s}: {n_miss:4d} missing ({pct_miss:.1f}%)")

print("\n✓ Step 1 complete.")
