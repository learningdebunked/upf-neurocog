"""
deap_eeg_pipeline.py  —  UPF-NeuroCog v2.0
============================================
Loads the DEAP dataset, extracts per-trial EEG features
(band power, cross-frequency coupling, coherence),
maps arousal/valence to cognitive fatigue states,
and exports a participant-level feature matrix for
diet→brain-state mapping.

DEAP Dataset:
  Access    : https://www.eecs.qmul.ac.uk/mmv/datasets/deap/download.html
              (register at the URL above; download is free for research)
  File      : data_preprocessed_python.zip
  Reference : Koelstra S. et al., IEEE Trans Affect Comput 3(1):18–31, 2012
              DOI: 10.1109/T-AFFC.2011.15

After downloading, place the extracted folder at:
  data/deap/data_preprocessed_python/

Contents per participant file (s01.dat … s32.dat):
  data   : (40 trials × 40 channels × 8064 timepoints) float32
             Channels 0-31: EEG (10-20 system, 128 Hz)
             Channels 32-39: peripheral (EMG, GSR, Resp, Temp, etc.)
  labels : (40 trials × 4 labels)
             Col 0: Valence (1-9), Col 1: Arousal (1-9)
             Col 2: Dominance (1-9), Col 3: Liking (1-9)

Usage
-----
  python deap_eeg_pipeline.py --deap_dir data/deap/data_preprocessed_python \
                               --outdir   data/deap_features
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal as sp_signal
from scipy.stats import entropy

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────
SRATE   = 128           # Hz (DEAP preprocessed)
N_SUBS  = 32
N_TRIALS= 40
N_EEG   = 32            # EEG channels (0-31)

# Frequency bands (Hz)
BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta":  (13, 30),
    "gamma": (30, 45),
}

# Frontal electrode indices (Fp1, Fp2, Fz, F3, F4)
FRONTAL_CH = [0, 1, 2, 3, 4]

# ─────────────────────────────────────────────────────────────
# Band power
# ─────────────────────────────────────────────────────────────

def bandpower(eeg: np.ndarray, fs: int, fmin: float, fmax: float) -> float:
    """Welch PSD band power for a single channel-trial epoch."""
    freqs, psd = sp_signal.welch(eeg, fs=fs, nperseg=fs * 2)
    idx = (freqs >= fmin) & (freqs <= fmax)
    return float(np.trapz(psd[idx], freqs[idx]))


def extract_band_powers(epoch: np.ndarray) -> dict:
    """
    Extract band powers for all EEG channels.
    epoch shape: (n_channels, n_times)
    Returns dict keyed by '{channel}_{band}'.
    """
    feats = {}
    for ch in range(N_EEG):
        for band, (lo, hi) in BANDS.items():
            feats[f"ch{ch:02d}_{band}"] = bandpower(epoch[ch], SRATE, lo, hi)
    return feats


# ─────────────────────────────────────────────────────────────
# Cross-Frequency Coupling  (Phase-Amplitude Coupling)
# ─────────────────────────────────────────────────────────────

def phase_amplitude_coupling(x: np.ndarray, fs: int,
                              phase_band: tuple, amp_band: tuple,
                              n_bins: int = 18) -> float:
    """
    Modulation Index (MI) for Phase-Amplitude Coupling.
    Tort et al. (2010), J Neurophysiol 104(2):1195-1210.
    DOI: 10.1152/jn.00106.2010

    Parameters
    ----------
    x          : 1-D EEG signal (samples)
    fs         : sampling rate (Hz)
    phase_band : (flo, fhi) for phase signal (e.g. theta: 4-8 Hz)
    amp_band   : (flo, fhi) for amplitude signal (e.g. gamma: 30-45 Hz)
    n_bins     : number of phase bins for MI computation

    Returns
    -------
    Modulation Index (float, higher = stronger coupling)
    """
    # Band-pass filter for phase
    b_phase, a_phase = sp_signal.butter(
        4, [phase_band[0] / (fs / 2), phase_band[1] / (fs / 2)], btype="band"
    )
    phi = np.angle(sp_signal.hilbert(sp_signal.filtfilt(b_phase, a_phase, x)))

    # Band-pass filter for amplitude
    b_amp, a_amp = sp_signal.butter(
        4, [amp_band[0] / (fs / 2), amp_band[1] / (fs / 2)], btype="band"
    )
    amp = np.abs(sp_signal.hilbert(sp_signal.filtfilt(b_amp, a_amp, x)))

    # Phase bins
    bins     = np.linspace(-np.pi, np.pi, n_bins + 1)
    mean_amp = np.array([
        amp[(phi >= bins[i]) & (phi < bins[i + 1])].mean()
        if np.any((phi >= bins[i]) & (phi < bins[i + 1])) else 0.0
        for i in range(n_bins)
    ])

    # Normalise → probability distribution
    if mean_amp.sum() == 0:
        return 0.0
    p = mean_amp / mean_amp.sum()

    # MI = KL-divergence from uniform
    mi = (np.log(n_bins) - entropy(p + 1e-12)) / np.log(n_bins)
    return float(mi)


def extract_cfc_features(epoch: np.ndarray) -> dict:
    """
    Compute theta-gamma and alpha-gamma PAC for frontal channels.
    These CFC pairs are the primary mechanistic bridge between
    dietary stress and measurable cognitive fatigue in this framework.

    Returns dict with MI values per channel pair.
    """
    feats = {}
    for ch in FRONTAL_CH:
        sig = epoch[ch]
        feats[f"ch{ch:02d}_theta_gamma_MI"] = phase_amplitude_coupling(
            sig, SRATE, BANDS["theta"], BANDS["gamma"]
        )
        feats[f"ch{ch:02d}_alpha_gamma_MI"] = phase_amplitude_coupling(
            sig, SRATE, BANDS["alpha"], BANDS["gamma"]
        )
        feats[f"ch{ch:02d}_delta_theta_MI"] = phase_amplitude_coupling(
            sig, SRATE, BANDS["delta"], BANDS["theta"]
        )
    return feats


# ─────────────────────────────────────────────────────────────
# Asymmetry (frontal alpha asymmetry — motivation / withdrawal)
# ─────────────────────────────────────────────────────────────

def frontal_alpha_asymmetry(epoch: np.ndarray) -> float:
    """
    FAA = ln(F4 alpha) - ln(F3 alpha)
    Positive → approach motivation; Negative → withdrawal / stress.
    F3=ch3, F4=ch4 in DEAP 10-20 layout.
    """
    f3_alpha = bandpower(epoch[3], SRATE, *BANDS["alpha"])
    f4_alpha = bandpower(epoch[4], SRATE, *BANDS["alpha"])
    if f3_alpha <= 0 or f4_alpha <= 0:
        return float("nan")
    return float(np.log(f4_alpha) - np.log(f3_alpha))


# ─────────────────────────────────────────────────────────────
# Arousal / valence label encoding
# ─────────────────────────────────────────────────────────────

def encode_av_labels(labels_row: np.ndarray) -> dict:
    """
    Encode arousal/valence into four quadrants (Russell's circumplex):
      HV-HA : happy / excited   (valence≥5, arousal≥5)
      HV-LA : content / relaxed (valence≥5, arousal<5)
      LV-HA : stressed / angry  (valence<5, arousal≥5)  ← cognitive fatigue marker
      LV-LA : sad / depressed   (valence<5, arousal<5)
    """
    valence, arousal = float(labels_row[0]), float(labels_row[1])
    quadrant = (
        "HV-HA" if valence >= 5 and arousal >= 5 else
        "HV-LA" if valence >= 5 and arousal < 5  else
        "LV-HA" if valence < 5  and arousal >= 5 else
        "LV-LA"
    )
    return {
        "valence": valence, "arousal": arousal,
        "dominance": float(labels_row[2]), "liking": float(labels_row[3]),
        "av_quadrant": quadrant,
        "cognitive_fatigue_proxy": int(valence < 5 and arousal >= 5),  # LV-HA
        "high_arousal": int(arousal >= 5),
        "negative_valence": int(valence < 5),
    }


# ─────────────────────────────────────────────────────────────
# Per-participant processing
# ─────────────────────────────────────────────────────────────

def process_participant(dat_file: Path) -> pd.DataFrame:
    """
    Process one DEAP .dat file → DataFrame with one row per trial.
    """
    with open(dat_file, "rb") as f:
        subject = pickle.load(f, encoding="latin1")

    data   = subject["data"]    # (40, 40, 8064)
    labels = subject["labels"]  # (40, 4)

    participant_id = int(dat_file.stem[1:])  # s01 → 1
    rows = []

    for trial in range(N_TRIALS):
        epoch = data[trial][:N_EEG, :]   # EEG only (32 ch)
        row   = {"participant": participant_id, "trial": trial + 1}

        # Band powers (frontal only for speed; extend to all for full analysis)
        for ch in FRONTAL_CH:
            for band, (lo, hi) in BANDS.items():
                row[f"ch{ch:02d}_{band}"] = bandpower(epoch[ch], SRATE, lo, hi)

        # CFC features
        row.update(extract_cfc_features(epoch))

        # Frontal alpha asymmetry
        row["FAA"] = frontal_alpha_asymmetry(epoch)

        # Arousal/valence labels
        row.update(encode_av_labels(labels[trial]))

        rows.append(row)

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="DEAP EEG Pipeline — UPF-NeuroCog v2.0")
    parser.add_argument("--deap_dir", required=True,
                        help="Path to data_preprocessed_python/ folder from DEAP")
    parser.add_argument("--outdir",   default="data/deap_features")
    parser.add_argument("--n_subs",   type=int, default=32)
    args = parser.parse_args()

    deap_dir = Path(args.deap_dir)
    out_dir  = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_frames = []
    for sub_id in range(1, args.n_subs + 1):
        dat_file = deap_dir / f"s{sub_id:02d}.dat"
        if not dat_file.exists():
            print(f"  [skip] {dat_file.name} not found")
            continue
        print(f"  [process] {dat_file.name}")
        df_sub = process_participant(dat_file)
        all_frames.append(df_sub)

    if not all_frames:
        print("No DEAP files processed. Download from https://www.eecs.qmul.ac.uk/mmv/datasets/deap/download.html")
        return

    full_df = pd.concat(all_frames, ignore_index=True)

    out_path = out_dir / "deap_eeg_features.csv"
    full_df.to_csv(out_path, index=False)
    print(f"\n[deap_pipeline] Saved → {out_path}")
    print(f"  Trials processed : {len(full_df):,}  ({full_df['participant'].nunique()} participants)")
    print(f"  Features         : {full_df.shape[1]}")
    print(f"\nArousal/valence quadrant distribution:")
    print(full_df["av_quadrant"].value_counts())
    print(f"\nMean theta-gamma MI (frontal ch0):")
    print(full_df["ch00_theta_gamma_MI"].describe().round(4))


if __name__ == "__main__":
    main()
