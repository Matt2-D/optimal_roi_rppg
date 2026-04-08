#creates a 29th synthetic ROI from the 6 best predetermined ROIs,
#  weighted by their individual autocorrelation SQI mean (AC_SQI)
#  (falls back to spectral SNR weighting if no HR CSV is available)
#creates a 30th synthetic ROI from the 6 best predetermined ROIs, unweighted
#
#For each distance CSV (e.g. data/custom/rgb/1.csv), this script:
  #1. Extracts the 6 target ROI time-series
  #2. Loads per-ROI AC_SQI mean from the HR CSV (or computes SNR if HR CSV absent)
  #3. Normalizes AC_SQI weights (values clamped to 0 if non-finite)
  #4. Produces a weighted-average R, G, B per frame
  #5. Overwrites the RGB CSV to include the two new ROIs

# Author: Matthew Dowell
# Date 3/10/2026

import os
import shutil
import numpy as np
import pandas as pd
from scipy.signal import welch


# Configuration

TARGET_ROIS = [
    "glabella",
    "lower medial forehead",
    "right malar",
    "left malar",
    "left lower lateral forehead",
    "right lower lateral forehead",
]

ROI29_NAME = "simple_composite"
ROI30_NAME = "sqi_weighted_composite"
ROI31_NAME = "snr_weighted_composite"
ROI32_NAME = "ac_weighted_composite"
ROI33_NAME = "fc_weighted_composite"
ROI34_NAME = "cra_weighted_composite"
ROI35_NAME = "full_frame_average"

MIN_HZ = 0.7
MAX_HZ = 2.5
NFFT   = 4096


# SNR fallback (used only when no HR CSV is available)

def compute_snr2(signal: np.ndarray, fps: float) -> float:
    signal = signal - np.mean(signal)
    if np.all(signal == 0) or np.all(~np.isfinite(signal)):
        return -np.inf

    signal = np.nan_to_num(signal, nan=0.0)
    freqs, power = welch(
        signal, fs=fps,
        nperseg=min(NFFT, len(signal)),
        nfft=NFFT, window="hann", scaling="density",
    )
    band  = (freqs >= MIN_HZ) & (freqs <= MAX_HZ)
    freqs = freqs[band]
    power = power[band]

    if len(power) == 0:
        return -np.inf

    max_index        = np.argmax(power)
    peak_bins        = np.arange(max(0, max_index - 2), min(len(power), max_index + 3))
    fundamental_freq = freqs[max_index]
    mask             = np.ones(len(power), dtype=bool)
    mask[peak_bins]  = False
    total_signal     = np.sum(power[peak_bins])

    harmonic_freq = fundamental_freq * 2
    if harmonic_freq <= MAX_HZ:
        harm_idx   = np.argmin(np.abs(freqs - harmonic_freq))
        harm_bins  = np.arange(max(0, harm_idx - 2), min(len(power), harm_idx + 3))
        harm_power = np.sum(power[harm_bins])
        if harm_power > 0.1 * total_signal:
            total_signal   += harm_power
            mask[harm_bins] = False

    total_noise = np.sum(power[mask])
    if total_noise <= 0:
        return -np.inf
    return 10 * np.log10(total_signal / total_noise)


def channel_snr(roi_df: pd.DataFrame, fps: float) -> float:
    snrs = []
    for ch in ["R", "G", "B"]:
        s = compute_snr2(roi_df[ch].values.astype(np.float64), fps)
        if np.isfinite(s):
            snrs.append(s)
    return float(np.mean(snrs)) if snrs else -np.inf

"""
                df_hr.loc[mask, 'SNR'] = sig_snr[:, i_roi]
                df_hr.loc[mask, 'FC_SQI'] = sig_fc[:, i_roi]
                df_hr.loc[mask, 'AC_SQI'] = sig_ac[:, i_roi]
                df_hr.loc[mask, 'CRA_SQI'] = sig_cra[:, i_roi]
                df_hr.loc[mask, 'SQI'] = sig_sqi[:, i_roi]
"""

def load_sqi_column(hr_csv_path: str, roi_names: list, col: str) -> np.ndarray:
    """Generic loader — reads mean of any SQI column per ROI."""
    df_hr = pd.read_csv(hr_csv_path, index_col=False)
    df_hr = df_hr.loc[:, ~df_hr.columns.str.startswith('Unnamed')]

    if col not in df_hr.columns:
        raise KeyError(f"'{col}' column not found in HR CSV.")

    means = np.full(len(roi_names), np.nan)
    for i, roi in enumerate(roi_names):
        roi_rows = df_hr.loc[df_hr['ROI'] == roi, col]
        if len(roi_rows) > 0:
            means[i] = float(np.nanmean(
                pd.to_numeric(roi_rows, errors='coerce').values
            ))
    return means


def normalise_weights(values: np.ndarray) -> np.ndarray:
    """Clamp non-finite to 0, normalise to sum=1."""
    w = np.where(np.isfinite(values), values, 0.0)
    w = np.clip(w, 0, None)
    s = w.sum()
    if s == 0:
        raise RuntimeError("All weights are zero after clamping.")
    return w / s


def snr_weights(roi_data: dict, fps: float) -> np.ndarray:
    """Compute SNR-based weights from raw RGB (fallback)."""
    snr_vals = np.array([channel_snr(roi_data[r], fps) for r in TARGET_ROIS])
    finite   = np.isfinite(snr_vals)
    if not np.any(finite):
        raise RuntimeError("All SNR values are non-finite.")
    linear = np.where(finite, 10 ** (snr_vals / 10), 0.0)
    return normalise_weights(linear)

# AC_SQI loader

def load_ac_sqi_weights(hr_csv_path: str, roi_names: list) -> np.ndarray:
    """
    Load mean autocorrelation SQI (AC_SQI) per ROI from the HR CSV.

    Returns
    -------
    np.ndarray of shape [len(roi_names)] — mean AC_SQI per ROI,
    NaN for any ROI not found in the CSV.
    """
    df_hr = pd.read_csv(hr_csv_path, index_col=False)
    df_hr = df_hr.loc[:, ~df_hr.columns.str.startswith('Unnamed')]

    if 'AC_SQI' not in df_hr.columns:
        raise KeyError(
            "'AC_SQI' column not found in HR CSV. "
            "Ensure main_rgb2hr.py has been run with SQI computation enabled."
        )

    ac_means = np.full(len(roi_names), np.nan)
    for i, roi in enumerate(roi_names):
        roi_rows = df_hr.loc[df_hr['ROI'] == roi, 'AC_SQI']
        if len(roi_rows) > 0:
            ac_means[i] = float(np.nanmean(
                pd.to_numeric(roi_rows, errors='coerce').values
            ))
    return ac_means

# SNR loader

def load_snr_weights(hr_csv_path: str, roi_names: list) -> np.ndarray:
    df_hr = pd.read_csv(hr_csv_path, index_col=False)
    df_hr = df_hr.loc[:, ~df_hr.columns.str.startswith('Unnamed')]

    if 'SNR' not in df_hr.columns:
        raise KeyError(
            "'SNR' column not found in HR CSV. "
            "Ensure main_rgb2hr.py has been run with SQI computation enabled."
        )

    snr_means = np.full(len(roi_names), np.nan)
    for i, roi in enumerate(roi_names):
        roi_rows = df_hr.loc[df_hr['ROI'] == roi, 'SNR']
        if len(roi_rows) > 0:
            snr_means[i] = float(np.nanmean(
                pd.to_numeric(roi_rows, errors='coerce').values
            ))
    return snr_means

# FC_SQI loader

def load_fc_sqi_weights(hr_csv_path: str, roi_names: list) -> np.ndarray:
    df_hr = pd.read_csv(hr_csv_path, index_col=False)
    df_hr = df_hr.loc[:, ~df_hr.columns.str.startswith('Unnamed')]

    if 'FC_SQI' not in df_hr.columns:
        raise KeyError(
            "'FC_SQI' column not found in HR CSV. "
            "Ensure main_rgb2hr.py has been run with SQI computation enabled."
        )

    fc_means = np.full(len(roi_names), np.nan)
    for i, roi in enumerate(roi_names):
        roi_rows = df_hr.loc[df_hr['ROI'] == roi, 'FC_SQI']
        if len(roi_rows) > 0:
            fc_means[i] = float(np.nanmean(
                pd.to_numeric(roi_rows, errors='coerce').values
            ))
    return fc_means

# CRA_SQI loader

def load_cra_sqi_weights(hr_csv_path: str, roi_names: list) -> np.ndarray:
    df_hr = pd.read_csv(hr_csv_path, index_col=False)
    df_hr = df_hr.loc[:, ~df_hr.columns.str.startswith('Unnamed')]

    if 'CRA_SQI' not in df_hr.columns:
        raise KeyError(
            "'FC_SQI' column not found in HR CSV. "
            "Ensure main_rgb2hr.py has been run with SQI computation enabled."
        )

    cra_means = np.full(len(roi_names), np.nan)
    for i, roi in enumerate(roi_names):
        roi_rows = df_hr.loc[df_hr['ROI'] == roi, 'CRA_SQI']
        if len(roi_rows) > 0:
            cra_means[i] = float(np.nanmean(
                pd.to_numeric(roi_rows, errors='coerce').values
            ))
    return cra_means

# total SQI loader

def load_sqi_weights(hr_csv_path: str, roi_names: list) -> np.ndarray:
    df_hr = pd.read_csv(hr_csv_path, index_col=False)
    df_hr = df_hr.loc[:, ~df_hr.columns.str.startswith('Unnamed')]

    if 'SQI' not in df_hr.columns:
        raise KeyError(
            "'FC_SQI' column not found in HR CSV. "
            "Ensure main_rgb2hr.py has been run with SQI computation enabled."
        )

    sqi_means = np.full(len(roi_names), np.nan)
    for i, roi in enumerate(roi_names):
        roi_rows = df_hr.loc[df_hr['ROI'] == roi, 'SQI']
        if len(roi_rows) > 0:
            sqi_means[i] = float(np.nanmean(
                pd.to_numeric(roi_rows, errors='coerce').values
            ))
    return sqi_means

# ROI builders

def build_roi29(roi_data: dict, frames: list,
                    time_lookup: dict) -> pd.DataFrame:
    """Equal-weight (simple average) composite — ROI 29."""
    records = []
    for frame in frames:
        r_vals, g_vals, b_vals = [], [], []
        for roi in TARGET_ROIS:
            row = roi_data[roi][roi_data[roi]["frame"] == frame]
            if len(row) == 0:
                continue
            r_vals.append(float(row["R"].iloc[0]))
            g_vals.append(float(row["G"].iloc[0]))
            b_vals.append(float(row["B"].iloc[0]))
        records.append({
            "frame": frame,
            "time": time_lookup.get(frame, np.nan),
            "ROI": ROI29_NAME,
            "R": float(np.mean(r_vals)) if r_vals else np.nan,
            "G": float(np.mean(g_vals)) if g_vals else np.nan,
            "B": float(np.mean(b_vals)) if b_vals else np.nan,
        })
    return pd.DataFrame(records, columns=["frame", "time", "ROI", "R", "G", "B"])


def build_roi30(roi_data: dict, weights: np.ndarray, frames: list,
                 time_lookup: dict) -> pd.DataFrame:

    """SQI-weighted composite — ROI 29."""
    records = []
    for frame in frames:
        r, g, b = 0.0, 0.0, 0.0
        for roi, w in zip(TARGET_ROIS, weights):
            row = roi_data[roi][roi_data[roi]["frame"] == frame]
            if len(row) == 0:
                continue
            r += w * float(row["R"].iloc[0])
            g += w * float(row["G"].iloc[0])
            b += w * float(row["B"].iloc[0])
        records.append({
            "frame": frame,
            "time": time_lookup.get(frame, np.nan),
            "ROI": ROI30_NAME,
            "R": r, "G": g, "B": b,
        })
    return pd.DataFrame(records, columns=["frame", "time", "ROI", "R", "G", "B"])

def build_roi31(roi_data: dict, weights: np.ndarray, frames: list,
                 time_lookup: dict) -> pd.DataFrame:
    """SNR-weighted composite — ROI 31."""
    records = []
    for frame in frames:
        r, g, b = 0.0, 0.0, 0.0
        for roi, w in zip(TARGET_ROIS, weights):
            row = roi_data[roi][roi_data[roi]["frame"] == frame]
            if len(row) == 0:
                continue
            r += w * float(row["R"].iloc[0])
            g += w * float(row["G"].iloc[0])
            b += w * float(row["B"].iloc[0])
        records.append({
            "frame": frame,
            "time":  time_lookup.get(frame, np.nan),
            "ROI":   ROI31_NAME,
            "R": r, "G": g, "B": b,
        })
    return pd.DataFrame(records, columns=["frame", "time", "ROI", "R", "G", "B"])

def build_roi32(roi_data: dict, weights: np.ndarray, frames: list,
                 time_lookup: dict) -> pd.DataFrame:
    """ac-weighted composite — ROI 32."""
    records = []
    for frame in frames:
        r, g, b = 0.0, 0.0, 0.0
        for roi, w in zip(TARGET_ROIS, weights):
            row = roi_data[roi][roi_data[roi]["frame"] == frame]
            if len(row) == 0:
                continue
            r += w * float(row["R"].iloc[0])
            g += w * float(row["G"].iloc[0])
            b += w * float(row["B"].iloc[0])
        records.append({
            "frame": frame,
            "time":  time_lookup.get(frame, np.nan),
            "ROI":   ROI32_NAME,
            "R": r, "G": g, "B": b,
        })
    return pd.DataFrame(records, columns=["frame", "time", "ROI", "R", "G", "B"])

def build_roi33(roi_data: dict, weights: np.ndarray, frames: list,
                 time_lookup: dict) -> pd.DataFrame:
    """fc-weighted composite — ROI 31."""
    records = []
    for frame in frames:
        r, g, b = 0.0, 0.0, 0.0
        for roi, w in zip(TARGET_ROIS, weights):
            row = roi_data[roi][roi_data[roi]["frame"] == frame]
            if len(row) == 0:
                continue
            r += w * float(row["R"].iloc[0])
            g += w * float(row["G"].iloc[0])
            b += w * float(row["B"].iloc[0])
        records.append({
            "frame": frame,
            "time":  time_lookup.get(frame, np.nan),
            "ROI":   ROI33_NAME,
            "R": r, "G": g, "B": b,
        })
    return pd.DataFrame(records, columns=["frame", "time", "ROI", "R", "G", "B"])

def build_roi34(roi_data: dict, weights: np.ndarray, frames: list,
                 time_lookup: dict) -> pd.DataFrame:
    """cra-weighted composite — ROI 31."""
    records = []
    for frame in frames:
        r, g, b = 0.0, 0.0, 0.0
        for roi, w in zip(TARGET_ROIS, weights):
            row = roi_data[roi][roi_data[roi]["frame"] == frame]
            if len(row) == 0:
                continue
            r += w * float(row["R"].iloc[0])
            g += w * float(row["G"].iloc[0])
            b += w * float(row["B"].iloc[0])
        records.append({
            "frame": frame,
            "time":  time_lookup.get(frame, np.nan),
            "ROI":   ROI34_NAME,
            "R": r, "G": g, "B": b,
        })
    return pd.DataFrame(records, columns=["frame", "time", "ROI", "R", "G", "B"])

def build_roi35(df: pd.DataFrame, frames: list,
                 time_lookup: dict) -> pd.DataFrame:
    """
    ROI 35 — unweighted average R, G, B across ALL ROIs per frame.
    Represents the whole-frame mean signal.
    """
    records = []
    for frame in frames:
        frame_rows = df[df["frame"] == frame]
        records.append({
            "frame": frame,
            "time":  time_lookup.get(frame, np.nan),
            "ROI":   ROI35_NAME,
            "R":     float(frame_rows["R"].mean()),
            "G":     float(frame_rows["G"].mean()),
            "B":     float(frame_rows["B"].mean()),
        })
    return pd.DataFrame(records, columns=["frame", "time", "ROI", "R", "G", "B"])
# Main per-file function

def append_composite_rois(csv_path: str, fps: float,
                          hr_csv_path: str = None) -> None:
    df = pd.read_csv(csv_path)
    df = df.reset_index(drop=True)   # ← add this line

    required = {"frame", "time", "ROI", "R", "G", "B"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(f"CSV missing columns: {missing_cols}")

    present_rois = set(df["ROI"].unique())
    missing_rois = [r for r in TARGET_ROIS if r not in present_rois]
    if missing_rois:
        raise ValueError(f"Missing ROIs in CSV: {missing_rois}")

    if ROI29_NAME in present_rois:
        print(f"  [SKIP] Composite ROIs already present in {csv_path}")
        return

    # Backup
    stem        = os.path.splitext(csv_path)[0]
    backup_path = f"{stem}_backup.csv"
    shutil.copy2(csv_path, backup_path)
    print(f"  Backup → {backup_path}")

    # Extract per-ROI data
    frames      = sorted(df["frame"].unique())
    ref_df      = df[df["ROI"] == TARGET_ROIS[0]].sort_values("frame")
    time_lookup = dict(zip(ref_df["frame"], ref_df["time"]))
    roi_data    = {roi: df[df["ROI"] == roi].sort_values("frame").reset_index(drop=True)
                   for roi in TARGET_ROIS}

    # Per-SQI weight arrays
    # Each dict entry: label (weights array, success bool)
    weight_map = {}

    hr_available = hr_csv_path and os.path.isfile(hr_csv_path)

    # SQI columns to ROI names
    sqi_cols = {
        'SQI':     ROI30_NAME,   # combined SQI
        'SNR':     ROI31_NAME,   # spectral SNR (stored as linear in HR CSV)
        'AC_SQI':  ROI32_NAME,   # autocorrelation
        'FC_SQI':  ROI33_NAME,   # frequency consistency
        'CRA_SQI': ROI34_NAME,   # cross-ROI agreement
        'FF_AVG': ROI35_NAME
    }

    loaded_weights = {}   # col → np.ndarray

    if hr_available:
        for col in sqi_cols:
            try:
                vals = load_sqi_column(hr_csv_path, TARGET_ROIS, col)
                # SNR is stored as linear (not dB) in the HR CSV from compute_snr
                # so treat it the same as the [0,1] SQI columns
                w = normalise_weights(vals)
                loaded_weights[col] = w
                print(f"  [{col}] weights loaded successfully.")
            except Exception as e:
                print(f"  [WARN] Could not load '{col}' weights: {e}")

    # SNR fallback for any missing weight
    snr_w_fallback = None
    def _get_snr_fallback():
        nonlocal snr_w_fallback
        if snr_w_fallback is None:
            print("  Computing SNR fallback from raw RGB...")
            snr_w_fallback = snr_weights(roi_data, fps)
            for roi, w in zip(TARGET_ROIS, snr_w_fallback):
                print(f"  [{roi:35s}]  SNR weight = {w:.4f}")
        return snr_w_fallback

    def resolve(col):
        if col in loaded_weights:
            return loaded_weights[col]
        return _get_snr_fallback()

    # Print final weights
    print(f"\n  Weighting method source: "
          f"{'HR CSV' if hr_available else 'SNR fallback (no HR CSV)'}")
    for col, roi_name in sqi_cols.items():
        w = resolve(col)
        print(f"\n  {roi_name}  [{col}]:")
        for roi, wv in zip(TARGET_ROIS, w):
            print(f"    [{roi:35s}]  {wv:.4f}")

    # Build all ROIs
    roi29_df = build_roi29(roi_data, frames, time_lookup).reset_index(drop=True)
    roi30_df = build_roi30(roi_data, resolve('SQI'), frames, time_lookup).reset_index(drop=True)
    roi31_df = build_roi31(roi_data, resolve('SNR'), frames, time_lookup).reset_index(drop=True)
    roi32_df = build_roi32(roi_data, resolve('AC_SQI'), frames, time_lookup).reset_index(drop=True)
    roi33_df = build_roi33(roi_data, resolve('FC_SQI'), frames, time_lookup).reset_index(drop=True)
    roi34_df = build_roi34(roi_data, resolve('CRA_SQI'), frames, time_lookup).reset_index(drop=True)
    roi35_df = build_roi35(df, frames, time_lookup).reset_index(drop=True)

    for name, rdf in [(ROI29_NAME, roi29_df), (ROI30_NAME, roi30_df),
                      (ROI31_NAME, roi31_df), (ROI32_NAME, roi32_df),
                      (ROI33_NAME, roi33_df), (ROI34_NAME, roi34_df), (ROI35_NAME, roi35_df)]:
        print(f"  {name}: {len(rdf)} frames")

    # Append and save
    df_updated = pd.concat(
        [df, roi29_df, roi30_df, roi31_df, roi32_df, roi33_df, roi34_df, roi35_df],
        ignore_index=True
    )
    df_updated.to_csv(csv_path, index=False)
    print(f"\n  Updated CSV → {csv_path}  ({df_updated['ROI'].nunique()} ROIs per frame)")

# Pipeline entry point

def main_combine_rois(
    name_dataset: str = "custom",
    list_attendant=None,
    distances=None,
    fps: float = 50.0,
    algorithms: list = None,
) -> None:
    """
    Append ROIs 29 and 30 into each distance RGB CSV.
    Looks for an existing HR CSV to use AC_SQI weighting; falls back to SNR.

    Parameters
    ----------
    algorithms : Algorithm names to search for HR CSVs in priority order.
                 Uses the first one found per distance.
    """
    if distances is None:
        list_attendant = [1, 2]
        distances = [1, 2]  # , 3, 4, 5]
    if algorithms is None:
        algorithms = ['LGI', 'CHROM', 'OMIT', 'POS']

    dir_crt = os.getcwd()
    rgb_dir = os.path.join(dir_crt, "data", name_dataset, "rgb")
    hr_dir  = os.path.join(dir_crt, "data", name_dataset, "hr")
    for attendant_id in list_attendant:
        for dist in distances:
            csv_path = os.path.join(rgb_dir, f"{attendant_id}_{dist}.csv")

            if not os.path.isfile(csv_path):
                print(f"[ROI29/30] WARNING: RGB CSV not found, skipping → {csv_path}")
                continue

            # Find first available HR CSV for AC_SQI weighting
            hr_csv_path = None
            for alg in algorithms:
                candidate = os.path.join(hr_dir, f"{attendant_id}_{dist}{alg}1.csv")
                if os.path.isfile(candidate):
                    hr_csv_path = candidate
                    print(f"  Using HR CSV for AC_SQI weights: {candidate}")
                    break

            if hr_csv_path is None:
                print(f"  [INFO] No HR CSV found for dist={dist} — will use SNR fallback.")

            print(f"\n{'='*60}")
            print(f"[ROI29/30] attendant{attendant_id} | distance {dist}")
            print(f"{'='*60}")

            append_composite_rois(csv_path=csv_path, fps=fps, hr_csv_path=hr_csv_path)

    print("\n[ROI29/30] Done.")
    print("main_rgb2hr.py and main_evaluation.py will pick up ROIs 29 & 30 automatically.")


if __name__ == "__main__":
    main_combine_rois(
        name_dataset="custom",
        list_attendant=[1,2],
        distances=[1, 2, 3, 4, 5],
        fps=50.0,
        algorithms=['LGI', 'CHROM', 'OMIT', 'POS'],
    )