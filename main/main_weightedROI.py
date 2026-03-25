#creates a 29th synthetic ROI from the 6 best predetermined ROIs, weighted by their individual SNR
#creates a 30th synthetic ROI from the 6 best predetermined ROI's, unweighted
#For each distance CSV (e.g. data/custom/rgb/1.csv), this script:
  #1. Extracts the 6 target ROI time-series
  #2. Computes per-ROI SNR using Welch PSD (same method as compute_snr)
  #3. Normalizes SNR weights (negatives clamped to 0)
  #4. Produces a weighted-average R, G, B per frame
  #5. overwrites the RGB CSV to include the two new ROI's

# Author: Matthew Dowell
# Date 3/10/2026
import os
import shutil

import numpy as np
import pandas as pd
from scipy.signal import welch


# The 6 predetermined best ROIs
TARGET_ROIS = [
    "glabella",
    "lower medial forehead",
    "right malar",
    "left malar",
    "left lower lateral forehead",
    "right lower lateral forehead",
]

ROI29_NAME = "snr_weighted_composite"
ROI30_NAME = "simple_composite"

# Physiological rPPG frequency range
MIN_HZ = 0.7
MAX_HZ = 2.5

# Welch parameters
NFFT = 4096


def compute_snr2(signal: np.ndarray, fps: float) -> float:
    signal = signal - np.mean(signal)
    if np.all(signal == 0):
        print(f"    [SNR diag] all-zero signal, len={len(signal)}")
        return -np.inf
    if np.all(~np.isfinite(signal)):
        print(f"    [SNR diag] all-NaN/inf signal, len={len(signal)}")
        return -np.inf

    # Replace any NaN with 0 before Welch
    signal = np.nan_to_num(signal, nan=0.0)

    freqs, power = welch(
        signal,
        fs=fps,
        nperseg=min(NFFT, len(signal)),
        nfft=NFFT,
        window="hann",
        scaling="density",
    )

    band = (freqs >= MIN_HZ) & (freqs <= MAX_HZ)
    freqs = freqs[band]
    power = power[band]

    if len(power) == 0:
        print(f"    [SNR diag] no bins in [{MIN_HZ},{MAX_HZ}] Hz. "
              f"fps={fps}, signal_len={len(signal)}, "
              f"freq_range=[{freqs.min() if len(freqs) else 'N/A'},"
              f"{freqs.max() if len(freqs) else 'N/A'}]")
        return -np.inf

    # Fundamental ±2 bins
    max_index = np.argmax(power)
    peak_bins = np.arange(max(0, max_index - 2), min(len(power), max_index + 3))
    fundamental_freq = freqs[max_index]

    mask = np.ones(len(power), dtype=bool)
    mask[peak_bins] = False
    total_signal = np.sum(power[peak_bins])

    # Conditional harmonic
    harmonic_freq = fundamental_freq * 2
    if harmonic_freq <= MAX_HZ:
        harm_idx = np.argmin(np.abs(freqs - harmonic_freq))
        harm_bins = np.arange(max(0, harm_idx - 2), min(len(power), harm_idx + 3))
        harm_power = np.sum(power[harm_bins])
        if harm_power > 0.1 * total_signal:
            total_signal += harm_power
            mask[harm_bins] = False

    total_noise = np.sum(power[mask])
    if total_noise <= 0:
        return -np.inf

    return 10 * np.log10(total_signal / total_noise)


def _channel_snr(roi_df: pd.DataFrame, fps: float) -> float:
    # Average SNR across R, G, B channels for one ROI.
    snrs = []
    for ch in ["R", "G", "B"]:
        s = compute_snr2(roi_df[ch].values.astype(np.float64), fps)
        if np.isfinite(s):
            snrs.append(s)
    return float(np.mean(snrs)) if snrs else -np.inf

def _build_roi29(roi_data: dict, weights: np.ndarray, frames: list,
                 time_lookup: dict) -> pd.DataFrame:
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
            "ROI":   ROI29_NAME,
            "R":     r,
            "G":     g,
            "B":     b,
        })
    return pd.DataFrame(records, columns=["frame", "time", "ROI", "R", "G", "B"])


def _build_roi30(roi_data: dict, frames: list,
                 time_lookup: dict) -> pd.DataFrame:
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
            "time":  time_lookup.get(frame, np.nan),
            "ROI":   ROI30_NAME,
            "R":     float(np.mean(r_vals)) if r_vals else np.nan,
            "G":     float(np.mean(g_vals)) if g_vals else np.nan,
            "B":     float(np.mean(b_vals)) if b_vals else np.nan,
        })
    return pd.DataFrame(records, columns=["frame", "time", "ROI", "R", "G", "B"])


def append_composite_rois(csv_path: str, fps: float) -> None:
    df = pd.read_csv(csv_path)

    required = {"frame", "time", "ROI", "R", "G", "B"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(f"CSV missing columns: {missing_cols}")

    present_rois = set(df["ROI"].unique())
    missing_rois = [r for r in TARGET_ROIS if r not in present_rois]
    if missing_rois:
        raise ValueError(f"Missing ROIs in CSV: {missing_rois}")

    # Guard: skip if already appended (idempotent)
    if ROI29_NAME in present_rois or ROI30_NAME in present_rois:
        print(f"  [SKIP] Composite ROIs already present in {csv_path}")
        return

    stem        = os.path.splitext(csv_path)[0]
    backup_path = f"{stem}_backup.csv"
    shutil.copy2(csv_path, backup_path)
    print(f"  Backup → {backup_path}")

    # per-ROI data & SNR
    frames      = sorted(df["frame"].unique())
    ref_df      = df[df["ROI"] == TARGET_ROIS[0]].sort_values("frame")
    time_lookup = dict(zip(ref_df["frame"], ref_df["time"]))

    roi_data = {}
    roi_snr  = {}
    for roi in TARGET_ROIS:
        roi_df        = df[df["ROI"] == roi].sort_values("frame").reset_index(drop=True)
        roi_data[roi] = roi_df
        snr           = _channel_snr(roi_df, fps)
        roi_snr[roi]  = snr
        print(f"  [{roi:35s}]  SNR = {snr:+.2f} dB")

    # SNR weights
    snr_values  = np.array([roi_snr[r] for r in TARGET_ROIS])
    finite_mask = np.isfinite(snr_values)

    if not np.any(finite_mask):
        raise RuntimeError("All 6 ROIs returned non-finite SNR. Cannot build ROI 29.")

    linear_snr = np.where(finite_mask, 10 ** (snr_values / 10), 0.0)
    linear_snr = np.clip(linear_snr, 0, None)
    weight_sum = linear_snr.sum()

    if weight_sum == 0:
        raise RuntimeError("All SNR weights are zero. Cannot build ROI 29.")

    weights = linear_snr / weight_sum

    print("\n  Normalized SNR weights (ROI 29):")
    for roi, w in zip(TARGET_ROIS, weights):
        print(f"  [{roi:35s}]  weight = {w:.4f}")


    roi29_df = _build_roi29(roi_data, weights, frames, time_lookup)
    roi30_df = _build_roi30(roi_data, frames, time_lookup)

    print(f"\n  ROI 29 ({ROI29_NAME}): {len(roi29_df)} frames")
    print(f"  ROI 30 ({ROI30_NAME}): {len(roi30_df)} frames")

    # overwrite main CSV
    df_updated = pd.concat([df, roi29_df, roi30_df], ignore_index=True)
    df_updated.to_csv(csv_path, index=False)

    total_rois = df_updated["ROI"].nunique()
    print(f"\n  Updated CSV → {csv_path}  ({total_rois} ROIs per frame)")

def main_combine_rois(
    name_dataset: str = "custom",
    attendant_id: int = 1,
    distances = None,
    fps: float = 50.0,
) -> None:

    if distances is None:
        distances = [1, 2, 3, 4, 5]

    dir_crt = os.getcwd()
    rgb_dir = os.path.join(dir_crt, "data", name_dataset, "rgb")

    for dist in distances:
        csv_path = os.path.join(rgb_dir, f"{dist}.csv")

        if not os.path.isfile(csv_path):
            print(f"[ROI29/30] WARNING: CSV not found, skipping → {csv_path}")
            continue

        print(f"\n{'='*60}")
        print(f"[ROI29/30] attendant{attendant_id} | distance {dist}")
        print(f"{'='*60}")

        append_composite_rois(csv_path=csv_path, fps=fps)

    print("\n[ROI29/30] Done.")
    print("main_rgb2hr.py and main_evaluation.py will pick up ROIs 29 & 30 automatically.")



if __name__ == "__main__":
    main_combine_rois(
        name_dataset="custom",
        attendant_id=1,
        distances=[1, 2, 3, 4, 5],
        fps=50.0,
    )