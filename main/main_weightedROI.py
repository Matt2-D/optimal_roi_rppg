#creates a 29th synthetic ROI from the 6 best predetermined ROIs, weighted by their individual SNR (spectral, via compute_snr logic).

#For each distance CSV (e.g. data/custom/rgb/1.csv), this script:
  #1. Extracts the 6 target ROI time-series
  #2. Computes per-ROI SNR using Welch PSD (same method as compute_snr)
  #3. Normalizes SNR weights (negatives clamped to 0)
  #4. Produces a weighted-average R, G, B per frame
  #5. Writes a new CSV alongside the source (e.g. 1_roi29.csv)
#

import os
import numpy as np
import pandas as pd
from scipy.signal import welch


# ── Configuration ─────────────────────────────────────────────────────────────

# The 6 predetermined best ROIs (must match ROI column strings exactly)
TARGET_ROIS = [
    "glabella",
    "lower medial forehead",
    "right malar",
    "left malar",
    "left lower lateral forehead",
    "right lower lateral forehead",
]

ROI29_NAME = "snr_weighted_composite"

# Physiological rPPG frequency range
MIN_HZ = 0.7
MAX_HZ = 2.5

# Welch parameters
NFFT = 4096


# ── SNR helper (mirrors compute_snr logic, operates on a 1-D numpy array) ────

def _compute_snr(signal: np.ndarray, fps: float) -> float:
    """
    Spectral SNR for a 1-D RGB channel signal.
    Mirrors the compute_snr method: fundamental ±2 bins + conditional harmonic.
    Returns -inf if computation fails (so weight clamps to 0).
    """
    signal = signal - np.mean(signal)           # remove DC
    if np.all(signal == 0):
        return -np.inf

    freqs, power = welch(
        signal,
        fs=fps,
        nperseg=min(NFFT, len(signal)),
        nfft=NFFT,
        window="hann",
        scaling="density",
    )

    # Restrict to physiological band
    band = (freqs >= MIN_HZ) & (freqs <= MAX_HZ)
    freqs = freqs[band]
    power = power[band]

    if len(power) == 0:
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
    """
    Average SNR across R, G, B channels for one ROI.
    Green channel typically dominates in rPPG but all three are included.
    """
    snrs = []
    for ch in ["R", "G", "B"]:
        s = _compute_snr(roi_df[ch].values.astype(np.float64), fps)
        if np.isfinite(s):
            snrs.append(s)
    return float(np.mean(snrs)) if snrs else -np.inf


# ── Main builder ──────────────────────────────────────────────────────────────

def build_roi29(
    csv_path: str,
    fps: float,
    output_path: str
) -> pd.DataFrame:

    df = pd.read_csv(csv_path)

    # Validate required columns
    required = {"frame", "time", "ROI", "R", "G", "B"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}  |  found: {list(df.columns)}")

    # Validate all 6 target ROIs are present
    present_rois = set(df["ROI"].unique())
    missing_rois = [r for r in TARGET_ROIS if r not in present_rois]
    if missing_rois:
        raise ValueError(f"Missing ROIs in CSV: {missing_rois}")

    frames = sorted(df["frame"].unique())

    # ── Step 1: extract per-ROI full time-series and compute SNR ─────────────
    roi_data   = {}   # roi_name → DataFrame (indexed by frame, sorted)
    roi_snr    = {}   # roi_name → float SNR

    for roi in TARGET_ROIS:
        roi_df = df[df["ROI"] == roi].sort_values("frame").reset_index(drop=True)
        roi_data[roi] = roi_df
        snr = _channel_snr(roi_df, fps)
        roi_snr[roi] = snr
        print(f"  [{roi:35s}]  SNR = {snr:+.2f} dB")

    # ── Step 2: compute normalized weights (clamp negatives to 0) ────────────
    snr_values = np.array([roi_snr[r] for r in TARGET_ROIS])

    # Shift so minimum finite value → 0, then clamp
    finite_mask = np.isfinite(snr_values)
    if not np.any(finite_mask):
        raise RuntimeError("All 6 ROIs returned non-finite SNR. Cannot build ROI 29.")

    # Linear SNR for weighting (convert from dB)
    linear_snr = np.where(finite_mask, 10 ** (snr_values / 10), 0.0)
    linear_snr = np.clip(linear_snr, 0, None)

    weight_sum = linear_snr.sum()
    if weight_sum == 0:
        raise RuntimeError("All SNR weights are zero. Cannot build ROI 29.")

    weights = linear_snr / weight_sum   # normalized, sums to 1.0

    print("\n  Normalized weights:")
    for roi, w in zip(TARGET_ROIS, weights):
        print(f"  [{roi:35s}]  weight = {w:.4f}")

    # ── Step 3: weighted average R, G, B per frame ────────────────────────────
    records = []
    for frame in frames:
        # Get time for this frame from any ROI
        time_val = roi_data[TARGET_ROIS[0]].loc[
            roi_data[TARGET_ROIS[0]]["frame"] == frame, "time"
        ]
        time_val = float(time_val.iloc[0]) if len(time_val) > 0 else np.nan

        r_weighted = 0.0
        g_weighted = 0.0
        b_weighted = 0.0

        for roi, w in zip(TARGET_ROIS, weights):
            row = roi_data[roi][roi_data[roi]["frame"] == frame]
            if len(row) == 0:
                continue    # missing frame for this ROI — skip contribution
            r_weighted += w * float(row["R"].iloc[0])
            g_weighted += w * float(row["G"].iloc[0])
            b_weighted += w * float(row["B"].iloc[0])

        records.append({
            "frame": frame,
            "time":  time_val,
            "ROI":   ROI29_NAME,
            "R":     r_weighted,
            "G":     g_weighted,
            "B":     b_weighted,
        })

    roi29_df = pd.DataFrame(records, columns=["frame", "time", "ROI", "R", "G", "B"])

    # ── Step 4: write output ──────────────────────────────────────────────────
    if output_path is None:
        stem = os.path.splitext(csv_path)[0]
        output_path = f"{stem}_roi29.csv"

    roi29_df.to_csv(output_path, index=False)
    print(f"\n  ROI 29 saved → {output_path}  ({len(roi29_df)} frames)")

    return roi29_df


# ── Pipeline entry point ──────────────────────────────────────────────────────

def main_combine_roi29(
    name_dataset: str = "custom",
    attendant_id: int = 1,
    distances: list[int] = None,
    fps: float = 50.0,
):
    if distances is None:
        distances = [1, 2, 3]

    dir_crt  = os.getcwd()
    rgb_dir  = os.path.join(dir_crt, "data", name_dataset, "rgb")

    for dist in distances:
        csv_path = os.path.join(rgb_dir, f"{dist}.csv")

        if not os.path.isfile(csv_path):
            print(f"[ROI29] WARNING: CSV not found, skipping → {csv_path}")
            continue

        print(f"\n{'='*60}")
        print(f"[ROI29] attendant{attendant_id} | distance {dist}")
        print(f"{'='*60}")

        output_path = os.path.join(rgb_dir, f"{dist}_roi29.csv")
        build_roi29(csv_path=csv_path, fps=fps, output_path=output_path)

    print("\n[ROI29] Done.")


if __name__ == "__main__":
    main_combine_roi29(
        name_dataset="custom",
        attendant_id=1,
        distances=[1, 2, 3],
        fps=50.0,
    )