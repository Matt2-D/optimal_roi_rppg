# Synthetic composite ROIs 29-35 built from the 6 best predetermined ROIs.
# Runs AFTER main_vid2rgb and BEFORE main_rgb2hr.
# ALL metrics computed from raw RGB — no HR CSV needed.
#
# ROI 29 — simple_composite        : equal-weight average
# ROI 30 — sqi_weighted_composite  : combined SQI (SNR+AC+FC+CRA)
# ROI 31 — snr_weighted_composite  : spectral SNR only
# ROI 32 — ac_weighted_composite   : autocorrelation SQI only
# ROI 33 — fc_weighted_composite   : frequency-consistency SQI only
# ROI 34 — cra_weighted_composite  : cross-ROI-agreement SQI only
# ROI 35 — full_frame_average      : mean of ALL 28 ROIs per frame

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
ROI36_NAME = "snr_weighted_all_rois"

MIN_HZ = 0.7
MAX_HZ = 2.5
NFFT   = 4096


# Metric functions (all operate on raw RGB DataFrames)

def _snr_db(roi_df: pd.DataFrame, fps: float) -> float:
    """Spectral SNR in dB averaged across R, G, B channels."""
    results = []
    for ch in ["R", "G", "B"]:
        sig = roi_df[ch].values.astype(np.float64)
        sig = sig - np.mean(sig)
        if np.all(sig == 0) or not np.any(np.isfinite(sig)):
            continue
        sig = np.nan_to_num(sig, nan=0.0)
        freqs, power = welch(sig, fs=fps,
                             nperseg=min(NFFT, len(sig)),
                             nfft=NFFT, window="hann", scaling="density")
        band = (freqs >= MIN_HZ) & (freqs <= MAX_HZ)
        f, p = freqs[band], power[band]
        if len(p) == 0:
            continue
        mi       = np.argmax(p)
        pbins    = np.arange(max(0, mi-2), min(len(p), mi+3))
        mask     = np.ones(len(p), dtype=bool); mask[pbins] = False
        ts       = np.sum(p[pbins])
        hf       = f[mi] * 2
        if hf <= MAX_HZ:
            hi    = np.argmin(np.abs(f - hf))
            hbins = np.arange(max(0, hi-2), min(len(p), hi+3))
            hp    = np.sum(p[hbins])
            if hp > 0.1 * ts:
                ts += hp; mask[hbins] = False
        noise = np.sum(p[mask])
        if noise > 0:
            results.append(10 * np.log10(ts / noise))
    return float(np.mean(results)) if results else -np.inf


def _ac_sqi(roi_df: pd.DataFrame, fps: float) -> float:
    """Autocorrelation SQI on green channel. Returns [0, 1]."""
    sig = roi_df["G"].values.astype(np.float64)
    sig = sig - np.mean(sig)
    sig = np.nan_to_num(sig, nan=0.0)
    if np.all(sig == 0):
        return 0.0
    corr = np.correlate(sig, sig, mode='full')
    corr = corr[len(corr) // 2:]
    if corr[0] == 0:
        return 0.0
    corr = corr / corr[0]
    lag_min = max(1, int(fps / MAX_HZ))
    lag_max = int(fps / MIN_HZ)
    if lag_max >= len(corr) or lag_min >= lag_max:
        return 0.0
    return float(np.clip(np.max(corr[lag_min:lag_max]), 0, 1))


def _fc_sqi(roi_df: pd.DataFrame, fps: float,
             window_sec: float = 6.0, stride_sec: float = 1.0) -> float:
    """Frequency consistency across sliding windows on green channel. Returns [0, 1]."""
    sig = roi_df["G"].values.astype(np.float64)
    sig = sig - np.mean(sig)
    sig = np.nan_to_num(sig, nan=0.0)
    win    = int(window_sec * fps)
    stride = int(stride_sec * fps)
    if len(sig) < win:
        return 1.0
    bpms = []
    idx = 0
    while idx + win <= len(sig):
        seg = sig[idx:idx + win]
        freqs, power = welch(seg, fs=fps,
                             nperseg=min(256, win),
                             window="hann", scaling="density")
        band = (freqs >= MIN_HZ) & (freqs <= MAX_HZ)
        if np.any(band):
            bpms.append(freqs[band][np.argmax(power[band])] * 60.0)
        idx += stride
    if len(bpms) < 2:
        return 1.0
    return float(np.exp(-np.std(bpms) / 10.0))


def _dominant_bpm(roi_df: pd.DataFrame, fps: float) -> float:
    """Dominant frequency in BPM from green channel."""
    sig = roi_df["G"].values.astype(np.float64)
    sig = sig - np.mean(sig)
    sig = np.nan_to_num(sig, nan=0.0)
    freqs, power = welch(sig, fs=fps,
                         nperseg=min(NFFT, len(sig)),
                         nfft=NFFT, window="hann", scaling="density")
    band = (freqs >= MIN_HZ) & (freqs <= MAX_HZ)
    if not np.any(band):
        return np.nan
    return float(freqs[band][np.argmax(power[band])] * 60.0)


def _compute_all_metrics(roi_data: dict, fps: float) -> dict:
    """
    Compute SNR, AC, FC, CRA, and combined SQI for all TARGET_ROIS
    directly from raw RGB. Returns dict of metric → np.ndarray[6].
    Each array contains values in [0,1] ready for normalise_weights().
    """
    n = len(TARGET_ROIS)

    # SNR in dB → [0,1] via typical rPPG range [-5, +15 dB]
    snr_db   = np.array([_snr_db(roi_data[r], fps) for r in TARGET_ROIS])
    snr_01   = np.clip((snr_db + 5) / 20.0, 0, 1)

    # Autocorrelation SQI — already [0,1]
    ac_vals  = np.array([_ac_sqi(roi_data[r], fps) for r in TARGET_ROIS])

    # Frequency consistency SQI — already [0,1]
    fc_vals  = np.array([_fc_sqi(roi_data[r], fps) for r in TARGET_ROIS])

    # Cross-ROI agreement: fraction of other ROIs within 5 BPM
    dom_bpms = np.array([_dominant_bpm(roi_data[r], fps) for r in TARGET_ROIS])
    cra_vals = np.zeros(n)
    for i in range(n):
        if not np.isfinite(dom_bpms[i]):
            continue
        others = np.delete(dom_bpms, i)
        valid  = others[np.isfinite(others)]
        if len(valid) > 0:
            cra_vals[i] = float(np.mean(np.abs(valid - dom_bpms[i]) <= 5.0))

    # Combined SQI
    combined = (0.35 * snr_01
              + 0.25 * fc_vals
              + 0.20 * ac_vals
              + 0.20 * cra_vals)

    # Print computed values
    print(f"\n  Raw metric values per ROI:")
    header = f"  {'ROI':35s}  {'SNR_01':>7}  {'AC':>7}  {'FC':>7}  {'CRA':>7}  {'SQI':>7}"
    print(header)
    for i, roi in enumerate(TARGET_ROIS):
        print(f"  {roi:35s}  {snr_01[i]:7.4f}  {ac_vals[i]:7.4f}  "
              f"{fc_vals[i]:7.4f}  {cra_vals[i]:7.4f}  {combined[i]:7.4f}")

    return {
        'SNR':     snr_01,
        'AC_SQI':  ac_vals,
        'FC_SQI':  fc_vals,
        'CRA_SQI': cra_vals,
        'SQI':     combined,
    }


# Weight normalization

def normalise_weights(values: np.ndarray) -> np.ndarray:
    """Clamp non-finite/negative to 0, normalise to sum=1."""
    w = np.where(np.isfinite(values), values, 0.0)
    w = np.clip(w, 0, None)
    s = w.sum()
    if s == 0:
        # All equal fallback — avoids crash on all-zero metrics
        return np.full(len(values), 1.0 / len(values))
    return w / s


# ROI builders

def _weighted_composite(roi_data: dict, weights: np.ndarray,
                         frames: list, time_lookup: dict,
                         roi_name: str) -> pd.DataFrame:
    records = []
    for frame in frames:
        r = g = b = 0.0
        for roi, w in zip(TARGET_ROIS, weights):
            row = roi_data[roi][roi_data[roi]["frame"] == frame]
            if len(row) == 0:
                continue
            r += w * float(row["R"].iloc[0])
            g += w * float(row["G"].iloc[0])
            b += w * float(row["B"].iloc[0])
        records.append({"frame": frame,
                        "time":  time_lookup.get(frame, np.nan),
                        "ROI":   roi_name,
                        "R": r, "G": g, "B": b})
    return pd.DataFrame(records, columns=["frame", "time", "ROI", "R", "G", "B"])


def _simple_composite(roi_data: dict, frames: list,
                       time_lookup: dict) -> pd.DataFrame:
    n = len(TARGET_ROIS)
    return _weighted_composite(roi_data, np.full(n, 1.0/n),
                                frames, time_lookup, ROI29_NAME)


def _full_frame_average(df: pd.DataFrame, frames: list,
                         time_lookup: dict) -> pd.DataFrame:
    """Average of ALL ROIs per frame (not just the 6 target ROIs)."""
    records = []
    for frame in frames:
        rows = df[df["frame"] == frame]
        records.append({"frame": frame,
                        "time":  time_lookup.get(frame, np.nan),
                        "ROI":   ROI35_NAME,
                        "R":     float(rows["R"].mean()),
                        "G":     float(rows["G"].mean()),
                        "B":     float(rows["B"].mean())})
    return pd.DataFrame(records, columns=["frame", "time", "ROI", "R", "G", "B"])

# SNR from Full frame average
def _snr_weighted_all_rois(df: pd.DataFrame, frames: list,
                            time_lookup: dict, fps: float) -> pd.DataFrame:
    """
    ROI 36 — SNR-weighted average across ALL 28 original ROIs per frame.
    SNR is computed from the green channel of each ROI's full time series.
    """
    all_rois = [roi for roi in df["ROI"].unique()
                if roi not in {ROI29_NAME, ROI30_NAME, ROI31_NAME,
                               ROI32_NAME, ROI33_NAME, ROI34_NAME, ROI35_NAME}]

    # Compute SNR for each ROI
    snr_vals = {}
    for roi in all_rois:
        roi_df = df[df["ROI"] == roi].sort_values("frame").reset_index(drop=True)
        snr_vals[roi] = _snr_db(roi_df, fps)

    # Convert dB → [0,1] then normalise
    rois_arr = list(snr_vals.keys())
    snr_arr  = np.array([snr_vals[r] for r in rois_arr])
    snr_01   = np.clip((snr_arr + 5) / 20.0, 0, 1)
    weights  = normalise_weights(snr_01)

    print(f"\n  ROI 36 SNR weights (all {len(rois_arr)} ROIs):")
    for roi, w in zip(rois_arr, weights):
        print(f"    {roi:35s}  {w:.4f}")

    # Pre-index each ROI by frame for speed
    roi_frames = {roi: df[df["ROI"] == roi].set_index("frame")
                  for roi in rois_arr}

    records = []
    for frame in frames:
        r = g = b = 0.0
        for roi, w in zip(rois_arr, weights):
            try:
                row = roi_frames[roi].loc[frame]
                r += w * float(row["R"])
                g += w * float(row["G"])
                b += w * float(row["B"])
            except KeyError:
                continue
        records.append({"frame": frame,
                        "time":  time_lookup.get(frame, np.nan),
                        "ROI":   ROI36_NAME,
                        "R": r, "G": g, "B": b})
    return pd.DataFrame(records, columns=["frame", "time", "ROI", "R", "G", "B"])

# Main per-file function

def append_composite_rois(csv_path: str, fps: float,
                          hr_csv_path: str = None) -> None:
    """
    Append ROIs 29-35 into the RGB CSV in-place.
    All SQI metrics are computed directly from raw RGB — no HR CSV needed.
    hr_csv_path is accepted but ignored (kept for pipeline compatibility).
    """
    df = pd.read_csv(csv_path).reset_index(drop=True)

    required = {"frame", "time", "ROI", "R", "G", "B"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    present = set(df["ROI"].unique())
    missing_rois = [r for r in TARGET_ROIS if r not in present]
    if missing_rois:
        raise ValueError(f"Target ROIs missing from CSV: {missing_rois}")

    # Always remove any existing composite ROIs and rebuild from scratch
    composite_names = {ROI29_NAME, ROI30_NAME, ROI31_NAME,
                       ROI32_NAME, ROI33_NAME, ROI34_NAME, ROI35_NAME, ROI36_NAME}
    if composite_names & present:
        print(f"  [INFO] Removing existing composite ROIs — rebuilding.")
        df = df[~df["ROI"].isin(composite_names)].reset_index(drop=True)
        # Update present and frames after removal
        present = set(df["ROI"].unique())
        frames = sorted(df["frame"].unique())

    # Backup
    stem = os.path.splitext(csv_path)[0]
    shutil.copy2(csv_path, f"{stem}_backup.csv")
    print(f"  Backup → {stem}_backup.csv")

    # Per-ROI data lookup
    frames      = sorted(df["frame"].unique())
    ref_df      = df[df["ROI"] == TARGET_ROIS[0]].sort_values("frame")
    time_lookup = dict(zip(ref_df["frame"], ref_df["time"]))
    roi_data    = {roi: df[df["ROI"] == roi].sort_values("frame").reset_index(drop=True)
                   for roi in TARGET_ROIS}

    #  Compute all metrics from raw RGB
    print("  Computing SQI metrics from raw RGB...")
    metrics = _compute_all_metrics(roi_data, fps)

    # Normalise each metric independently into weights
    w_snr = normalise_weights(metrics['SNR'])
    w_ac  = normalise_weights(metrics['AC_SQI'])
    w_fc  = normalise_weights(metrics['FC_SQI'])
    w_cra = normalise_weights(metrics['CRA_SQI'])
    w_sqi = normalise_weights(metrics['SQI'])

    # Print final weights
    print(f"\n  Normalised weights:")
    header = f"  {'ROI':35s}  {'SNR':>7}  {'AC':>7}  {'FC':>7}  {'CRA':>7}  {'SQI':>7}"
    print(header)
    for i, roi in enumerate(TARGET_ROIS):
        print(f"  {roi:35s}  {w_snr[i]:7.4f}  {w_ac[i]:7.4f}  "
              f"{w_fc[i]:7.4f}  {w_cra[i]:7.4f}  {w_sqi[i]:7.4f}")

    # Build ROIs
    roi29_df = _simple_composite(roi_data, frames, time_lookup).reset_index(drop=True)
    roi30_df = _weighted_composite(roi_data, w_sqi, frames, time_lookup, ROI30_NAME).reset_index(drop=True)
    roi31_df = _weighted_composite(roi_data, w_snr, frames, time_lookup, ROI31_NAME).reset_index(drop=True)
    roi32_df = _weighted_composite(roi_data, w_ac,  frames, time_lookup, ROI32_NAME).reset_index(drop=True)
    roi33_df = _weighted_composite(roi_data, w_fc,  frames, time_lookup, ROI33_NAME).reset_index(drop=True)
    roi34_df = _weighted_composite(roi_data, w_cra, frames, time_lookup, ROI34_NAME).reset_index(drop=True)
    roi35_df = _full_frame_average(df, frames, time_lookup).reset_index(drop=True)
    roi36_df = _snr_weighted_all_rois(df, frames, time_lookup, fps).reset_index(drop=True)

    print()
    for name, rdf in [(ROI29_NAME, roi29_df), (ROI30_NAME, roi30_df),
                      (ROI31_NAME, roi31_df), (ROI32_NAME, roi32_df),
                      (ROI33_NAME, roi33_df), (ROI34_NAME, roi34_df),
                      (ROI35_NAME, roi35_df),(ROI36_NAME, roi36_df)]:
        print(f"  {name}: {len(rdf)} frames")

    df_updated = pd.concat(
        [df, roi29_df, roi30_df, roi31_df, roi32_df,
         roi33_df, roi34_df, roi35_df, roi36_df],
        ignore_index=True
    )
    df_updated.to_csv(csv_path, index=False)
    print(f"\n  Updated → {csv_path}  ({df_updated['ROI'].nunique()} ROIs per frame)")


# Pipeline entry point

def main_combine_rois(
    name_dataset: str = "custom",
    list_attendant=None,
    distances=None,
    fps: float = 50.0,
    algorithms: list = None,
) -> None:
    if list_attendant is None:
        list_attendant = [1]
    if distances is None:
        distances = [1, 2, 3, 4, 5]
    if algorithms is None:
        algorithms = ['LGI', 'CHROM', 'OMIT', 'POS']

    dir_crt = os.getcwd()
    rgb_dir = os.path.join(dir_crt, "data", name_dataset, "rgb")
    hr_dir  = os.path.join(dir_crt, "data", name_dataset, "hr")

    for attendant_id in list_attendant:
        for dist in distances:
            csv_path = os.path.join(rgb_dir, f"{attendant_id}_{dist}.csv")

            if not os.path.isfile(csv_path):
                print(f"[ROI] WARNING: RGB CSV not found → {csv_path}")
                continue

            # HR CSV accepted but not required — kept for compatibility
            hr_csv_path = None
            for alg in algorithms:
                candidate = os.path.join(hr_dir, f"{attendant_id}_{dist}_{alg}1.csv")
                if os.path.isfile(candidate):
                    hr_csv_path = candidate
                    break

            print(f"\n{'='*60}")
            print(f"[ROI29-35] attendant{attendant_id} | distance {dist}")
            print(f"{'='*60}")

            append_composite_rois(csv_path=csv_path, fps=fps,
                                  hr_csv_path=hr_csv_path)

    print("\n[ROI29-35] Done.")


if __name__ == "__main__":
    main_combine_rois(
        name_dataset="custom",
        list_attendant=[1, 2],
        distances=[1, 2, 3, 4, 5],
        fps=50.0,
        algorithms=['LGI', 'CHROM', 'OMIT', 'POS'],
    )