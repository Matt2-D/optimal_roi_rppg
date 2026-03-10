"""
Generate ground truth BPM data using existing BVP data for UBFC-Phys dataset.
"""

# Author: Shuo Li
# Date: 2023/08/18
# Editor: Matthew Dowell
# Date: 03/10/2026

import warnings

from scipy.interpolate import interp1d

warnings.filterwarnings("ignore")  # Ignore unnecessary warnings.
import os
import sys
import cv2
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm
dir_crt = os.getcwd()
sys.path.append(os.path.join(dir_crt, 'util'))
from util import util_pyVHR

LEN_WINDOW     = 6      # Window length (seconds)
STRIDE_WINDOW  = 1      # Window stride (seconds)
NFFT           = 2048
MIN_HZ         = 0.65
MAX_HZ         = 4.0
FPS            = 50
NUM_FRAMES     = 3000   # Video frame count

def _estimate_bpm(sig_bvp: np.ndarray, fps: float) -> np.ndarray:
    """
    Sliding-window Welch BPM estimation on a 1-D BVP signal.
    Returns a per-frame BPM array (zeros filled by linear interpolation).
    """
    len_window_frame   = int(LEN_WINDOW * fps)
    stride_window_frame = int(STRIDE_WINDOW * fps)
    sig_bpm = np.zeros(len(sig_bvp))

    idx_crt = 0
    while (idx_crt + len_window_frame - 1) <= (len(sig_bvp) - 1):
        sig_slice = sig_bvp[idx_crt : idx_crt + len_window_frame]   # fixed: no -1 off-by-one

        # Welch expects shape [1, N]
        Pfreqs, Power = util_pyVHR.Welch(
            np.reshape(sig_slice, [1, len(sig_slice)]),
            fps, MIN_HZ, MAX_HZ, NFFT
        )
        Pmax = np.argmax(Power, axis=1)
        centre_idx = int(0.5 * (2 * idx_crt + len_window_frame - 1))
        sig_bpm[centre_idx] = float(Pfreqs.squeeze()[Pmax.squeeze()]) * 60.0   # Hz → BPM

        idx_crt += stride_window_frame

    # Interpolate zeros
    sig_bpm[sig_bpm == 0] = np.nan
    s = pd.Series(sig_bpm)
    s = s.interpolate(method='linear').ffill().bfill()
    return s.values

def main_gen_gtHR(dir_dataset):
    """Main function for generating ground truth BPM data for UBFC-Phys dataset.
    Parameters
    ----------
    dir_dataset: Directory of the dataset (UBFC-Phys).
    Params: A class containing the pre-defined parameters for the preliminary analysis.
    
    Returns
    -------

    """

    len_window = 6   # Window length in seconds.
    stride_window = 1   # Window stride in seconds.
    nFFT = 2048//1   # Freq. Resolution for STFTs.
    minHz = 0.65  # Minimal frequency in Hz.
    maxHz = 4.0   # Maximal frequency in Hz.

    # List of attendants.
    list_attendant = [1]
    # List of conditions.
    distances = [1, 2, 3]
    # Output directory
    dir_out = os.path.join(os.getcwd(), 'data', 'custom', 'gtHR')
    os.makedirs(dir_out, exist_ok=True)

    for num_attendant in tqdm(list_attendant, desc="Attendants"):
        for dist in tqdm(distances, desc=f"  Distances (att{num_attendant})", leave=False):

            # ── Locate ground truth CSV ───────────────────────────────────────
            # Expected location: <dataset>/attendant<n>/<dist>/pulse_data.csv
            # Searches one level of subfolders to handle timestamped subdirs.
            dist_folder = os.path.join(
                dir_dataset, f'attendant{num_attendant}', str(dist)
            )
            gt_csv = None
            # Direct location
            candidate = os.path.join(dist_folder, 'pulse_data.csv')
            if os.path.isfile(candidate):
                gt_csv = candidate
            else:
                # Search one subfolder deep (timestamped dirs)
                for sub in os.listdir(dist_folder):
                    candidate = os.path.join(dist_folder, sub, 'pulse_data.csv')
                    if os.path.isfile(candidate):
                        gt_csv = candidate
                        break

            if gt_csv is None:
                print(f"  [WARN] pulse_data.csv not found for att{num_attendant} dist{dist} — skipping")
                continue

            print(f"\n  Loading: {gt_csv}")
            df_gt =  pd.read_csv(gt_csv, index_col=None)

            # ── Extract and align BVP signal ──────────────────────────────────
            sig_bvp_raw = df_gt['Signal_Value'].values.astype(np.float64)

            # Normalise to zero-mean (removes DC offset ~500)
            sig_bvp_raw = sig_bvp_raw - np.mean(sig_bvp_raw)

            # Resample frames
            src_idx = np.linspace(0, 1, len(sig_bvp_raw))
            tgt_idx = np.linspace(0, 1, NUM_FRAMES)
            sig_bvp = interp1d(src_idx, sig_bvp_raw, kind='linear')(tgt_idx)

            # ── Option A: use sparse HR column directly ───────────────────────
            # Extract non-zero HR readings and interpolate to every frame.
            hr_sparse = df_gt['HR'].values.astype(np.float64)
            src_idx = np.linspace(0, 1, len(hr_sparse))
            tgt_idx = np.linspace(0, 1, NUM_FRAMES)
            hr_sparse_aligned = interp1d(src_idx, hr_sparse, kind='linear')(tgt_idx)

            # Replace zeros with NaN then interpolate
            hr_sparse_aligned[hr_sparse_aligned == 0] = np.nan
            hr_series = pd.Series(hr_sparse_aligned).interpolate(
                method='linear').ffill().bfill()
            sig_bpm_direct = hr_series.values

            # ── Option B: re-estimate BPM from Signal_Value via Welch ─────────
            sig_bpm_welch = _estimate_bpm(sig_bvp, FPS)

            # ── Save outputs ──────────────────────────────────────────────────
            stem = f'attendant{num_attendant}_dist{dist}'

            # BVP (normalised, frame-aligned)
            pd.Series(sig_bvp).to_csv(
                os.path.join(dir_out, f'{stem}_bvp.csv'),
                index=False, header=['Signal_Value']
            )

            # BPM — direct from HR column (preferred if sensor outputs HR)
            pd.Series(sig_bpm_direct).to_csv(
                os.path.join(dir_out, f'{stem}_bpm_direct.csv'),
                index=False, header=['BPM']
            )

            # BPM — Welch re-estimation from raw signal
            pd.Series(sig_bpm_welch).to_csv(
                os.path.join(dir_out, f'{stem}_bpm_welch.csv'),
                index=False, header=['BPM']
            )

            print(f"  Saved GT for att{num_attendant} dist{dist} → {dir_out}")


if __name__ == "__main__":
    # Generate ground truth HR for UBFC-Phys.
    dir_crt = os.getcwd()
    dir_dataset = os.path.join(dir_crt,'data','custom')
    main_gen_gtHR(dir_dataset=dir_dataset)