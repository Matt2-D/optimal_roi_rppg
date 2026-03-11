"""
Generate ground truth BPM data using existing BVP data for UBFC-Phys dataset.
"""

# Original Author: Shuo Li
# Date: 2023/08/18
# Editor: Matthew Dowell
# Date: 03/10/2026

import warnings
warnings.filterwarnings("ignore")

import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.interpolate import interp1d

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


def load_gt_csv(csv_path: str) -> pd.DataFrame:
    # Load and validate the ground truth CSV
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Ground truth CSV not found: {csv_path}")
    df = pd.read_csv(csv_path, index_col=None)
    required = {"Signal_Value", "HR", "Package_Num"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"Ground truth CSV missing columns: {missing}")
    return df


def align_to_frames(signal: np.ndarray, target_len: int) -> np.ndarray:
    #Resample signal to target_len frames
    src_idx = np.linspace(0, 1, len(signal))
    tgt_idx = np.linspace(0, 1, target_len)
    return interp1d(src_idx, signal, kind='linear')(tgt_idx)

def main_gen_gtHR(dir_dataset: str) -> None:
    """
    Generate ground truth BPM and BVP files for all attendants/distances.

    Parameters
    ----------
    dir_dataset : Root dataset directory
                  (e.g. data/custom  — attendant folders live inside)
    """

    list_attendant = [1]
    distances      = [1, 2, 3]

    # Output directory
    dir_out = os.path.join(os.getcwd(), 'data', 'custom', 'gtHR')
    os.makedirs(dir_out, exist_ok=True)

    for num_attendant in tqdm(list_attendant, desc="Attendants"):
        for dist in tqdm(distances, desc=f"  Distances (att{num_attendant})", leave=False):

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
            df_gt = load_gt_csv(gt_csv)

            #extract BVP
            sig_bvp_raw = df_gt['Signal_Value'].values.astype(np.float64)

            # Normalise to zero-mean (removes DC offset ~500)
            sig_bvp_raw = sig_bvp_raw - np.mean(sig_bvp_raw)

            # Resample from 2987 → 3000 frames
            sig_bvp = align_to_frames(sig_bvp_raw, NUM_FRAMES)

            # Remove zeros BEFORE resampling to avoid smearing zeros into signal.
            hr_sparse = df_gt['HR'].values.astype(np.float64)
            hr_sparse[hr_sparse == 0] = np.nan
            hr_filled = pd.Series(hr_sparse).interpolate(
                method='linear').ffill().bfill().values
            sig_bpm_direct = align_to_frames(hr_filled, NUM_FRAMES)

            # save outputs
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
            print(f"  Saved GT for att{num_attendant} dist{dist} → {dir_out}")


if __name__ == "__main__":
    dir_crt     = os.getcwd()
    dir_dataset = os.path.join(dir_crt, 'data', 'custom')
    main_gen_gtHR(dir_dataset=dir_dataset)