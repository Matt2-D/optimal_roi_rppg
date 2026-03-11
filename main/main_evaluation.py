"""
Performance evaluation of different rPPG methods.
"""

# Original Author: Shuo Li
# Date: 2023/08/05
# Editor: Matthew Dowell
# Date: 03/10/2026

import os
import sys
import numpy as np
import pandas as pd
dir_crt = os.getcwd()
sys.path.append(os.path.join(dir_crt, 'util'))
from util import util_analysis


def load_gt(dir_crt: str, num_attendant: int, dist: int, num_frames: int):
    # Load ground truth BVP and BPM for a given attendant/distance.
    #Returns gtBVP : np.ndarray  shape [num_frames]
    #gtBPM : np.ndarray  shape [num_frames]
    gt_dir  = os.path.join(dir_crt, 'data', 'custom', 'gtHR')
    stem    = f'attendant{num_attendant}_dist{dist}'

    bvp_path = os.path.join(gt_dir, f'{stem}_bvp.csv')
    bpm_path = os.path.join(gt_dir, f'{stem}_bpm_direct.csv')

    for p in [bvp_path, bpm_path]:
        if not os.path.isfile(p):
            raise FileNotFoundError(
                f"Ground truth file not found: {p}\n"
                "Run main_gen_gtHR.py first."
            )

    gtBVP = pd.read_csv(bvp_path).iloc[:, 0].values.astype(np.float64)
    gtBPM = pd.read_csv(bpm_path).iloc[:, 0].values.astype(np.float64)

    # Safety: trim or pad to num_frames
    def _fit(arr):
        if len(arr) >= num_frames:
            return arr[:num_frames]
        return np.pad(arr, (0, num_frames - len(arr)), mode='edge')

    return _fit(gtBVP), _fit(gtBPM)


def main_eval(name_dataset='custom', algorithm='CHROM'):
    """Evaluation pipeline for a given dataset using a given algorithm.
    Parameters
    ----------
    name_dataset: Name of the selected rPPG dataset. 
                  ['UBFC-rPPG', 'UBFC-Phys', 'LGI-PPGI', 'BUAA-MIHR'].
    algorithm: Selected rPPG algorithm. ['CHROM', 'LGI', 'OMIT', 'POS'].

    Returns
    -------

    """
    # Get current directory.
    dir_crt = os.getcwd()
    dir_option = os.path.join(dir_crt, 'config', 'options.yaml')
    # Parameter class initialization.
    Params = util_analysis.Params(dir_option=dir_option, name_dataset=name_dataset)
    # Groundtruth class initialization.
    GT = util_analysis.GroundTruth(dir_dataset=Params.dir_dataset, name_dataset=name_dataset)
    # Structures for different datasets.
    if name_dataset == "custom": #custom dataset
        list_attendant = [1]
        distances = [1, 2, 3, 4, 5]

        df_eval = pd.DataFrame(
            columns=['attendant', 'distance', 'ROI',
                     'DTW', 'PCC', 'CCC', 'RMSE', 'MAE', 'MAPE']
        )

        for num_attendant in list_attendant:
            for dist in distances:
                print(f"\n{'=' * 60}")
                print(f"[Eval] {name_dataset} | {algorithm} | "
                      f"attendant{num_attendant} | dist{dist}")
                print(f"{'=' * 60}")

                # Load HR CSV
                dir_hr = os.path.join(
                    dir_crt, 'data', name_dataset, 'hr',
                    f'{dist}_{algorithm}1.csv'
                )
                if not os.path.isfile(dir_hr):
                    print(f"  [WARN] HR CSV not found, skipping: {dir_hr}")
                    continue

                # index_col=False preserves all columns including 'frame'
                df_hr = pd.read_csv(dir_hr, index_col=False)

                # ── Derive ROI list from CSV (picks up ROIs 29 & 30) ─────────
                roi_names = list(df_hr['ROI'].unique())
                num_rois = len(roi_names)
                num_frames = df_hr['frame'].nunique()

                print(f"  ROIs found: {num_rois}  |  Frames: {num_frames}")

                # Load ground truth
                gtBVP, gtBPM = load_gt(dir_crt, num_attendant, dist, num_frames)

                #Build BVP / BPM arrays [frames x ROIs]
                sig_bvp = np.zeros([num_frames, num_rois])
                sig_bpm = np.zeros([num_frames, num_rois])

                for i_roi, roi_name in enumerate(roi_names):
                    mask = df_hr['ROI'].values == roi_name
                    bvp_vals = df_hr.loc[mask, 'BVP'].values
                    bpm_vals = df_hr.loc[mask, 'BPM'].values

                    # Guard against length mismatch
                    n = min(num_frames, len(bvp_vals))
                    sig_bvp[:n, i_roi] = bvp_vals[:n]
                    sig_bpm[:n, i_roi] = bpm_vals[:n]

                # Metrics per ROI
                # Pass roi_names so eval_pipe can label rows correctly
                df_metric = util_analysis.eval_pipe(
                    sig_bvp, sig_bpm,
                    gtBVP, gtBPM,
                    Params,
                    roi_names=roi_names
                )

                df_metric['attendant'] = num_attendant
                df_metric['distance'] = dist
                df_eval = pd.concat([df_eval, df_metric], ignore_index=True)

        # Save results
        result_dir = os.path.join(dir_crt, 'result', name_dataset)
        os.makedirs(result_dir, exist_ok=True)
        out_path = os.path.join(result_dir, f'evaluation_{algorithm}1.csv')
        df_eval.to_csv(out_path, index=False)
        print(f"\n[Eval] Results saved → {out_path}")
        print(df_eval.to_string())

    elif name_dataset == '!UBFC-rPPG':  # UBFC-rPPG dataset.
        list_attendant = [1] + list(range(3, 6)) + list(range(8, 19)) + [20] + \
                         list(range(22, 27)) + list(range(30, 50))  # Attendant sequence num.
        # Dataframe initialization.
        df_eval = pd.DataFrame(columns=['attendant', 'ROI', 'DTW', 'PCC', 'CCC', 'RMSE', 'MAE', 'MAPE'])
        # Loop over all data points.
        for num_attendant in list_attendant:
            print([name_dataset, algorithm, num_attendant])
            # Load BVP and HR signals.
            dir_hr = os.path.join(dir_crt, 'data', name_dataset, 'hr', str(num_attendant)+'_'+algorithm+'1.csv')
            df_hr = pd.read_csv(dir_hr, index_col=0)
            # Load ground truth.
            gtTime, gtTrace, gtHR = GT.get_GT(specification=['realistic', num_attendant], 
                                              num_frame_interp=int(len(df_hr)/len(Params.list_roi_name)), 
                                              slice=[0, 1])
            # Initialization of BVP and BPM arrays.
            sig_bvp = np.zeros(shape=[int(len(df_hr)/len(Params.list_roi_name)), len(Params.list_roi_name)])
            sig_bpm = np.zeros(shape=[int(len(df_hr)/len(Params.list_roi_name)), len(Params.list_roi_name)])
            for i_roi in range(len(Params.list_roi_name)):
                sig_bvp[:, i_roi] = df_hr.loc[df_hr['ROI'].values == Params.list_roi_name[i_roi], 'BVP']
                sig_bpm[:, i_roi] = df_hr.loc[df_hr['ROI'].values == Params.list_roi_name[i_roi], 'BPM']
            # Metrics calculation.
            df_metric = util_analysis.eval_pipe(sig_bvp, sig_bpm, gtTime, gtTrace, gtHR, Params)
            df_eval = pd.concat([df_eval, df_metric])
            df_eval.reset_index(drop=True, inplace=True)
            df_eval.loc[len(df_eval)-len(Params.list_roi_name):, 'attendant'] = num_attendant
            # Dataframe saving.
            df_eval.to_csv(os.path.join(dir_crt, 'result', name_dataset, 'evaluation_'+algorithm+'1.csv'))


    elif name_dataset == '!UBFC-Phys':   # UBFC-Phys dataset.
        # Name of attendants.
        list_attendant = list(range(1, 57))
        # Condition types.
        list_condition = [1]  # [1, 2, 3]
        # Dataframe initialization.
        df_eval = pd.DataFrame(columns=['attendant', 'condition', 'ROI', 'DTW', 'PCC', 'CCC', 'RMSE', 'MAE', 'MAPE'])
        # Loop over all data points.
        for num_attendant in list_attendant:
            for condition in list_condition:
                print([name_dataset, algorithm, num_attendant, condition])
                # Load BVP and HR signals.
                dir_hr = os.path.join(dir_crt, 'data', name_dataset, 'hr', str(num_attendant)+'_'+str(condition)+'_'+algorithm+'1.csv')
                df_hr = pd.read_csv(dir_hr, index_col=0)
                # Load groundtruth.
                gtTime, gtTrace, gtHR = GT.get_GT(specification=[num_attendant, condition], 
                                                  num_frame_interp=int(len(df_hr)/len(Params.list_roi_name)), 
                                                  slice=[0, 1])
                # Initialization of BVP and BPM arrays.
                sig_bvp = np.zeros(shape=[int(len(df_hr)/len(Params.list_roi_name)), len(Params.list_roi_name)])
                sig_bpm = np.zeros(shape=[int(len(df_hr)/len(Params.list_roi_name)), len(Params.list_roi_name)])
                for i_roi in range(len(Params.list_roi_name)):
                    sig_bvp[:, i_roi] = df_hr.loc[df_hr['ROI'].values == Params.list_roi_name[i_roi], 'BVP']
                    sig_bpm[:, i_roi] = df_hr.loc[df_hr['ROI'].values == Params.list_roi_name[i_roi], 'BPM']
                # Metrics calculation.
                df_metric = util_analysis.eval_pipe(sig_bvp, sig_bpm, gtTime, gtTrace, gtHR, Params)
                df_eval = pd.concat([df_eval, df_metric])
                df_eval.reset_index(drop=True, inplace=True)
                df_eval.loc[len(df_eval)-len(Params.list_roi_name):, 'attendant'] = num_attendant
                df_eval.loc[len(df_eval)-len(Params.list_roi_name):, 'condition'] = condition
                # Dataframe saving.
                df_eval.to_csv(os.path.join(dir_crt, 'result', name_dataset, 'evaluation_'+algorithm+'1.csv'))


    elif name_dataset == '!LGI-PPGI':   # LGI-PPGI dataset.
        list_attendant = ['angelo', 'david', 'alex', 'cpi', 'felix', 'harun']  # Attendant name.
        list_motion = ['resting', 'gym', 'rotation', 'talk']  # Motion type.
        # Dataframe initialization.
        df_eval = pd.DataFrame(columns=['attendant', 'motion', 'ROI', 'DTW', 'PCC', 'CCC', 'RMSE', 'MAE', 'MAPE'])
        # Loop over all data points.
        for name_attendant in list_attendant:
            for motion in list_motion:
                print([name_dataset, algorithm, name_attendant, motion])
                # Load BVP and HR signals.
                dir_hr = os.path.join(dir_crt, 'data', name_dataset, 'hr', name_attendant+'_'+motion+'_'+algorithm+'1.csv')
                df_hr = pd.read_csv(dir_hr, index_col=0)
                # Load groundtruth.
                gtTime, gtTrace, gtHR = GT.get_GT(specification=[name_attendant, motion], 
                                                  num_frame_interp=int(len(df_hr)/len(Params.list_roi_name)), 
                                                  slice=[0, 1])
                # Initialization of BVP and BPM arrays.
                sig_bvp = np.zeros(shape=[int(len(df_hr)/len(Params.list_roi_name)), len(Params.list_roi_name)])
                sig_bpm = np.zeros(shape=[int(len(df_hr)/len(Params.list_roi_name)), len(Params.list_roi_name)])
                for i_roi in range(len(Params.list_roi_name)):
                    sig_bvp[:, i_roi] = df_hr.loc[df_hr['ROI'].values == Params.list_roi_name[i_roi], 'BVP']
                    sig_bpm[:, i_roi] = df_hr.loc[df_hr['ROI'].values == Params.list_roi_name[i_roi], 'BPM']
                # Metrics calculation.
                df_metric = util_analysis.eval_pipe(sig_bvp, sig_bpm, gtTime, gtTrace, gtHR, Params)
                df_eval = pd.concat([df_eval, df_metric])
                df_eval.reset_index(drop=True, inplace=True)
                df_eval.loc[len(df_eval)-len(Params.list_roi_name):, 'attendant'] = name_attendant
                df_eval.loc[len(df_eval)-len(Params.list_roi_name):, 'motion'] = motion
                # Dataframe saving.
                df_eval.to_csv(os.path.join(dir_crt, 'result', name_dataset, 'evaluation_'+algorithm+'1.csv'))
        

    elif name_dataset == '!BUAA-MIHR':   # BUAA-MIHR dataset.
        # Sequnce num of attendants.
        list_attendant = list(range(4, 5))
        # Illumination levels.
        list_lux = ['lux 10.0', 'lux 15.8', 'lux 25.1', 'lux 39.8', 'lux 63.1', 'lux 100.0']
        # Attendant names.
        list_name = ['APH', 'GDB', 'HB', 'KLK', 'LW', 'LXR', 'LY', 'LZY', 'LMX', 'MXZ', 'PYC', 'QSH', 'WQT']
        # Dataframe initialization.
        df_eval = pd.DataFrame(columns=['attendant', 'lux', 'ROI', 'DTW', 'PCC', 'CCC', 'RMSE', 'MAE', 'MAPE'])
        # Loop over all data points.
        for num_attendant in list_attendant:
            for lux in list_lux:
                print([name_dataset, algorithm, num_attendant, lux])
                # Load BVP and HR signals.
                dir_hr = os.path.join(dir_crt, 'data', name_dataset, 'hr', str(num_attendant).zfill(2) + \
                                      '_' + str(lux).replace(' ', '') + '_' + algorithm+'1.csv')
                df_hr = pd.read_csv(dir_hr, index_col=None)
                # Load groundtruth.
                name = list_name[num_attendant-1]
                gtTime, gtTrace, gtHR = GT.get_GT(specification=[num_attendant, lux, name], 
                                                  num_frame_interp=int(len(df_hr)/len(Params.list_roi_name)), 
                                                  slice=[0, 1])
                # Initialization of BVP and BPM arrays.
                sig_bvp = np.zeros(shape=[int(len(df_hr)/len(Params.list_roi_name)), len(Params.list_roi_name)])
                sig_bpm = np.zeros(shape=[int(len(df_hr)/len(Params.list_roi_name)), len(Params.list_roi_name)])
                for i_roi in range(len(Params.list_roi_name)):
                    sig_bvp[:, i_roi] = df_hr.loc[df_hr['ROI'].values == Params.list_roi_name[i_roi], 'BVP']
                    sig_bpm[:, i_roi] = df_hr.loc[df_hr['ROI'].values == Params.list_roi_name[i_roi], 'BPM']
                # Metrics calculation.
                df_metric = util_analysis.eval_pipe(sig_bvp, sig_bpm, gtTime, gtTrace, gtHR, Params)
                df_eval = pd.concat([df_eval, df_metric])
                df_eval.reset_index(drop=True, inplace=True)
                # Assign data point information.
                df_eval.loc[len(df_eval)-len(Params.list_roi_name):, 'attendant'] = num_attendant
                df_eval.loc[len(df_eval)-len(Params.list_roi_name):, 'lux'] = lux.replace('lux ', '')
                # Dataframe saving.
                df_eval.to_csv(os.path.join(dir_crt, 'result', name_dataset, 'evaluation_'+algorithm+'1.csv'))


if __name__ == "__main__":
    list_dataset = ['custom']  # ['UBFC-rPPG', 'UBFC-Phys', 'LGI-PPGI', 'BUAA-MIHR'].
    list_algorithm = ['CHROM', 'LGI', 'OMIT', 'POS']  # ['CHROM', 'LGI', 'OMIT', 'POS'].
    # Loop over all selected rPPG datasets.
    for name_dataset in list_dataset:
        # Loop over all selected rPPG algorithms.
        for algorithm in list_algorithm:
            print([name_dataset, algorithm])
            main_eval(name_dataset=name_dataset, algorithm=algorithm)