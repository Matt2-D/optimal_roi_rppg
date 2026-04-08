"""
Transform raw RGB traces to BVP and HR signals.
"""

# Original Author: Shuo Li
# Date: 2023/07/18
# Editor: Matthew Dowell
# Date: 03/10/2026

import warnings
warnings.filterwarnings("ignore")  # Ignore unnecessary warnings.
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Turn off oneDNN custom operations.
import sys
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
dir_crt = os.getcwd()
sys.path.append(os.path.join(dir_crt, 'util'))
from main.util import util_analysis


def main_rgb2hr(name_dataset, algorithm):
    """Main function for transforming RGB traces to HR-related signals.
    Parameters
    ----------
    name_dataset: Name of the selected dataset.
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

    # RGB signal -> bvp signal.
    if name_dataset == 'custom':
        # Sequence num of attendants.
        list_attendant = [1,2]
        distances = [1, 2]  # , 3, 4, 5]
        # Loop over all attendants.
        for attendant in list_attendant:
            for dist in distances:
                dir_sig_rgb = os.path.join(dir_crt, 'data', name_dataset, 'rgb', f'{attendant}_{dist}.csv')

                if not os.path.isfile(dir_sig_rgb):
                    print(f"  [SKIP] RGB CSV not found for attendant = {attendant}, dist={dist}: {dir_sig_rgb}")
                    continue

                df_rgb = pd.read_csv(dir_sig_rgb)

                # Diagnostic
                print(f"dist={dist} | shape={df_rgb.shape} | frame dtype={df_rgb['frame'].dtype} | "
                      f"frame NaNs={df_rgb['frame'].isna().sum()} | ROIs={df_rgb['ROI'].nunique()}")

                df_rgb['frame'] = pd.to_numeric(df_rgb['frame'], errors='coerce')
                df_rgb = df_rgb.dropna(subset=['frame'])

                if df_rgb.empty:
                    print(f"  [SKIP] No valid frames after cleaning for dist={dist}")
                    continue

                df_rgb['frame'] = df_rgb['frame'].astype(int)
                num_frames = int(df_rgb['frame'].max())
                # ... rest of loop unchanged
                #ROI list from CSV
                roi_names = list(df_rgb['ROI'].unique())
                num_rois = len(roi_names)
                # Clean the frame column
                df_rgb['frame'] = pd.to_numeric(df_rgb['frame'], errors='coerce')

                # Drop rows where frame is NaN
                df_rgb = df_rgb.dropna(subset=['frame'])

                # Convert to int
                df_rgb['frame'] = df_rgb['frame'].astype(int)

                # Now safe to compute max
                df_rgb['frame'] = pd.to_numeric(df_rgb['frame'], errors='coerce')
                df_rgb = df_rgb.dropna(subset=['frame'])
                df_rgb['frame'] = df_rgb['frame'].astype(int)
                #num_frames = int(df_rgb['frame'].max())  # must be after cleaning, and cast to int
                # RGB signal initialization.
                sig_rgb = np.zeros([num_frames,num_rois, 3])
                # Loop over all ROIs.
                for i_roi, roi_name in enumerate(roi_names):
                    mask = df_rgb['ROI'].values == roi_name
                    sig_rgb[:, i_roi, 0] = df_rgb.loc[mask, 'R'].values  # Red channel.
                    sig_rgb[:, i_roi, 1] = df_rgb.loc[mask, 'G'].values  # Green channel.
                    sig_rgb[:, i_roi, 2] = df_rgb.loc[mask, 'B'].values  # Blue channel.

                # RGB video information.

                # Get video fps.
                Params.fps = 50   # or whatever your dataset uses
                # RGB signal -> bvp signal & bpm signal.
                # Update unpack
                sig_bvp, sig_bpm, sig_snr, sig_fc, sig_ac, sig_cra, sig_sqi = \
                    util_analysis.rppg_hr_pipe(sig_rgb=sig_rgb, method=algorithm, Params=Params)

                # Update DataFrame columns
                df_hr = pd.DataFrame(
                    columns=['frame', 'time', 'ROI', 'BVP', 'BPM', 'SNR', 'FC_SQI', 'AC_SQI', 'CRA_SQI', 'SQI'],
                    index=list(range(len(df_rgb)))
                )
                df_hr.loc[:, ['frame', 'time', 'ROI']] = df_rgb.loc[:, ['frame', 'time', 'ROI']].values

                for i_roi, roi_name in enumerate(roi_names):
                    mask = df_hr['ROI'].values == roi_name
                    df_hr.loc[mask, 'BVP'] = sig_bvp[:, i_roi]
                    df_hr.loc[mask, 'BPM'] = sig_bpm[:, i_roi]
                    df_hr.loc[mask, 'SNR'] = sig_snr[:, i_roi]
                    df_hr.loc[mask, 'FC_SQI'] = sig_fc[:, i_roi]
                    df_hr.loc[mask, 'AC_SQI'] = sig_ac[:, i_roi]
                    df_hr.loc[mask, 'CRA_SQI'] = sig_cra[:, i_roi]
                    df_hr.loc[mask, 'SQI'] = sig_sqi[:, i_roi]

                # Data saving.
                dir_save_data = os.path.join(dir_crt, 'data', name_dataset, 'hr',
                                             str(attendant) + '_' + str(dist) + algorithm + '1.csv')
                df_hr.to_csv(dir_save_data, index=False)

    if name_dataset == '!UBFC-rPPG':
        # Sequence num of attendants.
        list_attendant = [1] + list(range(3, 6)) + list(range(8, 19)) + [20] + list(range(22, 27)) + list(range(30, 50))
        # Loop over all attendants.
        for num_attendant in tqdm(list_attendant):
            # Parse the RGB signal from the RGB dataframe. Size = [num_frames, num_ROI, rgb_channels(3)].
            dir_sig_rgb = os.path.join(dir_crt, 'data', name_dataset, 'rgb', str(num_attendant)+'1.csv')
            df_rgb = pd.read_csv(dir_sig_rgb)
            # RGB signal initialization.
            sig_rgb = np.zeros([df_rgb['frame'].max(), len(np.unique(df_rgb['ROI'].values)), 3])
            # Loop over all ROIs.
            for i_roi in range(len(Params.list_roi_name)):
                sig_rgb[:, i_roi, 0] = df_rgb.loc[df_rgb['ROI'].values == Params.list_roi_name[i_roi], 'R']  # Red channel.
                sig_rgb[:, i_roi, 1] = df_rgb.loc[df_rgb['ROI'].values == Params.list_roi_name[i_roi], 'G']  # Green channel.
                sig_rgb[:, i_roi, 2] = df_rgb.loc[df_rgb['ROI'].values == Params.list_roi_name[i_roi], 'B']  # Blue channel.
            # RGB video information.
            dir_vid = os.path.join(Params.dir_dataset, 'DATASET_2', 'subject'+str(num_attendant), 'vid.avi')
            # Get video fps.
            capture = cv2.VideoCapture(dir_vid)
            Params.fps = capture.get(cv2.CAP_PROP_FPS)
            # RGB signal -> bvp signal & bpm signal.
            sig_bvp, sig_bpm = util_analysis.rppg_hr_pipe(sig_rgb=sig_rgb, method=algorithm, Params=Params)
            # Create the dataframe to save the HR-related data (bvp signal & bpm signal).
            df_hr = pd.DataFrame(columns=['frame', 'time', 'ROI', 'BVP', 'BPM'], index=list(range(len(df_rgb))))
            df_hr.loc[:, ['frame', 'time', 'ROI']] = df_rgb.loc[:, ['frame', 'time', 'ROI']]
            # Loop over all ROIs.
            for i_roi in range(len(Params.list_roi_name)):
                df_hr.loc[df_hr['ROI'].values == Params.list_roi_name[i_roi], 'BVP'] = sig_bvp[:, i_roi]  # BVP signal.
                df_hr.loc[df_hr['ROI'].values == Params.list_roi_name[i_roi], 'BPM'] = sig_bpm[:, i_roi]  # BPM signal.
            # Data saving.
            dir_save_data = os.path.join(dir_crt, 'data', name_dataset, 'hr', str(num_attendant)+'_'+algorithm+'1.csv')
            df_hr.to_csv(dir_save_data, index=False)

    elif name_dataset == '!UBFC-Phys':
        # List of attendants.
        list_attendant = list(range(1, 57))
        # Condition types.
        list_condition = [1]   # [1, 2, 3]
        for num_attendant in tqdm(list_attendant):
            for condition in list_condition:
                # Parse the RGB signal from the RGB dataframe. Size = [num_frames, num_ROI, rgb_channels(3)].
                dir_sig_rgb = os.path.join(dir_crt, 'data', name_dataset, 'rgb', str(num_attendant) + '_' + str(condition) + '1.csv')
                df_rgb = pd.read_csv(dir_sig_rgb)
                # RGB signal initialization.
                sig_rgb = np.zeros([df_rgb['frame'].max(), len(np.unique(df_rgb['ROI'].values)), 3])
                # Loop over all ROIs.
                for i_roi in range(len(Params.list_roi_name)):
                    sig_rgb[:, i_roi, 0] = df_rgb.loc[df_rgb['ROI'].values == Params.list_roi_name[i_roi], 'R']  # Red channel.
                    sig_rgb[:, i_roi, 1] = df_rgb.loc[df_rgb['ROI'].values == Params.list_roi_name[i_roi], 'G']  # Green channel.
                    sig_rgb[:, i_roi, 2] = df_rgb.loc[df_rgb['ROI'].values == Params.list_roi_name[i_roi], 'B']  # Blue channel.
                # RGB video information.
                dir_vid = os.path.join(Params.dir_dataset, 's'+str(num_attendant), 'vid_s'+str(num_attendant)+'_T'+str(condition)+'.avi')
                # Get video fps.
                capture = cv2.VideoCapture(dir_vid)
                Params.fps = capture.get(cv2.CAP_PROP_FPS)
                # RGB signal -> bvp signal & bpm signal.
                sig_bvp, sig_bpm = util_analysis.rppg_hr_pipe(sig_rgb=sig_rgb, method=algorithm, Params=Params)
                # Create the dataframe to save the HR-related data (bvp signal & bpm signal).
                df_hr = pd.DataFrame(columns=['frame', 'time', 'ROI', 'BVP', 'BPM'], index=list(range(len(df_rgb))))
                df_hr.loc[:, ['frame', 'time', 'ROI']] = df_rgb.loc[:, ['frame', 'time', 'ROI']]
                # Loop over all ROIs.
                for i_roi in range(len(Params.list_roi_name)):
                    df_hr.loc[df_hr['ROI'].values == Params.list_roi_name[i_roi], 'BVP'] = sig_bvp[:, i_roi]  # BVP signal.
                    df_hr.loc[df_hr['ROI'].values == Params.list_roi_name[i_roi], 'BPM'] = sig_bpm[:, i_roi]  # BPM signal.
                # Data saving.
                dir_save_data = os.path.join(dir_crt, 'data', name_dataset, 'hr', str(num_attendant) + '_' + \
                                             str(condition) + '_' + algorithm + '1.csv')
                df_hr.to_csv(dir_save_data, index=False)

    elif name_dataset == '!LGI-PPGI':
        # Name of attendants.
        list_attendant = ['alex', 'angelo', 'cpi', 'david', 'felix', 'harun']
        # Motion types.
        list_motion = ['gym', 'resting', 'talk', 'rotation']
        for attendant in tqdm(list_attendant):
            for motion in list_motion:
                # Parse the RGB signal from the RGB dataframe. Size = [num_frames, num_ROI, rgb_channels(3)].
                dir_sig_rgb = os.path.join(dir_crt, 'data', name_dataset, 'rgb', attendant+'_'+motion+'1.csv')
                df_rgb = pd.read_csv(dir_sig_rgb)
                # RGB signal initialization.
                sig_rgb = np.zeros([df_rgb['frame'].max(), len(np.unique(df_rgb['ROI'].values)), 3])
                # Loop over all ROIs.
                for i_roi in range(len(Params.list_roi_name)):
                    sig_rgb[:, i_roi, 0] = df_rgb.loc[df_rgb['ROI'].values == Params.list_roi_name[i_roi], 'R']   # Red channel.
                    sig_rgb[:, i_roi, 1] = df_rgb.loc[df_rgb['ROI'].values == Params.list_roi_name[i_roi], 'G']   # Green channel.
                    sig_rgb[:, i_roi, 2] = df_rgb.loc[df_rgb['ROI'].values == Params.list_roi_name[i_roi], 'B']   # Blue channel.
                # RGB video information.
                dir_vid = os.path.join(Params.dir_dataset, attendant, attendant+'_'+motion, 'cv_camera_sensor_stream_handler.avi')
                # Get video fps.
                capture = cv2.VideoCapture(dir_vid)
                Params.fps = capture.get(cv2.CAP_PROP_FPS)
                # RGB signal -> bvp signal & bpm signal.
                sig_bvp, sig_bpm = util_analysis.rppg_hr_pipe(sig_rgb=sig_rgb, method=algorithm, Params=Params)
                # Create the dataframe to save the HR-related data (bvp signal & bpm signal).
                df_hr = pd.DataFrame(columns=['frame', 'time', 'ROI', 'BVP', 'BPM'], index=list(range(len(df_rgb))))
                df_hr.loc[:, ['frame', 'time', 'ROI']] = df_rgb.loc[:, ['frame', 'time', 'ROI']]
                # Loop over all ROIs.
                for i_roi in range(len(Params.list_roi_name)):
                    df_hr.loc[df_hr['ROI'].values == Params.list_roi_name[i_roi], 'BVP'] = sig_bvp[:, i_roi]  # BVP signal.
                    df_hr.loc[df_hr['ROI'].values == Params.list_roi_name[i_roi], 'BPM'] = sig_bpm[:, i_roi]  # BPM signal.
                # Data saving.
                dir_save_data = os.path.join(dir_crt, 'data', name_dataset, 'hr', attendant+'_'+motion+'_'+algorithm+'1.csv')
                df_hr.to_csv(dir_save_data, index=False)

    elif name_dataset == '!BUAA-MIHR':
        # Sequnce num of attendants.
        list_attendant = list(range(1, 14))
        # Lux levels.
        list_lux = ['lux 10.0', 'lux 15.8', 'lux 25.1', 'lux 39.8', 'lux 63.1', 'lux 100.0']
        # Attendant names.
        list_name = ['APH', 'GDB', 'HB', 'KLK', 'LW', 'LXR', 'LY', 'LZY', 'LMX', 'MXZ', 'PYC', 'QSH', 'WQT']
        # Loop over all attendants.
        for num_attendant in tqdm(list_attendant):
            # Loop over all illumination levels.
            for lux in list_lux:
                # Parse the RGB signal from the RGB dataframe. Size = [num_frames, num_ROI, rgb_channels(3)].
                dir_sig_rgb = os.path.join(dir_crt, 'data', name_dataset, 'rgb', str(num_attendant).zfill(2)+'_'+lux.replace(' ', '')+'1.csv')
                df_rgb = pd.read_csv(dir_sig_rgb, index_col=None)
                # RGB signal initialization.
                sig_rgb = np.zeros([df_rgb['frame'].max(), len(np.unique(df_rgb['ROI'].values)), 3])
                # Loop over all ROIs.
                for i_roi in range(len(Params.list_roi_name)):
                    sig_rgb[:, i_roi, 0] = df_rgb.loc[df_rgb['ROI'].values == Params.list_roi_name[i_roi], 'R']  # Red channel.
                    sig_rgb[:, i_roi, 1] = df_rgb.loc[df_rgb['ROI'].values == Params.list_roi_name[i_roi], 'G']  # Green channel.
                    sig_rgb[:, i_roi, 2] = df_rgb.loc[df_rgb['ROI'].values == Params.list_roi_name[i_roi], 'B']  # Blue channel.
                # RGB video information.
                dir_vid = os.path.join(Params.dir_dataset, 'Sub '+str(num_attendant).zfill(2), lux, \
                                       lux.replace(' ', '') + '_' + list_name[num_attendant-1]+'.avi')
                # Get video fps.
                capture = cv2.VideoCapture(dir_vid)
                Params.fps = capture.get(cv2.CAP_PROP_FPS)
                # RGB signal -> bvp signal & bpm signal.
                sig_bvp, sig_bpm = util_analysis.rppg_hr_pipe(sig_rgb=sig_rgb, method=algorithm, Params=Params)
                # Create the dataframe to save the HR-related data (bvp signal & bpm signal).
                df_hr = pd.DataFrame(columns=['frame', 'time', 'ROI', 'BVP', 'BPM'], index=list(range(len(df_rgb))))
                df_hr.loc[:, ['frame', 'time', 'ROI']] = df_rgb.loc[:, ['frame', 'time', 'ROI']]
                # Loop over all ROIs.
                for i_roi in range(len(Params.list_roi_name)):
                    df_hr.loc[df_hr['ROI'].values == Params.list_roi_name[i_roi], 'BVP'] = sig_bvp[:, i_roi]  # BVP signal.
                    df_hr.loc[df_hr['ROI'].values == Params.list_roi_name[i_roi], 'BPM'] = sig_bpm[:, i_roi]  # BPM signal.
                # Data saving.
                dir_save_data = os.path.join(dir_crt, 'data', name_dataset, 'hr', str(num_attendant).zfill(2) + \
                                             '_' + str(lux).replace(' ', '') + '_' + algorithm+'1.csv')
                df_hr.to_csv(dir_save_data, index=False)


if __name__ == "__main__":
    # Available datasets.
    list_dataset = ['custom','UBFC-rPPG', 'UBFC-Phys', 'LGI-PPGI', 'BUAA-MIHR']   # ['UBFC-rPPG', 'UBFC-Phys', 'LGI-PPGI', 'BUAA-MIHR'].
    # Selected rPPG algorithms.
    list_algorithm = ['LGI', 'OMIT', 'CHROM', 'POS']   # ['LGI', 'OMIT', 'CHROM', 'POS'].
    for name_dataset in list_dataset:
        for algorithm in list_algorithm:
            print([name_dataset, algorithm])
            main_rgb2hr(name_dataset=name_dataset, algorithm=algorithm)