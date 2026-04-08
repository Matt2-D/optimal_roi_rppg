"""
Extract raw RGB traces from facial videos.
"""

# Original Author: Shuo Li
# Date: 2023/05/05
# Editor: Matthew Dowell
# Date: 03/10/2026

import warnings
warnings.filterwarnings("ignore")  # Ignore unnecessary warnings.
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Turn off oneDNN custom operations.
import sys
import pandas as pd
from tqdm import tqdm
from main.CropSense import main_cropsense
import cv2
dir_crt = os.getcwd()
sys.path.append(os.path.join(dir_crt, 'util'))
from main.util import util_pyVHR, util_analysis
from main.CropSense.main_cropsense import run_cropsense


def main_vid2rgb(name_dataset):
    """Main function for face detection of videos.
    Parameters
    ----------
    name_dataset: Name of the selected dataset.
                  [UBFC-rPPG, UBFC-Phys, LGI-PPGI, BUAA-MIHR].
    
    Returns
    -------

    """
    # Get current directory.
    dir_crt = os.getcwd()
    dir_option = os.path.join(dir_crt, 'config', 'options.yaml')
    # Parameter class initialization.
    Params = util_analysis.Params(dir_option=dir_option, name_dataset=name_dataset)

    if name_dataset == 'custom':
        list_attendant = [1,2]
        distances = [1, 2] #, 3, 4, 5]

        df_nan = pd.DataFrame(columns=['attendant', 'distance', 'num_nan'])

        #os.listdir(frame_folder)
        for attendant_id in list_attendant:
            for dist in distances:
                # Path to distance folder
                dist_folder = os.path.join(
                    Params.dir_dataset,
                    f'attendant{attendant_id}',
                    str(dist)
                )
                raw_folder     = os.path.join(dist_folder, "raw")
                cropped_folder = os.path.join(dist_folder, "cropped")


                # Validate raw frames exist
                if not os.path.isdir(raw_folder):
                    raise FileNotFoundError(
                        f"Expected raw frames at: {raw_folder}\n"
                    )

                # CropSense
                #crop_stats = run_cropsense(input_dir=raw_folder,
                    output_dir=cropped_folder,
                    croptype="face",  # change to "upperbody"/"fullbody" if needed
                    top_margin=0.2,
                    bottom_margin=0.2,
                    parallel=False  # set to True to use multiprocessing)
                # Find the folder that contains PNG frames
                png_parent = None
                for item in os.listdir(dist_folder):
                    full = os.path.join(dist_folder, item)
                    if os.path.isdir(full) and item.startswith(f'attendant{attendant_id}-'):
                        png_parent = full
                        break

                #if crop_stats["faces_detected"] == 0:
                    #print(f"[Pipeline] WARNING: No faces detected for dist={dist}. Skipping.")
                    #continue



                # Extract RGB signal
                Params.fps = 50
                df_rgb, num_nan = util_analysis.frames_to_sig_stable(
                    frame_folder=cropped_folder,
                    Params=Params
                )

                # Save RGB signals
                save_dir = os.path.join(dir_crt, "data", name_dataset, "rgb")
                os.makedirs(save_dir,exist_ok=True) #make sure the directory exists

                save_path = os.path.join(
                    dir_crt, 'data', name_dataset, 'rgb', f'{attendant_id}_{dist}.csv'
                )
                df_rgb.to_csv(save_path, index=False)
                print("[Pipeline] RGB saved in,", save_path)

                # Save NaN events
                df_nan.loc[len(df_nan)] = [attendant_id, dist, num_nan]

            # Save NaN summary
            nan_path = os.path.join(dir_crt, 'data', name_dataset, 'rgb', 'nan_event.csv')
            df_nan.to_csv(nan_path, index=False)
    # Video -> RGB signal.
    elif name_dataset == '!UBFC-rPPG':
        # Sequnce num of attendants.
        list_attendant = [1] + list(range(3, 6)) + list(range(8, 19)) + [20] + list(range(22, 27)) + list(range(30, 50))
        # Create a log to save num of nan.
        df_nan = pd.DataFrame(columns=['attendant', 'num_nan'])
        for num_attendant in tqdm(list_attendant):
            print([name_dataset, num_attendant])
            # Video directory.
            dir_vid = os.path.join(Params.dir_dataset, 'DATASET_2', 'subject'+str(num_attendant), 'vid.avi')
            # Video fps.
            Params.fps = util_pyVHR.get_fps(dir_vid)
            # RGB signal extraction.
            df_rgb, num_nan = util_analysis.vid_to_sig(dir_vid=dir_vid, Params=Params)
            # Save RGB signals.
            dir_save_data = os.path.join(dir_crt, 'data', name_dataset, 'rgb', str(num_attendant)+'1.csv')
            df_rgb.to_csv(dir_save_data, index=False)
            # Save nan events.
            df_nan.loc[len(df_nan)] = [num_attendant, num_nan] 
            dir_save_nan = os.path.join(dir_crt,'data', name_dataset, 'rgb', 'nan_event.csv')
            df_nan.to_csv(dir_save_nan, index=False)


    elif name_dataset == '!UBFC-Phys':
        # Name of attendants.
        list_attendant = list(range(1, 57))
        # Condition types.
        list_condition = [1, 2, 3]  # [1, 2, 3]
        # Create a log to save num of nan.
        df_nan = pd.DataFrame(columns=['attendant', 'condition', 'num_nan'])
        for num_attendant in tqdm(list_attendant):
            for condition in list_condition:
                print([name_dataset, num_attendant, condition])
                # Video directory.
                dir_vid = os.path.join(Params.dir_dataset, 's'+str(num_attendant), 'vid_s'+str(num_attendant)+'_T'+str(condition)+'.avi')
                # Video fps.
                Params.fps = util_pyVHR.get_fps(dir_vid)
                # RGB signal extraction.
                df_rgb, num_nan = util_analysis.vid_to_sig(dir_vid=dir_vid, Params=Params)
                # Save RGB signals.
                dir_save_data = os.path.join(dir_crt, 'data', name_dataset, 'rgb', str(num_attendant)+'_'+str(condition)+'1.csv')
                df_rgb.to_csv(dir_save_data, index=False)
                # Save nan events.
                df_nan.loc[len(df_nan)] = [num_attendant, condition, num_nan] 
                dir_save_nan = os.path.join(dir_crt,'data', name_dataset, 'rgb', 'nan_event.csv')
                df_nan.to_csv(dir_save_nan, index=False)


    elif name_dataset == '!LGI-PPGI':
        # Name of attendants.
        list_attendant = ['alex', 'angelo', 'cpi', 'david', 'felix', 'harun']
        # Motion types.
        list_motion = ['talk', 'gym', 'resting', 'rotation']
        # Create a log to save num of nan.
        df_nan = pd.DataFrame(columns=['attendant', 'motion', 'num_nan'])
        for attendant in tqdm(list_attendant):
            for motion in list_motion:
                print([name_dataset, attendant, motion])
                # Video directory.
                dir_vid = os.path.join(Params.dir_dataset, attendant, attendant+'_'+motion, 'cv_camera_sensor_stream_handler.avi')
                # Video fps.
                Params.fps = util_pyVHR.get_fps(dir_vid)
                # RGB signal extraction.
                df_rgb, num_nan = util_analysis.vid_to_sig(dir_vid=dir_vid, Params=Params)
                # Save RGB signals.
                dir_save_data = os.path.join(dir_crt, 'data', name_dataset, 'rgb', attendant+'_'+motion+'1.csv')
                df_rgb.to_csv(dir_save_data, index=False)
                # Save nan events.
                df_nan.loc[len(df_nan)] = [attendant, motion, num_nan] 
                dir_save_nan = os.path.join(dir_crt,'data', name_dataset, 'rgb', 'nan_event.csv')
                df_nan.to_csv(dir_save_nan, index=False)
 

    elif name_dataset == '!BUAA-MIHR':
        # Sequnce num of attendants.
        list_attendant = list(range(1, 14))
        # Illumination levels.
        list_lux = ['lux 10.0', 'lux 15.8', 'lux 25.1', 'lux 39.8', 'lux 63.1', 'lux 100.0']
        # Attendant names.
        list_name = ['APH', 'GDB', 'HB', 'KLK', 'LW', 'LXR', 'LY', 'LZY', 'LMX', 'MXZ', 'PYC', 'QSH', 'WQT']
        # Create a log to save num of nan.
        df_nan = pd.DataFrame(columns=['attendant', 'lux', 'num_nan'])
        for num_attendant in tqdm(list_attendant):
            for lux in list_lux:
                name = list_name[num_attendant-1]
                print([num_attendant, name, lux])
                # Video directory.
                dir_vid = os.path.join(Params.dir_dataset, 'Sub '+str(num_attendant).zfill(2), lux, lux.replace(' ', '')+'_'+name+'.avi')
                # Video fps.
                Params.fps = util_pyVHR.get_fps(dir_vid)
                # RGB signal extraction.
                df_rgb, num_nan = util_analysis.vid_to_sig(dir_vid=dir_vid, Params=Params)
                # Save RGB signals.
                dir_save_data = os.path.join(dir_crt, 'data', name_dataset, 'rgb', str(num_attendant).zfill(2)+'_'+str(lux).replace(' ', '')+'1.csv')
                df_rgb.to_csv(dir_save_data, index=False)
                # Save nan events.
                df_nan.loc[len(df_nan)] = [num_attendant, lux, num_nan] 
                dir_save_nan = os.path.join(dir_crt,'data', name_dataset, 'rgb', 'nan_event.csv')
                df_nan.to_csv(dir_save_nan, index=False)


if __name__ == "__main__":
    # Available datasets.
    list_dataset = ['custom','UBFC-rPPG', 'UBFC-Phys', 'LGI-PPGI', 'BUAA-MIHR']  # ['UBFC-rPPG', 'UBFC-Phys', 'LGI-PPGI', 'BUAA-MIHR'].
    # Extract RGB signal.
    for name_dataset in list_dataset:
        main_vid2rgb(name_dataset=name_dataset)