"""
Utils for the analysis of the optimal ROI selection under different conditions.
"""

# Original Author: Shuo Li
# Date: 2023/08/05
# Editor: Matthew Dowell
# Date: 03/10/2026

import warnings
warnings.filterwarnings("ignore")  # Ignore unnecessary warnings.
import os
import cv2
import yaml
import util_pyVHR
import numpy as np
import pandas as pd
import mediapipe as mp
from tqdm import tqdm
from xml.dom import minidom
from sklearn import metrics
from dtaidistance import dtw
from scipy.signal import resample


class Params():
    """Load the pre-defined parameters for preliminary analysis from a YAML file. 
       Create a class.
    """

    def __init__(self, dir_option, name_dataset) -> None:
        """Parameter calss initialization.

        Parameters
        ----------
        dir_option: Directory of the YAML file.
        name_dataset: Name of datasets. ['UBFC-rPPG', 'UBFC-Phys', 'LGI-PPGI', 'BUAA-MIHR']

        Returns
        -------

        """

        # Options.
        self.options = yaml.safe_load(open(dir_option))
        # Url.
        self.url = self.options[name_dataset]['url']
        # Dataset directory.
        self.dir_dataset = self.options[name_dataset]['dir_dataset']
        # Face detection parameters.
        self.max_num_faces = self.options[name_dataset]['max_num_faces']  # Number of target faces.
        self.minDetectionCon = self.options[name_dataset]['minDetectionCon']  # Minimal detection confidence.
        self.minTrackingCon = self.options[name_dataset]['minTrackingCon']  # Minimal tracking confidence.
        # The list containing sequence numbers of selected keypoints of different ROIs. Size = [num_roi].
        self.list_roi_num = self.options[name_dataset]['list_roi_num']
        # The list containing names of different ROIs. Size = [num_roi].
        self.list_roi_name = self.options[name_dataset]['list_roi_name']
        # RGB signal -> windowed signal.
        self.len_window = self.options[name_dataset]['len_window']  # Window length in seconds.
        self.stride_window = self.options[name_dataset]['stride_window']  # Window stride in seconds.
        self.fps = self.options[name_dataset]['fps']  # Frames per second.


class GroundTruth():
    """Load the groundtruth data. (time, PPG waveform, PPG HR). 
       Create a class.
    """

    def __init__(self, dir_dataset, name_dataset) -> None:
        """Groundtruth class initialization.

        Parameters
        ----------
        dir_option: Directory of the YAML file.
        name_dataset: Name of datasets. ['UBFC-rPPG', 'UBFC-Phys', 'LGI-PPGI', 'BUAA-MIHR']

        Returns
        -------

        """

        # Directory of the dataset.
        self.dir_dataset = dir_dataset
        # Dataset name. ['UBFC-rPPG', 'UBFC-Phys', 'LGI-PPGI', 'BUAA-MIHR'].
        self.name_dataset = name_dataset
    
    def get_GT(self, specification, num_frame_interp, slice):
        """Get the ground truth data.

        Parameters
        ----------
        specification: Specificy the dataset.
                       UBFC-rPPG: [condition, num_attendant]
                                  'simple' ~ [5-8, 10-12].
                                  'realistic' ~ [1, 3-5, 8-18, 20, 22-26, 30-49].
                                  Example: ['simple', 6]
                       UBFC-Phys: [num_attendant, num_task].
                                  num_attendant: [1-56].
                                  num_task: [1, 2, 3] - [rest, speech, arithmetic].
                                  Example: [2, 2].
                       LGI-PPGI: [name_attendant, motion].
                                 name_attendant: ['alex', 'angelo', 'cpi', 'david', 'felix', 'harun'].
                                 motion: ['gym', 'resting', 'rotation', 'talk'].
                                 Example: ['alex', 'gym'].
                       BUAA-MIHR: [num_attendant, lux, name].
                                  num_attendant: [1-14].
                                  lux: ['lux 1.0', 'lux 1.6', 'lux 2.5', 'lux 4.0', 'lux 6.3', 'lux 10.0', 
                                        'lux 15.8', 'lux 25.1', 'lux 39.8', 'lux 63.1', 'lux 100.0'].
                                  name: ['APH', 'GDB', 'HB', 'KLK', 'LW', 'LXR', 'LY', 'LZY', 'LMX', 'MXZ', 'PYC', 'QSH', 'WQT'].
        num_frame_interp: Total number of frames after interpolation.
        slice: Select a time window of the signal. [start time, end time]. The time is normalized into [0, 1].

        Returns
        -------
        gtTime: Ground truth time in numpy array. Size = [num_frames].
        gtTrace: Ground truth PPG waveform data in numpy array. Size = [num_frames].
        gtHR: Ground truth HR data in numpy array. Size = [num_frames].
        """
        if self.name_dataset == 'custom':
            dir_crt = r'C:\Users\mattt\projects\rPPG\optimal_roi_rppg\main\data\custom'
            #dir_crt = r"C:\Users\dowellm2\rPPG\optimal_roi_rppg-master\main\data\custom\Yuao1m\1\20251202_155614\Yuao-1m-1min-finger-ground-truth-data-1.csv"
            df_GT = pd.read_csv(dir_crt, header=0)

            # Inspect columns once to confirm names
            print("Columns:", list(df_GT.columns))

            # Adjust these names to match your file
            TIME_COL_NAME = "PC_Timestamp_ms"  # or "timestamp", etc.
            TRACE_COL_NAME = "Signal_Value"  # the PPG amplitude column
            HR_COL_NAME = "HR"  # optional; may or may not exist

            # Convert to numeric safely
            gtTime = pd.to_numeric(df_GT[TIME_COL_NAME], errors='coerce').to_numpy()
            gtTrace = pd.to_numeric(df_GT[TRACE_COL_NAME], errors='coerce').to_numpy(dtype=float)

            gtHR = None
            gtHRcol = pd.to_numeric(df_GT[HR_COL_NAME], errors='coerce').to_numpy(dtype=float)
            # Treat zeros as missing if your device uses 0 to mean "no reading"
            gtHR = np.where(gtHRcol == 0, np.nan, gtHRcol)


        elif self.name_dataset == '!UBFC-rPPG':  # UBFC-rPPG dataset.
            
            if specification[0] == 'simple':  # Simple. 
                dir_crt = os.path.join(self.dir_dataset, 'DATASET_1', str(specification[1])+'-gt', 'gtdump.xmp')
                df_GT = pd.read_csv(dir_crt, header=None)
                gtTime = df_GT[0].values/1000
                gtTrace = df_GT[3].values
                gtHR = df_GT[1].values
                
            elif specification[0] == 'realistic':  # Realistic.
                dir_crt = os.path.join(self.dir_dataset, 'DATASET_2', 'subject'+str(specification[1]), 'ground_truth.txt')
                npy_GT = np.loadtxt(dir_crt)
                gtTime = npy_GT[2, :]
                gtTrace = npy_GT[0, :]
                gtHR = npy_GT[1, :]


        elif self.name_dataset == '!UBFC-Phys':  # UBFC-Phys dataset.
            # Groundtruth BVP trace.
            dir_bvp = os.path.join(self.dir_dataset, 's'+str(specification[0]), 'bvp_s'+str(specification[0])+'_T'+str(specification[1])+'1.csv')
            gtTrace = np.loadtxt(dir_bvp)
            # Groundtruth video.
            dir_vid = os.path.join(self.dir_dataset, 's'+str(specification[0]), 'vid_s'+str(specification[0])+'_T'+str(specification[1])+'.avi')
            capture = cv2.VideoCapture(dir_vid)
            fps = capture.get(cv2.CAP_PROP_FPS)  # Frame rate.
            num_frame = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))   # Number of frames.
            duration = num_frame/fps  # Video duration. (sec).
            # Groundtruth time.
            gtTime = np.linspace(start=0, stop=duration, num=num_frame)
            # Groundtruth hr.
            dir_bpm = os.path.join(self.dir_dataset, 's'+str(specification[0]), 'bpm_s'+str(specification[0])+'_T'+str(specification[1])+'1.csv')
            gtHR = np.loadtxt(dir_bpm)


        elif self.name_dataset == '!LGI-PPGI':  # LGI-PPGI dataset.
            dir_vid = os.path.join(self.dir_dataset, str(specification[0]), specification[0]+'_'+specification[1], 'cv_camera_sensor_stream_handler.avi')
            dir_xml = os.path.join(self.dir_dataset, specification[0], specification[0]+'_'+specification[1], 'cms50_stream_handler.xml')
            dom = minidom.parse(dir_xml)
            # Ground truth heart rate.
            value_HR = dom.getElementsByTagName('value1')
            # Ground truth trace.
            value_Trace = dom.getElementsByTagName('value2')
            gtHR = []
            gtTrace = []
            for i in range(len(value_HR)):
                HR_tmp = value_HR[i].firstChild.data
                if '\n' not in HR_tmp:  # Exclude invalid data.
                    gtHR.append(int(HR_tmp))
                Trace_tmp = value_Trace[i].firstChild.data
                if '\n' not in Trace_tmp:  # Exclude invalid data.
                    gtTrace.append(int(Trace_tmp))
            # Ground truth time.
            capture = cv2.VideoCapture(dir_vid)
            fps = capture.get(cv2.CAP_PROP_FPS)  # Frame rate.
            num_frame = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))  # Number of frames.
            duration = num_frame/fps  # Video duration. (sec).
            gtTime = np.linspace(start=0, stop=duration, num=num_frame)
            # list -> numpy array.
            gtHR = np.array(gtHR)
            gtTrace = np.array(gtTrace)

        
        elif self.name_dataset == '!BUAA-MIHR':  # BUAA-MIHR dataset.
            dir_crt = os.path.join(self.dir_dataset, 'Sub '+str(specification[0]).zfill(2), specification[1])
            # PPG trace wave.
            gtTrace = np.loadtxt(os.path.join(dir_crt, specification[1].replace(' ', '')+'_'+specification[2]+'_wave.csv'))
            # Time stamp.
            # RGB video information.
            dir_vid = os.path.join(dir_crt, specification[1].replace(' ', '')+'_'+specification[2]+'.avi')
            # Get video fps.
            capture = cv2.VideoCapture(dir_vid)
            fps = capture.get(cv2.CAP_PROP_FPS)
            num_frame = capture.get(cv2.CAP_PROP_FRAME_COUNT)
            duration = num_frame/fps
            gtTime = np.linspace(start=0, stop=duration, num=int(num_frame))
            # HR data.
            df_HR = pd.read_csv(os.path.join(dir_crt, specification[1].replace(' ', '')+'_'+specification[2]+'1.csv'))
            gtHR = df_HR['PULSE'].values
            # HR signal resampling.
        
        # Resampling according to gtTime.
        gtTrace = resample(x=gtTrace, num=num_frame_interp)
        gtHR = resample(x=gtHR, num=num_frame_interp)
        # Time windowing.
        frame_start = round(slice[0] * len(gtTime))
        frame_end = round(slice[1] * len(gtTime))
        gtTime = gtTime[frame_start:frame_end]
        gtTrace = gtTrace[frame_start:frame_end]
        gtTrace = (gtTrace - np.min(gtTrace))/(np.max(gtTrace) - np.min(gtTrace))  # Normalize into [0, 1].
        gtHR = gtHR[frame_start:frame_end]

        return gtTime, gtTrace, gtHR


EMA_ALPHA = 0.25  # EMA weight for new landmark position (0=frozen, 1=no smoothing)
POSE_UPDATE_THRESH = 0.015  # Normalised landmark shift to trigger a full update (~1.5% of frame)
QUALITY_GATE_ZSCORE = 3.5  # Flag frames where landmark shift z-score > this value
MIN_VALID_FRAMES = 0.5  # Fraction of frames that must be valid; warn if below this

class KalmanPoint:
    #Minimal constant-velocity Kalman filter for one (x, y) landmark.

    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)  # state=[x,y,vx,vy], obs=[x,y]
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1],
                                             [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-4
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-2
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)
        self.initialized = False

    def update(self, x: float, y: float):
        meas = np.array([[x], [y]], dtype=np.float32)
        if not self.initialized:
            self.kf.statePre = np.array([[x], [y], [0], [0]], dtype=np.float32)
            self.initialized = True
        self.kf.predict()
        state = self.kf.correct(meas)
        return float(state[0]), float(state[1])



class FaceDetector():
    """A class for face detection, segmentation and RGB signal extraction."""

    def __init__(self, Params, smoothing='ema',
                 use_quality_gate=True, use_motion_comp=True):
        """Class initialization.
        Parameters
        ----------
        Params: A class containing the pre-defined parameters.

        Returns
        -------

        """

        # Confidence.
        self.minDetectionCon = Params.minDetectionCon  # Minimal detection confidence.
        self.minTrackingCon = Params.minTrackingCon  # Minimal tracking confidence.
        # Mediapipe utils.
        self.mpFaceDetection = mp.solutions.face_detection
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.minDetectionCon)  # Face detection.
        self.mpDraw = mp.solutions.drawing_utils  # Drawing utils.
        self.faceMesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=Params.max_num_faces,
            min_detection_confidence=self.minDetectionCon,
            min_tracking_confidence=self.minTrackingCon
            )  # Face mesh.
        # ROI params.
        # The list containing sequence numbers of selected keypoints of different ROIs. Size = [num_roi].
        self.list_roi_num = np.array(Params.list_roi_num, dtype=object)
        # The list containing names of different ROIs. Size = [num_roi].
        self.list_roi_name = np.array(Params.list_roi_name, dtype=object)
        # Jitter reduction
        self.use_quality_gate = use_quality_gate
        self.use_motion_comp = use_motion_comp
        self.smoothing = smoothing
        self.smoothed_landmarks = None  # shape [468, 3]
        self.prev_landmarks = None  # for shift computation
        self.shift_history = []  # list of per-frame mean shifts

        if smoothing == 'kalman':
            self.kalman_filters = [KalmanPoint() for _ in range(468)]

    # smooth a new raw landmark array

    def smooth_landmarks(self, raw: np.ndarray) -> np.ndarray:
        """Apply EMA or Kalman smoothing to raw [468,3] landmark array."""
        if self.smoothing == 'kalman':
            smoothed = raw.copy()
            for i, kf in enumerate(self.kalman_filters):
                sx, sy = kf.update(raw[i, 0], raw[i, 1])
                smoothed[i, 0] = sx
                smoothed[i, 1] = sy
            return smoothed
        else:
            # EMA
            if self.smoothed_landmarks is None:
                return raw.copy()
            return (EMA_ALPHA * raw
                    + (1.0 - EMA_ALPHA) * self.smoothed_landmarks)

    # mean landmark shift

    def landmark_shift(self, new: np.ndarray) -> float:
        """Mean Euclidean shift of x,y coords vs previous smoothed position."""
        if self.prev_landmarks is None:
            return 0.0
        diff = new[:, :2] - self.prev_landmarks[:, :2]
        return float(np.mean(np.linalg.norm(diff, axis=1)))

    def extract_landmark(self, img):
        img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.faceMesh.process(img_RGB)
        quality = True

        if results.multi_face_landmarks:
            for face_landmark in results.multi_face_landmarks:
                raw = np.zeros([len(face_landmark.landmark), 3], dtype=np.float32)
                for i, lm in enumerate(face_landmark.landmark):
                    raw[i] = [lm.x, lm.y, lm.z]

            shift = self.landmark_shift(raw)
            self.shift_history.append(shift)

            # Quality gate: flag ONLY — does NOT block the update
            if self.use_quality_gate and len(self.shift_history) > 10:
                mu = np.mean(self.shift_history)
                sig = np.std(self.shift_history) + 1e-9
                if (shift - mu) / sig > QUALITY_GATE_ZSCORE:
                    quality = False

            # ALWAYS update with EMA smoothing — never freeze landmarks
            smoothed = self.smooth_landmarks(raw)
            self.smoothed_landmarks = smoothed
            self.prev_landmarks = smoothed.copy()

            return self.smoothed_landmarks, quality

        else:
            self.shift_history.append(0.0)
            return np.nan, False

    def extract_RGB(self, img, loc_landmark):
        if not isinstance(loc_landmark, np.ndarray) or np.isnan(loc_landmark).any():
            return np.full((self.list_roi_num.shape[0], 3), np.nan)

        img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height_img = img.shape[0]
        width_img = img.shape[1]

        # COPY before scaling — never mutate self.smoothed_landmarks
        lm = loc_landmark.copy()
        lm[:, 0] = lm[:, 0] * width_img
        lm[:, 1] = lm[:, 1] * height_img

        sig_rgb = np.zeros(shape=[self.list_roi_num.shape[0], 3])
        zeros = np.zeros(img.shape, dtype=np.uint8)

        for i_roi in range(self.list_roi_num.shape[0]):
            roi_name = self.list_roi_name[i_roi]
            mask = cv2.fillPoly(
                zeros.copy(),
                [lm[self.list_roi_num[self.list_roi_name == roi_name][0], :2].astype(int)],
                color=(1, 1, 1)
            )
            img_masked = np.multiply(img_RGB, mask)
            mask_sum = mask.sum()
            if mask_sum > 0:
                sig_rgb[i_roi, :] = 3 * img_masked.sum(0).sum(0) / mask_sum
            else:
                sig_rgb[i_roi, :] = np.nan

        return sig_rgb

    def faceMeshDraw(self, img, roi_name):
        """Draw a face mesh annotations on the input image.
        Parameters
        ----------
        img: The input image of the current frame.
        roi_name: Name of the roi. The name should be in the name list.

        Returns
        -------
        img_draw: The output image after drawing the ROI of the current frame. 
        """

        # Always have a default to return
        img_draw = img.copy()

        img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR -> RGB.
        results = self.faceMesh.process(img_RGB)  # Apply face mesh.
        mp_face_mesh = mp.solutions.face_mesh_connections
        # Draw landmarks on the image.
        if results.multi_face_landmarks:
            # Loop over all detected faces.
            # In this experiment, we only detect one face in one video.
            for face_landmark in results.multi_face_landmarks:
                # Landmark points.
                self.mpDraw.draw_landmarks(
                    image=img,
                    landmark_list=face_landmark,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=self.mpDraw.DrawingSpec(thickness=1, circle_radius=1),
                    connection_drawing_spec=self.mpDraw.DrawingSpec(thickness=1, circle_radius=1)
                )
                # Decompose the 3D face landmarks.
                height_img = img.shape[0]
                width_img = img.shape[1]
                loc_landmark = np.zeros([len(face_landmark.landmark), 2], dtype=np.int32)  # Coordinates of 2D landmarks.
                for i in range(len(face_landmark.landmark)):
                    loc_landmark[i, 0] = face_landmark.landmark[i].x * width_img
                    loc_landmark[i, 1] = face_landmark.landmark[i].y * height_img
                # Create a zero vector for mask construction.
                zeros = np.zeros(img.shape, dtype=np.uint8)
                # ROI-forehead-nose-leftcheek-rightcheek-underlip. Colorization.
                mask = cv2.fillPoly(zeros.copy(), [loc_landmark[self.list_roi_num[self.list_roi_name==roi_name][0], :]], color=(1, 1, 1))
                img_draw = img + mask * 50
            
        return img_draw

def motion_compensate(df_rgb: pd.DataFrame,
                       roi_names: list) -> pd.DataFrame:
    #Regress out the global luminance proxy from each ROI's R, G, B traces.

    frames = sorted(df_rgb['frame'].unique())
    # Build global green proxy [num_frames]
    green_proxy = np.array([
        df_rgb.loc[df_rgb['frame'] == f, 'G'].mean()
        for f in frames
    ])
    # Normalise proxy to zero mean
    green_proxy = green_proxy - green_proxy.mean()

    # Regress proxy out of each channel of each ROI
    for roi in roi_names:
        mask = df_rgb['ROI'] == roi
        for ch in ['R', 'G', 'B']:
            trace = df_rgb.loc[mask, ch].values.astype(np.float64)
            if np.isnan(trace).all():
                continue
            # OLS regression: trace = a*proxy + b  →  residual = trace - a*proxy
            valid = np.isfinite(trace) & np.isfinite(green_proxy)
            if valid.sum() < 2:
                continue
            a = (np.cov(trace[valid], green_proxy[valid])[0, 1]
                 / (np.var(green_proxy[valid]) + 1e-12))
            df_rgb.loc[mask, ch] = trace - a * green_proxy

    return df_rgb

# Stabilized frames_to_sig

def frames_to_sig_stable(frame_folder: str, Params,
                         smoothing: str = 'ema',
                         use_quality_gate: bool = True,
                         use_motion_comp: bool = True):
    #Adds: Temporal landmark smoothing (EMA or Kalman)
    # Motion-aware update threshold (suppress micro-jitter)
    # Quality gating (outlier frames → NaN → interpolated)
    #Signal-space motion compensation (global luminance regression)

    Detector = FaceDetector(
        Params,
        smoothing=smoothing,
        use_quality_gate=use_quality_gate,
        use_motion_comp=use_motion_comp,
    )

    df_rgb = pd.DataFrame(columns=['frame', 'time', 'ROI', 'R', 'G', 'B'])
    num_frame = 0
    gated_frames = 0

    frame_files = sorted([
        f for f in os.listdir(frame_folder)
        if not f.endswith("face_1.png")
    ])

    progress = tqdm(frame_files, desc="[stable] Extracting ROI signals",
                    dynamic_ncols=True)

    for f in progress:
        img_frame = cv2.imread(os.path.join(frame_folder, f))
        if img_frame is None:
            continue

        # Stabilized landmark extraction
        loc_landmark, quality = Detector.extract_landmark(img=img_frame)

        if not quality:
            gated_frames += 1

        # Extract RGB (uses stabilized landmarks; gated frames → NaN)
        if quality:
            sig_rgb = Detector.extract_RGB(img=img_frame,
                                           loc_landmark=loc_landmark)
        else:
            sig_rgb = np.nan  # will be interpolated below

        # Build per-frame rows
        num_frame += 1
        df_tmp = pd.DataFrame(
            columns=['frame', 'time', 'ROI', 'R', 'G', 'B'],
            index=range(len(Params.list_roi_name))
        )
        for i_roi, roi_name in enumerate(Params.list_roi_name):
            df_tmp.loc[i_roi, 'ROI'] = roi_name
            if (isinstance(sig_rgb, float) and np.isnan(sig_rgb)) or \
                    (isinstance(sig_rgb, np.ndarray) and np.isnan(sig_rgb).any()):
                df_tmp.loc[i_roi, ['R', 'G', 'B']] = np.nan
            else:
                df_tmp.loc[i_roi, ['R', 'G', 'B']] = sig_rgb[i_roi, :]

        df_tmp['frame'] = num_frame
        df_tmp['time'] = num_frame * Params.fps
        df_tmp[['frame']] = df_tmp[['frame']].astype('int')
        df_tmp[['time', 'R', 'G', 'B']] = df_tmp[['time', 'R', 'G', 'B']].astype('float')
        df_rgb = pd.concat([df_rgb, df_tmp])

    df_rgb = df_rgb.reset_index(drop=True)
    num_nan = df_rgb.isnull().sum().sum()

    # Interpolate NaN frames (quality-gated or no-detection)
    for roi_name in Params.list_roi_name:
        mask = df_rgb['ROI'] == roi_name
        df_rgb.loc[mask] = (df_rgb.loc[mask]
                            .interpolate(method='linear')
                            .ffill()
                            .bfill())

    # Signal-space motion compensation (after interpolation)
    if use_motion_comp:
        df_rgb = motion_compensate(df_rgb, list(Params.list_roi_name))

    # Report gating stats
    total = num_frame if num_frame > 0 else 1
    gate_pct = 100 * gated_frames / total
    print(f"\n[stable] Frames processed : {num_frame}")
    print(f"[stable] Frames gated out : {gated_frames} ({gate_pct:.1f}%)")
    print(f"[stable] NaN values before interpolation: {num_nan}")
    if gated_frames / total > (1 - MIN_VALID_FRAMES):
        print(f"[stable] WARNING: >{100 * (1 - MIN_VALID_FRAMES):.0f}% of frames "
              "were gated — check lighting or face occlusion.")

    return df_rgb, num_nan

def frames_to_sig(frame_folder, Params):
    """Transform the input frames into RGB signals.
       Return the signals as pandas dataframe.

    Parameters
    ----------
    frame_folder: Directory of the raw input frames.
    Params: A class containing the pre-defined parameters for the preliminary analysis.

    Returns
    -------
    df_rgb: Dataframe containing the RGB signal of the input video.
    num_nan: Number of nan values of the extracted RGB signal.
        """
    counter = 0
    # Create the face detection object.
    Detector_crt = FaceDetector(Params=Params)
    # Create the dataframe containing the RGB signals and other necessary data.
    df_rgb = pd.DataFrame(columns=['frame', 'time', 'ROI', 'R', 'G', 'B'])
    # Start processing each frame.
    num_frame = 0
    # grabbing frames instead of .avi video
    frame_files = sorted(os.listdir(frame_folder))
    progress_bar = tqdm(frame_files, desc="[optimal_roi] Mapping", dynamic_ncols=True)
    for f in frame_files:
        counter += 1
        if counter > 500:
            continue
        # Skip any file ending in face_1.png
        if f.endswith("face_1.png"):
            continue
        img_frame = cv2.imread(os.path.join(frame_folder, f))
        if img_frame is None:
            print("Skipping unreadable file:")
            continue
        # Detect facial landmark keypoints. The locations are normalized into [0, 1].
        loc_landmark = Detector_crt.extract_landmark(img=img_frame)  # Size = [468, 3]
        # Extract RGB signal.

        progress_bar.update(1)
        #print(counter, "frames out of 4000")
        sig_rgb = Detector_crt.extract_RGB(img=img_frame, loc_landmark=loc_landmark)  # Size = [num_roi, 3].
        # Loop over all ROIs and save the RGB data.
        df_rgb_tmp = pd.DataFrame(columns=['frame', 'time', 'ROI', 'R', 'G', 'B'],
                                  index=list(range(0, len(Params.list_roi_name))))
        for i_roi in range(len(Params.list_roi_name)):
            # ROI name.
            df_rgb_tmp.loc[i_roi, 'ROI'] = Params.list_roi_name[i_roi]
            if (isinstance(sig_rgb, float) and np.isnan(sig_rgb)) or \
                    (isinstance(sig_rgb, np.ndarray) and np.isnan(sig_rgb).any()):
                # If no face is detected.
                df_rgb_tmp.loc[i_roi, ['R', 'G', 'B']] = np.nan
            else:
                # show image
                #roi_name = Params.list_roi_name[i_roi]  # index -> name
                #img_draw = Detector_crt.faceMeshDraw(img_frame, roi_name)
                #cv2.imshow("img_draw", img_draw)
                #cv2.waitKey(1)  # let window update

                # If the face is detected.
                # RGB channels.
                df_rgb_tmp.loc[i_roi, ['R', 'G', 'B']] = sig_rgb[i_roi, :]
        # Sequence number of frame.
        num_frame = num_frame + 1
        df_rgb_tmp.loc[:, 'frame'] = num_frame
        # Time of the current frame.
        df_rgb_tmp.loc[:, 'time'] = num_frame * Params.fps
        # Change data format into numeric.
        df_rgb_tmp[['frame']] = df_rgb_tmp[['frame']].astype('int')
        df_rgb_tmp[['time', 'R', 'G', 'B']] = df_rgb_tmp[['time', 'R', 'G', 'B']].astype('float')
        # Attach to the main dataframe.
        df_rgb = pd.concat([df_rgb, df_rgb_tmp])
        # Dataframe reindex.
    df_rgb = df_rgb.reset_index(drop=True)
    # For frames with nan values, use time interpolation.
    num_nan = df_rgb.isnull().sum().sum()
    for roi_name in Params.list_roi_name:
        df_rgb.loc[df_rgb['ROI'].values == roi_name, :] = df_rgb.loc[df_rgb['ROI'].values == roi_name, :].interpolate(
            method='linear')
        df_rgb.loc[df_rgb['ROI'].values == roi_name, :] = df_rgb.loc[df_rgb['ROI'].values == roi_name, :].fillna(
            method='ffill')
        df_rgb.loc[df_rgb['ROI'].values == roi_name, :] = df_rgb.loc[df_rgb['ROI'].values == roi_name, :].fillna(
            method='bfill')

    return df_rgb, num_nan
def vid_to_sig(dir_vid, Params):
    """Transform the input video into RGB signals. 
       Return the signals as pandas dataframe.

    Parameters
    ----------
    dir_vid: Directory of the input video.
    Params: A class containing the pre-defined parameters for the preliminary analysis.

    Returns
    -------
    df_rgb: Dataframe containing the RGB signal of the input video.
    num_nan: Number of nan values of the extracted RGB signal.
    """

    # Input video.
    video_crt = cv2.VideoCapture(dir_vid)
    # Create the face detection object.
    Detector_crt = FaceDetector(Params=Params)
    # Create the dataframe containing the RGB signals and other necessary data.
    df_rgb = pd.DataFrame(columns=['frame', 'time', 'ROI', 'R', 'G', 'B'])
    # Start processing each frame.
    num_frame = 0
    while(video_crt.isOpened()):
        ret, img_frame = video_crt.read()
        if (ret == False) or (cv2.waitKey(1) & 0xFF == ord('q')):
            # Terminate in the end.
            break
        # Detect facial landmark keypoints. The locations are normalized into [0, 1].
        loc_landmark, quality = Detector_crt.extract_landmark(img=img_frame)
        # Extract RGB signal.
        sig_rgb = Detector_crt.extract_RGB(img=img_frame, loc_landmark=loc_landmark)  # Size = [num_roi, 3].
        # Loop over all ROIs and save the RGB data.
        df_rgb_tmp = pd.DataFrame(columns=['frame', 'time', 'ROI', 'R', 'G', 'B'], index=list(range(0, len(Params.list_roi_name))))
        for i_roi in range(len(Params.list_roi_name)):
            # ROI name.
            df_rgb_tmp.loc[i_roi, 'ROI'] = Params.list_roi_name[i_roi]
            if (np.isnan(sig_rgb)).any() == True:
                # If no face is detected.
                df_rgb_tmp.loc[i_roi, ['R', 'G', 'B']] = np.nan
            else:
                # If the face is detected.
                # RGB channels.
                df_rgb_tmp.loc[i_roi, ['R', 'G', 'B']] = sig_rgb[i_roi, :]
        # Sequence number of frame.
        num_frame = num_frame + 1
        df_rgb_tmp.loc[:, 'frame'] = num_frame
        # Time of the current frame.
        df_rgb_tmp.loc[:, 'time'] = num_frame * Params.fps
        # Change data format into numeric.
        df_rgb_tmp[['frame']] = df_rgb_tmp[['frame']].astype('int')
        df_rgb_tmp[['time', 'R', 'G', 'B']] = df_rgb_tmp[['time', 'R', 'G', 'B']].astype('float')
        # Attach to the main dataframe.
        df_rgb = pd.concat([df_rgb, df_rgb_tmp])
    # Dataframe reindex.
    df_rgb = df_rgb.reset_index(drop=True)
    # For frames with nan values, use time interpolation. 
    num_nan = df_rgb.isnull().sum().sum()
    for roi_name in Params.list_roi_name:
        df_rgb.loc[df_rgb['ROI'].values==roi_name, :] = df_rgb.loc[df_rgb['ROI'].values==roi_name, :].interpolate(method='linear')
        df_rgb.loc[df_rgb['ROI'].values==roi_name, :] = df_rgb.loc[df_rgb['ROI'].values==roi_name, :].fillna(method='ffill')
        df_rgb.loc[df_rgb['ROI'].values==roi_name, :] = df_rgb.loc[df_rgb['ROI'].values==roi_name, :].fillna(method='bfill')

    return df_rgb, num_nan


def sig_to_windowed(sig_rgb, Params):
    """Transform the original RGB signals into windowed RGB signals.

    Parameters
    ----------
    sig_rgb: The extracted RGB signal of different ROIs. Size: [num_frames, num_ROI, rgb_channels].
    Params: A class containing the pre-defined parameters for the preliminary analysis.

    Returns
    -------
    sig_rgb_win: The windowed rgb signals. Size: [num_estimators, rgb_channels, window_frames].
    timesES: An array of times in seconds.
    """

    # Parameter parsing.
    len_window = Params.len_window  # Window length in seconds.
    stride_window = Params.stride_window  # Window overlap in seconds.
    fps = Params.fps  # Frames per second.
    # Signal windowing.
    sig_rgb_win , timesES = util_pyVHR.sig_windowing(sig_rgb, len_window, stride_window, fps)

    return sig_rgb_win, timesES


def sig_windowed_to_bvp(sig_rgb_win, method, Params):
    """Transform the windowed RGB signals into blood volume pulse (BVP) signals.

    Parameters
    ----------
    sig_rgb_win: The windowed rgb signals. Size: [num_estimators, rgb_channels, window_frames].
    method: Selected rPPG algorithm. ['CHROM', 'GREEN', 'ICA', 'LGI', 'OMIT', 'PBV', 'POS', 'OMIT'].
    Params: Pre-defined parameter structure.

    Returns
    -------
    sig_bvp_win: The windowed bvp(Blood Volume Pulse) signal.
    """

    # Selected rPPG algorithms. Windowed signal -> bvp signal.
    if method == 'CHROM':
        sig_bvp_win = util_pyVHR.RGB_sig_to_BVP(windowed_sig=sig_rgb_win, fps=Params.fps, method=util_pyVHR.cpu_CHROM)
    elif method == 'GREEN':
        sig_bvp_win = util_pyVHR.RGB_sig_to_BVP(windowed_sig=sig_rgb_win, fps=Params.fps, method=util_pyVHR.cpu_GREEN)
    elif method == 'ICA':
        sig_bvp_win = util_pyVHR.RGB_sig_to_BVP(windowed_sig=sig_rgb_win, fps=Params.fps, method=util_pyVHR.cpu_ICA, params={'component': 'second_comp'})
    elif method == 'LGI':
        sig_bvp_win = util_pyVHR.RGB_sig_to_BVP(windowed_sig=sig_rgb_win, fps=Params.fps, method=util_pyVHR.cpu_LGI)
    elif method == 'OMIT':
        sig_bvp_win = util_pyVHR.RGB_sig_to_BVP(windowed_sig=sig_rgb_win, fps=Params.fps, method=util_pyVHR.cpu_OMIT)
    elif method == 'PBV':
        sig_bvp_win = util_pyVHR.RGB_sig_to_BVP(windowed_sig=sig_rgb_win, fps=Params.fps, method=util_pyVHR.cpu_PBV)
    elif method == 'PCA':
        sig_bvp_win = util_pyVHR.RGB_sig_to_BVP(windowed_sig=sig_rgb_win, fps=Params.fps, method=util_pyVHR.cpu_PCA, params={'component': 'second_comp'})
    elif method == 'POS':
        sig_bvp_win = util_pyVHR.RGB_sig_to_BVP(windowed_sig=sig_rgb_win, fps=Params.fps, method=util_pyVHR.cpu_POS, params={'fps':Params.fps})
    
    return sig_bvp_win


def rppg_hr_pipe(sig_rgb, method, Params):
    """The complete pipeline of transforming raw RGB traces into BVP & HR signals.

    Parameters
    ----------
    sig_rgb: The extracted RGB signal of different ROIs. Size: [num_frames, num_ROI, rgb_channels].
    method: Selected rPPG algorithm. ['CHROM', 'GREEN', 'ICA', 'LGI', 'OMIT', 'PBV', 'POS', 'OMIT'].
    Params: A class containing the pre-defined parameters for the preliminary analysis.

    Returns
    -------
    sig_bvp: Blood volume pulse (BVP) signal of different ROI without windowing. Size=[num_frames, num_ROI].
    sig_bpm: Beats per minute (BPM) signal of different ROI. Size=[num_frames, num_ROI].
    """
    # RGB signal -> windowed RGB signal.
    sig_rgb_win, timeES = sig_to_windowed(sig_rgb=sig_rgb, Params=Params)
    # Windowed RGB signal -> windowed raw bvp signal.
    sig_bvp_win = sig_windowed_to_bvp(sig_rgb_win=sig_rgb_win, method=method, Params=Params)

    # --- 3. Filter BVP windows ---
    sig_bvp_win_filtered = util_pyVHR.apply_filter(
        sig_bvp_win,
        util_pyVHR.BPfilter,
        params={'order': 6, 'minHz': 0.65, 'maxHz': 4.0, 'fps': Params.fps}
    )

    # Replace NaN windows
    for i in range(len(sig_bvp_win_filtered)):
        if np.any(np.isnan(sig_bvp_win_filtered[i])):
            if i == 0:
                sig_bvp_win_filtered[i] = np.ones_like(sig_bvp_win_filtered[i])
            else:
                sig_bvp_win_filtered[i] = sig_bvp_win_filtered[i - 1]

    sig_bvp = 0 #base case
    # --- 4. De-window BVP back to full-length ---
    for i in range(len(sig_bvp_win_filtered)):
        if i == 0:
            sig_bvp = sig_bvp_win_filtered[i][:, :round(Params.fps * Params.stride_window)]
        else:
            sig_bvp = np.concatenate(
                (sig_bvp, sig_bvp_win_filtered[i][:, :round(Params.fps * Params.stride_window)]),
                axis=1
            )
    print("sig_rgb shape:", getattr(sig_rgb, "shape", None))
    N = sig_rgb.shape[0]
    win = round(Params.fps * Params.len_window)
    hop = max(1, round(Params.fps * Params.stride_window))
    print(f"N frames={N}, win={win} samples, hop={hop} samples, fps={Params.fps}")

    # Add last partial window
    sig_bvp = np.concatenate(
        (sig_bvp, sig_bvp_win_filtered[-1][:, round(Params.fps * Params.stride_window):]),
        axis=1
    )

    # --- 5. Compute BPM + SNR per window AND per ROI ---
    num_windows = len(sig_bvp_win_filtered)
    num_rois = sig_bvp_win_filtered[0].shape[0]

    multi_sig_bpm = np.zeros((num_windows, num_rois))
    multi_sig_snr = np.zeros((num_windows, num_rois))

    for w, win in enumerate(sig_bvp_win_filtered):
        for r in range(num_rois):
            roi_signal = win[r, :].astype(np.float32)

            bpm_obj = util_pyVHR.BPM(
                data=roi_signal,
                fps=Params.fps,
                minHz=0.65,
                maxHz=4.0
            )

            bpm_val = bpm_obj.BVP_to_BPM()  # already in BPM
            snr_val = bpm_obj.compute_snr()#compute_snr_hr(bpm_val)

            multi_sig_bpm[w, r] = bpm_val
            multi_sig_snr[w, r] = snr_val

    #multi_sig_bpm = np.array(multi_sig_bpm)  # shape: [num_windows, num_estimators]
    #multi_sig_snr = np.array(multi_sig_snr)  # shape: [num_windows]

    # Replace NaN BPM windows
    for i in range(len(multi_sig_bpm)):
        if np.any(np.isnan(multi_sig_bpm[i])):
            if i == 0:
                multi_sig_bpm[i] = np.zeros_like(multi_sig_bpm[i])
            else:
                multi_sig_bpm[i] = multi_sig_bpm[i - 1]

    # --- 6. Interpolate BPM + SNR back to frame-level ---
    sig_bvp_old = sig_bvp.T  # [num_frames, num_ROI]
    sig_bpm_old = multi_sig_bpm.T  # [num_ROI, num_windows]

    sig_bvp = np.zeros_like(sig_rgb[:, :, 0])
    sig_bpm = np.zeros_like(sig_rgb[:, :, 0])
    sig_snr = np.zeros_like(sig_rgb[:, :, 0])

    for i_roi in range(sig_bpm_old.shape[0]):
        # BVP interpolation
        sig_bvp[:, i_roi] = np.interp(
            x=np.linspace(0, len(sig_bvp), len(sig_bvp)),
            xp=np.linspace(0, len(sig_bvp), len(sig_bvp_old)),
            fp=sig_bvp_old[:, i_roi]
        )

        # BPM interpolation
        sig_bpm[:, i_roi] = np.interp(
            x=np.linspace(0, len(sig_bpm), len(sig_bpm)),
            xp=np.linspace(0, len(sig_bpm), len(sig_bpm_old[i_roi])),
            fp=sig_bpm_old[i_roi]
        )

        # SNR interpolation (per ROI)
        sig_snr[:, i_roi] = np.interp(
            x=np.linspace(0, len(sig_snr), len(sig_snr)),
            xp=np.linspace(0, len(multi_sig_snr[:, i_roi]), len(multi_sig_snr[:, i_roi])),
            fp=multi_sig_snr[:, i_roi]
        )

    return sig_bvp, sig_bpm, sig_snr


def eval_pipe(sig_bvp, sig_bpm, gtTrace, gtHR, Params, roi_names=None):
    """
    Clean evaluation for custom dataset.
    Evaluates BPM vs GT BPM, per ROI, per frame.

    Parameters
    ----------
    sig_bvp : ignored (kept for compatibility)
    sig_bpm : array [num_frames, num_rois]
    gtTrace : ignored (kept for compatibility)
    gtHR    : ground truth BPM [num_frames]
    Params  : unused except for compatibility
    roi_names : list of ROI names (e.g., ["ROI29", "ROI30"])
    """

    if roi_names is None:
        raise ValueError("roi_names must be provided for custom dataset")

    num_rois = len(roi_names)
    num_frames = len(gtHR)

    # Metric arrays
    list_PCC = np.zeros(num_rois)
    list_CCC = np.zeros(num_rois)
    list_RMSE = np.zeros(num_rois)
    list_MAE = np.zeros(num_rois)
    list_MAPE = np.zeros(num_rois)

    # --- Metric helpers ---
    def ccc(x, y):
        mean_x, mean_y = np.mean(x), np.mean(y)
        var_x, var_y = np.var(x), np.var(y)
        cov = np.mean((x - mean_x) * (y - mean_y))
        return (2 * cov) / (var_x + var_y + (mean_x - mean_y) ** 2)

    def safe_rmse(a, b):
        return np.sqrt(np.mean((a - b) ** 2))

    def safe_mae(a, b):
        return np.mean(np.abs(a - b))

    def safe_mape(a, b):
        return np.mean(np.abs((a - b) / a)) * 100

    # --- Evaluate each ROI independently ---
    for i in range(num_rois):
        est = sig_bpm[:, i].astype(float)
        gt = gtHR.astype(float)

        # Ensure equal length
        N = min(len(est), len(gt))
        est = est[:N]
        gt = gt[:N]

        # Metrics
        list_PCC[i] = np.corrcoef(est, gt)[0, 1]
        list_CCC[i] = ccc(est, gt)
        list_RMSE[i] = safe_rmse(est, gt)
        list_MAE[i] = safe_mae(est, gt)
        list_MAPE[i] = safe_mape(est, gt)

    # Build output DataFrame
    df_metric = pd.DataFrame({
        "ROI": roi_names,
        "DTW": [np.nan] * num_rois,  # Not used for custom dataset
        "PCC": list_PCC,
        "CCC": list_CCC,
        "RMSE": list_RMSE,
        "MAE": list_MAE,
        "MAPE": list_MAPE,
    })

    return df_metric

def safe_rmse(a, b):
    """Correct RMSE (sqrt of MSE), with NaN guards."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() == 0:
        return np.nan
    return float(np.sqrt(metrics.mean_squared_error(a[m], b[m])))


def safe_mae(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() == 0:
        return np.nan
    return float(metrics.mean_absolute_error(a[m], b[m]))

def safe_mape(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() == 0:
        return np.nan
    return float(metrics.mean_absolute_percentage_error(a[m], b[m]))
