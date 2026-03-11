
    #1. main_vid2rgb.py      — face crop + extract per-ROI RGB signals
    #2. main_weightedROI.py  — append ROI 29 (SNR-weighted) + ROI 30 (simple avg)
    ##3. main_rgb2hr.py       — RGB → BVP + BPM per ROI per algorithm
    #4. main_gen_gtHR.py     — generate ground truth BPM/BVP from Arduino CSV
    #5. main_evaluation.py   — evaluate all algorithms against ground truth

# Author: Matthew Dowell
# Date: 03/10/2026

import os
import sys
import traceback
import time

dir_crt = os.getcwd()
sys.path.append(os.path.join(dir_crt, 'util'))

# ── Stage toggles — set False to skip a completed stage ──────────────────────
RUN_VID2RGB     = True
RUN_WEIGHTED_ROI = True
RUN_RGB2HR      = True
RUN_GEN_GTHR    = True
RUN_EVALUATION  = True

NAME_DATASET  = 'custom'
ALGORITHMS    = ['CHROM', 'LGI', 'OMIT', 'POS']
ATTENDANT_ID  = 1
DISTANCES     = [1, 2, 3]
FPS           = 50.0


def run_stage(stage_name: str, fn, *args, **kwargs):
    """Run a pipeline stage with timing and error reporting."""
    print(f"\n{'#'*60}")
    print(f"#  STAGE: {stage_name}")
    print(f"{'#'*60}")
    t0 = time.time()
    try:
        fn(*args, **kwargs)
        elapsed = time.time() - t0
        print(f"\n[OK] {stage_name} completed in {elapsed:.1f}s")
    except Exception as e:
        print(f"\n[ERROR] {stage_name} failed:")
        traceback.print_exc()
        print(f"\nPipeline halted at: {stage_name}")
        sys.exit(1)

def stage_vid2rgb():
    from main_vid2rgb import main_vid2rgb
    main_vid2rgb(name_dataset=NAME_DATASET)


def stage_weighted_roi():
    from main_weightedROI import main_combine_rois
    main_combine_rois(
        name_dataset=NAME_DATASET,
        attendant_id=ATTENDANT_ID,
        distances=DISTANCES,
        fps=FPS,
    )


def stage_rgb2hr():
    from main_rgb2hr import main_rgb2hr
    for algorithm in ALGORITHMS:
        print(f"\n  Algorithm: {algorithm}")
        main_rgb2hr(name_dataset=NAME_DATASET, algorithm=algorithm)


def stage_gen_gthr():
    from main_gen_gtHR import main_gen_gtHR
    dir_dataset = os.path.join(dir_crt, 'data', NAME_DATASET)
    main_gen_gtHR(dir_dataset=dir_dataset,name_dataset=NAME_DATASET)


def stage_evaluation():
    from main_evaluation import main_eval
    for algorithm in ALGORITHMS:
        print(f"\n  Algorithm: {algorithm}")
        main_eval(name_dataset=NAME_DATASET, algorithm=algorithm)


if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"  rPPG PIPELINE  —  dataset: {NAME_DATASET}")
    print(f"  Algorithms : {ALGORITHMS}")
    print(f"  Distances  : {DISTANCES}")
    print(f"  Attendant  : {ATTENDANT_ID}")
    print(f"{'='*60}")

    pipeline_start = time.time()

    if RUN_VID2RGB:
        run_stage("1 — vid2rgb (crop + RGB extraction)", stage_vid2rgb)

    if RUN_WEIGHTED_ROI:
        run_stage("2 — weighted ROI (append ROI 29 & 30)", stage_weighted_roi)

    if RUN_RGB2HR:
        run_stage("3 — rgb2hr (RGB → BVP/BPM)", stage_rgb2hr)

    if RUN_GEN_GTHR:
        run_stage("4 — gen_gtHR (ground truth BPM)", stage_gen_gthr)

    if RUN_EVALUATION:
        run_stage("5 — evaluation (metrics vs ground truth)", stage_evaluation)

    total = time.time() - pipeline_start
    print(f"\n{'='*60}")
    print(f"  PIPELINE COMPLETE  —  total time: {total:.1f}s")
    print(f"  Results → result/{NAME_DATASET}/")
    print(f"{'='*60}\n")