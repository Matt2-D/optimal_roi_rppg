



import os
import cv2
from tqdm import tqdm
# CropSense internals
import variable
from image_processing import process_image

# wrapper around CropSense
#accepts input_dir / output_dir
# removes prompts

# crop settings
DEFAULT_CROPTYPE         = "face"          # "face" | "upperbody" | "fullbody"
DEFAULT_TOP_MARGIN       = 0.2             # fractional top margin
DEFAULT_BOTTOM_MARGIN    = 0.2             # fractional bottom margin
DEFAULT_OUTPUT_RES       = variable.output_res
DEFAULT_PREVIEW_OUT_RES  = variable.preview_output_res
DEFAULT_PREVIEW_DBG_RES  = variable.preview_debug_max_res


def run_cropsense(
    input_dir: str, #folder with raw png's (attendant#/#/raw)
    output_dir: str, #destination folder for cropped frames (attendant#/#/cropped)
    debug_dir: str | None = None, #leave as none for now
    error_dir: str | None = None, #leave as none for now
    croptype: str = DEFAULT_CROPTYPE,
    top_margin: float = DEFAULT_TOP_MARGIN,
    bottom_margin: float = DEFAULT_BOTTOM_MARGIN,
    output_res: int = DEFAULT_OUTPUT_RES,
    parallel: bool = False,
) -> dict:  # Returns dict with keys: total, processed, faces_detected, errors

    # output directories
    base = os.path.dirname(output_dir)
    if debug_dir is None:
        debug_dir = os.path.join(base, "debug")
    if error_dir is None:
        error_dir = os.path.join(base, "error")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(debug_dir,  exist_ok=True)
    os.makedirs(error_dir,  exist_ok=True)

    # Collect input files
    supported = {".png"} #avoids csv's
    input_files = [
        os.path.join(input_dir, f)
        for f in sorted(os.listdir(input_dir))
        if os.path.splitext(f)[1].lower() in supported
    ]

    if not input_files:
        raise FileNotFoundError(f"No supported images found in: {input_dir}")

    print(f"[CropSense] {len(input_files)} images found in {input_dir}")
    print(f"[CropSense] Output → {output_dir}")

    # single-threaded processing
    if not parallel:
        error_count = 0
        progress = tqdm(input_files, desc="[CropSense] Cropping", dynamic_ncols=True)
        for image_path in progress:
            err = process_image(
                image_path,
                error_dir,
                output_dir,
                debug_dir,
                output_res,
                variable.preview_output_res,
                variable.preview_debug_max_res,
                show_preview=False,
                croptype=croptype,
                top_margin_value=top_margin,
                bottom_margin_value=bottom_margin,
            )
            if err:
                error_count += 1

        cv2.destroyAllWindows()

    # multi-threaded processing
    else:
        import sys
        import multiprocessing

        def _worker(args):
            return process_image(*args)

        worker_args = [
            (
                f, error_dir, output_dir, debug_dir,
                output_res,
                variable.preview_output_res,
                variable.preview_debug_max_res,
                False,
                croptype,
                top_margin,
                bottom_margin,
            )
            for f in input_files
        ]

        if sys.platform == "win32":
            multiprocessing.freeze_support()

        error_count = 0
        with multiprocessing.Pool() as pool:
            with tqdm(total=len(input_files),
                      desc="[CropSense] Cropping (parallel)",
                      dynamic_ncols=True) as pbar:
                for err in pool.imap_unordered(_worker, worker_args):
                    error_count += err
                    pbar.update(1)

        cv2.destroyAllWindows()

    # summary output
    total           = len(input_files)
    faces_detected  = len(os.listdir(output_dir))
    processed       = total - error_count

    print(f"[CropSense] Total frames   : {total}")
    print(f"[CropSense] Processed      : {processed}")
    print(f"[CropSense] Faces detected : {faces_detected}")
    print(f"[CropSense] Errors         : {error_count}")

    return {
        "total":           total,
        "processed":       processed,
        "faces_detected":  faces_detected,
        "errors":          error_count,
    }