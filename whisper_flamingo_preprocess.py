import dlib
import cv2
import os
import numpy as np
import skvideo.io
from tqdm import tqdm
import sys
import json # To parse ffmpeg output for FPS
import subprocess 

try: 
    from av_hubert.avhubert.preparation.align_mouth import landmarks_interpolate, crop_patch, write_video_ffmpeg
except ImportError as e:
    print(f"ImportError: {e}")
    # Import AV-HuBERT preprocessing helpers
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "av_hubert", "avhubert", "preparation"))
    from align_mouth import landmarks_interpolate, crop_patch, write_video_ffmpeg

def detect_landmark(image, detector, predictor):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    rects = detector(gray, 1)
    if len(rects) == 0:
        return None
    shape = predictor(gray, rects[0])
    coords = np.zeros((68, 2), dtype=np.int32)
    for i in range(68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

def get_video_fps(video_path):
    """
    Uses ffprobe to get the frames per second of a video file.
    """
    try:
        # Command to get stream information in JSON format
        cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=avg_frame_rate",
            "-of", "json",
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        
        # Parse the frame rate string (e.g., "60/1" or "25/1")
        frame_rate_str = data["streams"][0]["avg_frame_rate"]
        num, den = map(int, frame_rate_str.split('/'))
        
        return num / den
    except Exception as e:
        print(f"Error getting video FPS for {video_path}: {e}")
        print("Make sure ffprobe is installed and in your PATH.")
        return None # Indicate failure


def preprocess_video(input_path, output_path, face_predictor_path, mean_face_path):    
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(face_predictor_path)
    mean_face_landmarks = np.load(mean_face_path)
    STD_SIZE = (256, 256)
    stablePntsIDs = [33, 36, 39, 42, 45]
    crop_width = 96
    crop_height = 96
    window_margin = 12
    start_idx = 48
    stop_idx = 68


    print(f"[INFO] Reading video: {input_path}")
    frames = skvideo.io.vread(input_path) # Your existing frame read
    num_original_frames = len(frames)
    print(f"[INFO] Total frames: {num_original_frames}")

    # Get the original FPS of the input video
    input_fps = get_video_fps(input_path)
    if input_fps is None:
        print("Could not determine input video FPS. Exiting.")
        return # Or raise an error

    print(f"[INFO] Input video FPS: {input_fps}")

    detected_landmarks = []
    for frame in tqdm(frames, desc="Detecting landmarks"):
        landmarks = detect_landmark(frame, detector, predictor)
        # --- CRITICAL CHANGE HERE ---
        # If landmarks are None (no face detected), append None.
        # landmarks_interpolate is designed to handle None for interpolation.
        if landmarks is None:
            detected_landmarks.append(None)
        else:
            detected_landmarks.append(landmarks)


    # The DEBUG prints you added are now even more valuable here to confirm lengths
    print(f"[DEBUG] Length of raw detected_landmarks (should be num_original_frames): {len(detected_landmarks)}")
    print(f"[INFO] Interpolating landmarks...")
    preprocessed_landmarks = landmarks_interpolate(detected_landmarks)
    print(f"[DEBUG] Length of preprocessed_landmarks: {len(preprocessed_landmarks)}")

    # Ensure preprocessed_landmarks length matches num_original_frames before passing to crop_patch
    if len(preprocessed_landmarks) != num_original_frames:
        print(f"[CRITICAL WARNING] Interpolated landmarks count ({len(preprocessed_landmarks)}) does not match original frame count ({num_original_frames}). Adjusting.")
        # If landmarks_interpolate somehow messes up the length, force it.
        # This shouldn't be needed if landmarks_interpolate behaves as expected (which your debug showed it does now).
        if len(preprocessed_landmarks) > num_original_frames:
            preprocessed_landmarks = preprocessed_landmarks[:num_original_frames]
        else: # Pad with the last valid landmark if it's shorter (unlikely but for completeness)
            last_landmark = preprocessed_landmarks[-1]
            preprocessed_landmarks.extend([last_landmark] * (num_original_frames - len(preprocessed_landmarks)))
        print(f"[DEBUG] Adjusted preprocessed_landmarks length: {len(preprocessed_landmarks)}")


    print(f"[INFO] Cropping mouth ROIs and writing to: {output_path}")
    rois = crop_patch(
        frames, # Pass the loaded frames directly!
        preprocessed_landmarks,
        mean_face_landmarks,
        stablePntsIDs,
        STD_SIZE,
        window_margin,
        start_idx,
        stop_idx,
        crop_height,
        crop_width
    )
    print(f"[DEBUG] Number of ROIs (frames to be written): {len(rois)}")

    # ADD THESE NEW DEBUG CHECKS:
    if rois is None:
        print("[CRITICAL ERROR] crop_patch returned None. No ROIs generated.")
        return # Stop execution if no ROIs

    if len(rois) == 0:
        print("[CRITICAL ERROR] crop_patch returned an empty list of ROIs. No frames to write.")
        return # Stop execution if empty ROIs

    # Inspect the first ROI if it exists
    if len(rois) > 0:
        print(f"[DEBUG] Shape of first ROI: {rois[0].shape}, dtype: {rois[0].dtype}")
        # Make sure it's 3D (height, width, channels) and uint8
        if len(rois[0].shape) != 3 or rois[0].dtype != np.uint8:
            print("[CRITICAL ERROR] ROIs are not in expected (H, W, C) uint8 format. This could cause ffmpeg issues.")

    # Final sanity check before writing
    if len(rois) != num_original_frames:
        print(f"[CRITICAL ERROR] Final ROI count ({len(rois)}) does not match original frame count ({num_original_frames}). Output video duration will be incorrect.")
        # If this STILL happens, the bug is deep within crop_patch's windowing logic or its return.
        # You might have to manually trim or pad `rois` here as a last resort:
        # if len(rois) > num_original_frames:
        #     rois = rois[:num_original_frames]
        # elif len(rois) < num_original_frames:
        #     # You'd need a strategy to pad missing frames, e.g., with black frames or last valid ROI
        #     # rois.extend([np.zeros_like(rois[0])] * (num_original_frames - len(rois)))
        pass # Let the error print if it still occurs

    write_video_ffmpeg(rois, output_path, "/usr/bin/ffmpeg", out_fps=input_fps)
    print(f"[DONE] Output written to {output_path}")
    

if __name__ == "__main__":
    # Ensure subprocess is imported for get_video_fps

    # --- Parameters ---
    input_video = "/media/kevin/WD8TB/Work Projects/STRONG_video_files/test/T18S2M_face_with_aligned_audio_trimmed.mp4"
    output_video = "/media/kevin/WD8TB/Work Projects/STRONG_video_files/test/T18S2M_face_lips_cropped.mp4"
    
    # Download these files if you don't have them
    # shape_predictor_68_face_landmarks.dat: https://github.com/davisking/dlib-models/blob/master/shape_predictor_68_face_landmarks.dat.bz2 (unzip it)
    # mean_face.npy: This should be provided by the AV-HuBERT project or generated
    face_predictor_path = "shape_predictor_68_face_landmarks.dat"
    mean_face_path = "mean_face.npy" # Ensure this file exists in your working directory

    # Verify paths for necessary dlib and mean_face files
    if not os.path.exists(face_predictor_path):
        print(f"Error: Dlib face predictor not found at {face_predictor_path}")
        print("Please download it from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 and extract it.")
        sys.exit(1)
    if not os.path.exists(mean_face_path):
        print(f"Error: Mean face file not found at {mean_face_path}")
        print("This file should be provided with the AV-HuBERT setup or generated.")
        print("You might find it in `av_hubert/avhubert/preparation/` or related data directories.")
        sys.exit(1)


    preprocess_video(input_video, output_video, face_predictor_path, mean_face_path)