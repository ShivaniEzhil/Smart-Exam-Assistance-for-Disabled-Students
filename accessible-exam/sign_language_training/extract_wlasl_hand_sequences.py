"""
Extract 63-dim hand landmark sequences from WLASL video clips.
Output: WLASL_Hand_Data/<gloss>/<video_id>.npy with shape (SEQ_LEN, 63).

Prerequisites:
  1. WLASL videos in start_kit/videos/ (run video_downloader.py + preprocess.py first).
  2. hand_landmarker.task in accessible-exam/ or sign_language_training/.

Usage (from accessible-exam or repo root):
  python sign_language_training/extract_wlasl_hand_sequences.py [--videos path] [--max-glosses 100]
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np
import mediapipe as mp

# Project root = accessible-exam
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
WLASL_MASTER = os.path.join(SCRIPT_DIR, "WLASL-master")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "WLASL_Hand_Data")
SEQ_LEN = 30
FEAT_DIM = 21 * 3  # 63


def get_hand_detector():
    """Lazy MediaPipe Hand Landmarker (same as gesture_model)."""
    try:
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
    except ImportError:
        return None
    model_path = os.path.join(PROJECT_ROOT, "hand_landmarker.task")
    if not os.path.isfile(model_path):
        model_path = os.path.join(SCRIPT_DIR, "hand_landmarker.task")
    if not os.path.isfile(model_path):
        print("hand_landmarker.task not found in", PROJECT_ROOT, "or", SCRIPT_DIR)
        return None
    base_options = python.BaseOptions(
        model_asset_path=model_path,
        delegate=python.BaseOptions.Delegate.CPU,
    )
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(options)


def extract_keypoints(detector, frame_rgb):
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result = detector.detect(mp_image)
    if not result.hand_landmarks:
        return None
    landmarks = result.hand_landmarks[0]
    return np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten().astype(np.float32)


def video_to_sequence(video_path, detector, seq_len=SEQ_LEN, feat_dim=FEAT_DIM):
    """Read video, run hand detector per frame, return (seq_len, feat_dim) or None."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    frames_kp = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        kp = extract_keypoints(detector, frame_rgb)
        if kp is not None and kp.size == feat_dim:
            frames_kp.append(kp)
    cap.release()
    if len(frames_kp) < 3:
        return None
    frames_kp = np.array(frames_kp, dtype=np.float32)
    # Trim or pad to seq_len
    if len(frames_kp) >= seq_len:
        # Take evenly spaced frames
        indices = np.linspace(0, len(frames_kp) - 1, seq_len, dtype=int)
        out = frames_kp[indices]
    else:
        pad = np.zeros((seq_len - len(frames_kp), feat_dim), dtype=np.float32)
        out = np.concatenate([frames_kp, pad], axis=0)
    return out


def main():
    parser = argparse.ArgumentParser(description="Extract hand sequences from WLASL videos")
    parser.add_argument("--videos", default=None, help="Path to folder of video clips (default: WLASL-master/start_kit/videos)")
    parser.add_argument("--max-glosses", type=int, default=100, help="Max number of glosses to process (default 100)")
    parser.add_argument("--json", default=None, help="Path to WLASL_v0.3.json")
    args = parser.parse_args()

    videos_dir = args.videos or os.path.join(WLASL_MASTER, "start_kit", "videos")
    json_path = args.json or os.path.join(WLASL_MASTER, "start_kit", "WLASL_v0.3.json")

    if not os.path.isfile(json_path):
        print("WLASL JSON not found:", json_path)
        sys.exit(1)
    if not os.path.isdir(videos_dir):
        print("Videos dir not found:", videos_dir)
        print("Run WLASL start_kit/video_downloader.py and preprocess.py first.")
        sys.exit(1)

    with open(json_path, "r", encoding="utf-8") as f:
        content = json.load(f)

    # Build list (gloss, video_id) for instances we have video for
    tasks = []
    gloss_counts = {}
    for entry in content:
        gloss = entry["gloss"]
        for inst in entry["instances"]:
            vid = str(inst["video_id"])
            path = os.path.join(videos_dir, vid + ".mp4")
            if os.path.isfile(path):
                tasks.append((gloss, vid, path))
                gloss_counts[gloss] = gloss_counts.get(gloss, 0) + 1

    # Restrict to top max_glosses by count
    sorted_glosses = sorted(gloss_counts.keys(), key=lambda g: -gloss_counts[g])
    allowed = set(sorted_glosses[: args.max_glosses])
    tasks = [(g, v, p) for g, v, p in tasks if g in allowed]

    print(f"Processing {len(tasks)} clips for {len(allowed)} glosses. Output: {OUTPUT_DIR}")
    detector = get_hand_detector()
    if detector is None:
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for gloss in allowed:
        os.makedirs(os.path.join(OUTPUT_DIR, gloss), exist_ok=True)

    done, skip = 0, 0
    for gloss, vid, path in tasks:
        out_path = os.path.join(OUTPUT_DIR, gloss, vid + ".npy")
        if os.path.isfile(out_path):
            skip += 1
            continue
        seq = video_to_sequence(path, detector)
        if seq is not None:
            np.save(out_path, seq)
            done += 1
        if (done + skip) % 100 == 0:
            print(f"  {done} saved, {skip} skipped (cached)")

    print(f"Done. Saved {done} sequences. Run train_word_model.py next.")


if __name__ == "__main__":
    main()
