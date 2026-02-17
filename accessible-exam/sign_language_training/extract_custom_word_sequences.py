"""
Extract 63-dim hand landmark sequences from your own word video folders.
Input: directory with one folder per word, each containing .mp4 or .mov clips.
Output: WLASL_Hand_Data or Custom_Hand_Data with <word>/<clip_id>.npy (shape 30x63).

Run from accessible-exam:
  python sign_language_training/extract_custom_word_sequences.py --videos custom_word_videos [--output Custom_Hand_Data]
"""
import argparse
import os
import sys

import cv2
import numpy as np
import mediapipe as mp

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
SEQ_LEN = 30
FEAT_DIM = 21 * 3  # 63
VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv")


def get_hand_detector():
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
    try:
        base_options = python.BaseOptions(
            model_asset_path=model_path,
            delegate=python.BaseOptions.Delegate.CPU,
        )
    except Exception:
        base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.3,
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
    if len(frames_kp) >= seq_len:
        indices = np.linspace(0, len(frames_kp) - 1, seq_len, dtype=int)
        out = frames_kp[indices]
    else:
        pad = np.zeros((seq_len - len(frames_kp), feat_dim), dtype=np.float32)
        out = np.concatenate([frames_kp, pad], axis=0)
    return out


def main():
    parser = argparse.ArgumentParser(description="Extract hand sequences from custom word videos")
    parser.add_argument("--videos", required=True, help="Path to folder containing word subfolders (e.g. custom_word_videos)")
    parser.add_argument("--output", default="Custom_Hand_Data", help="Output directory (default: Custom_Hand_Data)")
    args = parser.parse_args()

    videos_dir = os.path.abspath(args.videos)
    if not os.path.isdir(videos_dir):
        print("Videos directory not found:", videos_dir)
        sys.exit(1)

    output_dir = os.path.join(PROJECT_ROOT, args.output)
    os.makedirs(output_dir, exist_ok=True)

    word_folders = sorted([
        d for d in os.listdir(videos_dir)
        if os.path.isdir(os.path.join(videos_dir, d)) and not d.startswith(".")
    ])
    if not word_folders:
        print("No word subfolders found in", videos_dir)
        sys.exit(1)

    tasks = []
    for word in word_folders:
        word_path = os.path.join(videos_dir, word)
        for fname in os.listdir(word_path):
            if fname.startswith("."):
                continue
            ext = os.path.splitext(fname)[1].lower()
            if ext not in VIDEO_EXTENSIONS:
                continue
            path = os.path.join(word_path, fname)
            clip_id = os.path.splitext(fname)[0]
            tasks.append((word, clip_id, path))

    print(f"Found {len(tasks)} clips in {len(word_folders)} words. Output: {output_dir}")
    detector = get_hand_detector()
    if detector is None:
        sys.exit(1)

    for word in word_folders:
        os.makedirs(os.path.join(output_dir, word), exist_ok=True)

    done, skip = 0, 0
    for word, clip_id, path in tasks:
        out_path = os.path.join(output_dir, word, clip_id + ".npy")
        if os.path.isfile(out_path):
            skip += 1
            continue
        seq = video_to_sequence(path, detector)
        if seq is not None:
            np.save(out_path, seq)
            done += 1
        if (done + skip) % 20 == 0 and (done + skip) > 0:
            print(f"  {done} saved, {skip} skipped")

    print(f"Done. Saved {done} sequences. Run: python sign_language_training/train_word_model.py --data-dir {args.output}")


if __name__ == "__main__":
    main()
