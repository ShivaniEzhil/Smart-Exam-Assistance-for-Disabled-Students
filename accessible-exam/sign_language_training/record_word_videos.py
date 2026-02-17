"""
Record sign-language word videos from webcam for custom word model training.
Saves to custom_word_videos/<word>/<id>.mp4. No MediaPipe or Flask required.

Run from accessible-exam:  python sign_language_training/record_word_videos.py
"""
import argparse
import os
import sys
import time

import cv2

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DEFAULT_OUT = os.path.join(PROJECT_ROOT, "custom_word_videos")
RECORD_SEC = 2.5
FPS = 15


def main():
    parser = argparse.ArgumentParser(description="Record word videos for training")
    parser.add_argument("--out-dir", default=DEFAULT_OUT, help="Output directory (default: custom_word_videos)")
    args = parser.parse_args()
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, FPS)

    print("Record word videos for custom dataset.")
    print("  Enter a word label, then press SPACE to record 2.5s. Repeat. Press Q to quit.")
    print("  Videos saved under:", out_dir)
    print()

    while True:
        word = input("Word label (or Q to quit): ").strip()
        if not word:
            continue
        if word.upper() == "Q":
            break
        word_dir = os.path.join(out_dir, word)
        os.makedirs(word_dir, exist_ok=True)
        existing = [f for f in os.listdir(word_dir) if f.endswith(".mp4")]
        next_id = len(existing) + 1
        clip_name = f"{next_id:03d}.mp4"
        save_path = os.path.join(word_dir, clip_name)

        print(f"  Recording '{word}' -> {clip_name}. Press SPACE to start.")
        recording = False
        start_time = None
        frames = []
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if recording:
                frames.append(frame)
                elapsed = time.time() - start_time
                cv2.putText(
                    frame, f"Recording {elapsed:.1f}s / {RECORD_SEC}s",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
                )
                if elapsed >= RECORD_SEC:
                    break
            else:
                cv2.putText(
                    frame, "Press SPACE to start recording",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                )
            cv2.imshow("Record word", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                if not recording:
                    recording = True
                    start_time = time.time()
            if key == ord("q"):
                recording = False
                frames = []
                break

        if frames:
            h, w = frames[0].shape[:2]
            writer = cv2.VideoWriter(save_path, fourcc, FPS, (w, h))
            for f in frames:
                writer.write(f)
            writer.release()
            print(f"  Saved {save_path} ({len(frames)} frames)")
        cv2.destroyWindow("Record word")

    cap.release()
    print("Done. Run extract_custom_word_sequences.py next.")


if __name__ == "__main__":
    main()
