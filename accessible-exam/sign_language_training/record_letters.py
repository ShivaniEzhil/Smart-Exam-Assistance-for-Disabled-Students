"""
Record letter (A–Z, 1–9) images from webcam for alphabet model training.
Saves to wlasl_alphabet_images/<Letter>/<id>.png. Run process_images.py after.

Run from accessible-exam:  python sign_language_training/record_letters.py

Tips for accurate letter recognition:
- Use good lighting, plain background
- Show one hand clearly to camera
- Hold each sign steady when capturing
- Record 30–50 samples per letter for best accuracy
"""
import argparse
import os
import sys

import cv2

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(SCRIPT_DIR, "wlasl_alphabet_images")
VALID_LETTERS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ123456789")


def main():
    parser = argparse.ArgumentParser(description="Record letter images for alphabet training")
    parser.add_argument("--out-dir", default=DATA_FOLDER, help="Output folder (default: wlasl_alphabet_images)")
    args = parser.parse_args()
    out_root = os.path.abspath(args.out_dir)
    os.makedirs(out_root, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Record letter images for A–Z, 1–9 finger-spelling.")
    print("  Enter letter (e.g. A, B, H, 1), hold sign, press SPACE to capture.")
    print("  Record 30–50 images per letter for best accuracy. Press Q to quit.")
    print("  Images saved to:", out_root)
    print()

    while True:
        letter = input("Letter to record (A–Z or 1–9, or Q to quit): ").strip().upper()
        if not letter:
            continue
        if letter == "Q":
            break
        if letter not in VALID_LETTERS:
            print(f"  Invalid. Use A–Z or 1–9.")
            continue

        letter_dir = os.path.join(out_root, letter)
        os.makedirs(letter_dir, exist_ok=True)

        print(f"  Show '{letter}' to camera. Press SPACE to capture. ESC when done.")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            existing = [f for f in os.listdir(letter_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            count = len(existing)
            cv2.putText(
                frame, f"Letter: {letter} | SPACE=capture, ESC=next letter",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
            )
            cv2.putText(
                frame, f"Captured: {count} for {letter}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1
            )
            cv2.imshow("Record letter", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                next_id = count + 1
                img_name = f"{next_id:03d}.png"
                save_path = os.path.join(letter_dir, img_name)
                cv2.imwrite(save_path, frame)
                print(f"  Saved {img_name} ({count + 1} total for {letter})")
            if key == 27:  # ESC
                break

        cv2.destroyWindow("Record letter")

    cap.release()
    print("Done. Next: python sign_language_training/process_images.py")
    print("Then: python sign_language_training/train_alphabet_model.py")


if __name__ == "__main__":
    main()
