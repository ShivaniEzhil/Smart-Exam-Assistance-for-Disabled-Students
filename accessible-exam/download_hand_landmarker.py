#!/usr/bin/env python3
"""
Download MediaPipe hand_landmarker.task into the accessible-exam directory.
Required for sign language (gesture) recognition. Run once before using the deaf exam.

Usage: python3 download_hand_landmarker.py
"""
import os
import sys
import urllib.request

# Official MediaPipe model URL
HAND_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "hand_landmarker.task")


def main():
    if os.path.isfile(OUTPUT_PATH):
        print(f"Already exists: {OUTPUT_PATH}")
        print("Delete it first if you want to re-download.")
        return 0
    print(f"Downloading hand_landmarker.task to {OUTPUT_PATH} ...")
    try:
        urllib.request.urlretrieve(HAND_LANDMARKER_URL, OUTPUT_PATH)
        print("Done. Restart the app for sign language to work.")
        return 0
    except Exception as e:
        print(f"Download failed: {e}", file=sys.stderr)
        print("Download manually from:", HAND_LANDMARKER_URL, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
