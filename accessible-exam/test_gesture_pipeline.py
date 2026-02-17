"""
Quick test: load one frame (or create a dummy), run GestureModel.process_frame, print result.
Run from accessible-exam:  python test_gesture_pipeline.py
If you have a test image:  python test_gesture_pipeline.py path/to/image.jpg
"""
import sys
import os
import numpy as np
import cv2

os.chdir(os.path.dirname(os.path.abspath(__file__)))

def main():
    from gesture_model import GestureModel
    model = GestureModel()
    if not model.detector:
        print("FAIL: Hand Landmarker not loaded (check hand_landmarker.task)")
        return
    if not model.ml_model:
        print("FAIL: Alphabet model not loaded (check action.pkl)")
        return
    print("Gesture model loaded OK.")

    if len(sys.argv) > 1:
        path = sys.argv[1]
        frame = cv2.imread(path)
        if frame is None:
            print("Could not read image:", path)
            return
        print("Testing with image:", path)
    else:
        # Create a minimal test image (likely no hand -> None)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:] = (240, 220, 200)
        print("Testing with dummy image (no hand; expect None).")

    result = model.process_frame(frame)
    print("Result:", repr(result))
    if result:
        print("SUCCESS: Alphabet detection works.")
    else:
        print("No hand detected in this image. Try with a real photo of a hand.")

if __name__ == "__main__":
    main()
