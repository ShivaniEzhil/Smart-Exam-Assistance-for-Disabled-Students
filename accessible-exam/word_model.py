"""
Word-level sign recognition for long answers (WLASL-style).
"""

import os
import json
import pickle
import numpy as np
import cv2

SEQ_LEN = 30
FEAT_DIM = 63


class WordModel:
    def __init__(self):
        self.ml_model = None
        self.classes = []
        self.detector = None
        base_dir = os.path.dirname(os.path.abspath(__file__))
        pkl_path = os.path.join(base_dir, "word_model.pkl")
        json_path = os.path.join(base_dir, "word_classes.json")
        if os.path.isfile(pkl_path):
            try:
                with open(pkl_path, "rb") as f:
                    self.ml_model = pickle.load(f)
                if os.path.isfile(json_path):
                    with open(json_path, "r", encoding="utf-8") as f:
                        self.classes = json.load(f)
                print("Word model loaded (word_model.pkl)")
            except Exception as e:
                print("Error loading word model:", e)
        else:
            print("word_model.pkl not found. Run train_word_model.py after WLASL extraction.")
        try:
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
            mp_model_path = os.path.join(base_dir, "hand_landmarker.task")
            if not os.path.isfile(mp_model_path):
                return
            base_options = python.BaseOptions(
                model_asset_path=mp_model_path,
                delegate=python.BaseOptions.Delegate.CPU,
            )
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                num_hands=1,
                min_hand_detection_confidence=0.3,
            )
            self.detector = vision.HandLandmarker.create_from_options(options)
        except Exception as e:
            print("WordModel: Hand Landmarker init failed:", e)
            self.detector = None

    def set_detector(self, detector):
        """Use an external Hand Landmarker (e.g. from gesture model) when our own init failed."""
        if self.detector is None and detector is not None:
            self.detector = detector
            print("WordModel: using shared hand detector")

    def _frame_to_keypoints(self, frame_bgr):
        if self.detector is None:
            return None
        import mediapipe as mp
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self.detector.detect(mp_image)
        if not result.hand_landmarks:
            return None
        landmarks = result.hand_landmarks[0]
        return np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten().astype(np.float64)

    def process_sequence(self, frames):
        if not self.ml_model:
            return None
        if not self.detector:
            return None
        keypoints_list = []
        for frame in frames:
            kp = self._frame_to_keypoints(frame)
            if kp is not None and kp.size == FEAT_DIM:
                keypoints_list.append(kp)
        if len(keypoints_list) < 3:
            return None
        arr = np.array(keypoints_list, dtype=np.float64)
        if len(arr) >= SEQ_LEN:
            indices = np.linspace(0, len(arr) - 1, SEQ_LEN, dtype=int)
            seq = arr[indices]
        else:
            pad = np.zeros((SEQ_LEN - len(arr), FEAT_DIM), dtype=np.float64)
            seq = np.concatenate([arr, pad], axis=0)
        flat = seq.flatten().reshape(1, -1)
        pred = self.ml_model.predict(flat)[0]
        if hasattr(self.ml_model, "classes_") and isinstance(pred, (int, np.integer)):
            idx = int(pred)
            cls_list = self.classes if self.classes else getattr(self.ml_model, "classes_", [])
            if isinstance(cls_list, np.ndarray):
                cls_list = cls_list.tolist()
            if 0 <= idx < len(cls_list):
                return str(cls_list[idx])
        return str(pred) if pred is not None else None
