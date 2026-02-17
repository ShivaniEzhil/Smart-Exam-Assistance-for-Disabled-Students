import cv2
import mediapipe as mp
import numpy as np
import pickle
import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from landmark_utils import normalize_landmarks


def _load_cnn_model(dir_path):
    """Load CNN model and class names if present. Returns wrapper with predict/predict_proba/classes_ or None."""
    cnn_path = os.path.join(dir_path, "action_cnn.keras")
    classes_path = os.path.join(dir_path, "action_classes.pkl")
    if not os.path.exists(cnn_path) or not os.path.exists(classes_path):
        return None
    try:
        from tensorflow import keras
        model = keras.models.load_model(cnn_path)
        with open(classes_path, "rb") as f:
            classes = pickle.load(f)
        # Wrapper so gesture_model sees same interface as sklearn (predict, predict_proba, classes_)
        class CNNWrapper:
            def __init__(self, keras_model, class_names):
                self._model = keras_model
                self.classes_ = np.array(class_names)

            def predict(self, X):
                X = np.asarray(X, dtype=np.float64)
                if X.ndim == 1:
                    X = X.reshape(1, 21, 3)
                elif X.ndim == 2 and X.shape[1] == 63:
                    X = X.reshape(-1, 21, 3)
                probs = self._model.predict(X, verbose=0)
                indices = np.argmax(probs, axis=1)
                return np.array([self.classes_[i] for i in indices])

            def predict_proba(self, X):
                X = np.asarray(X, dtype=np.float64)
                if X.ndim == 1:
                    X = X.reshape(1, 21, 3)
                elif X.ndim == 2 and X.shape[1] == 63:
                    X = X.reshape(-1, 21, 3)
                return self._model.predict(X, verbose=0)
        return CNNWrapper(model, classes)
    except Exception as e:
        print(f"Error loading CNN model: {e}")
        return None


class GestureModel:
    def __init__(self):
        dir_path = os.path.dirname(__file__)
        self.ml_model = None
        # Prefer CNN if available (better accuracy from spatial patterns)
        cnn = _load_cnn_model(dir_path)
        if cnn is not None:
            self.ml_model = cnn
            print("Loaded CNN letter model: action_cnn.keras")
        else:
            self.model_path = os.path.join(dir_path, "action.pkl")
            if os.path.exists(self.model_path):
                try:
                    with open(self.model_path, 'rb') as f:
                        self.ml_model = pickle.load(f)
                    print("Loaded custom ML model: action.pkl")
                except Exception as e:
                    print(f"Error loading ML model: {e}")
            else:
                print("Warning: action.pkl not found. Run training first (train_alphabet_model.py or train_alphabet_cnn.py).")

        self.mp_model_path = os.path.join(os.path.dirname(__file__), 'hand_landmarker.task')
        self.detector = None
        if not os.path.exists(self.mp_model_path):
            print(f"Error: MediaPipe model {self.mp_model_path} not found.")
            return
        try:
            base_options = python.BaseOptions(
                model_asset_path=self.mp_model_path,
                delegate=python.BaseOptions.Delegate.CPU,
            )
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                num_hands=1,
                min_hand_detection_confidence=0.3,
            )
            self.detector = vision.HandLandmarker.create_from_options(options)
            print("MediaPipe HandLandmarker initialized (CPU).")
        except Exception as e:
            print(f"Failed to initialize MediaPipe: {e}")
            print("Tip: On macOS, run the app in a normal desktop session (not over SSH). If the error persists, see README_GESTURE_TROUBLESHOOTING.md")

    def process_frame(self, frame):
        if not self.detector:
            return None
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            detection_result = self.detector.detect(mp_image)
            if not detection_result.hand_landmarks:
                return None
            landmarks = detection_result.hand_landmarks[0]
            keypoints = np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten().astype(np.float64)
            if keypoints.size != 63:
                return None
            keypoints = normalize_landmarks(keypoints)
            if not self.ml_model:
                return "Model Not Loaded"
            prediction = self.ml_model.predict([keypoints])
            action = prediction[0]
            # Require reasonable confidence to reduce wrong letters (was 0.12)
            if hasattr(self.ml_model, 'predict_proba'):
                probs = self.ml_model.predict_proba([keypoints])[0]
                max_prob = float(np.max(probs))
                if max_prob < 0.35:
                    return None
            if hasattr(self.ml_model, 'classes_') and isinstance(action, (int, np.integer)):
                idx = int(action)
                if 0 <= idx < len(self.ml_model.classes_):
                    return str(self.ml_model.classes_[idx])
            return str(action)
        except Exception as e:
            print(f"Error processing frame: {e}")
            return None

    def process_frames_vote(self, frames, min_votes=2):
        """Process multiple frames; return majority prediction with optional confidence weighting."""
        if not frames:
            return None
        from collections import Counter
        votes = []
        confs = []
        for f in frames:
            letter, conf = self.process_frame_with_confidence(f)
            if letter and letter != "Model Not Loaded":
                votes.append(letter)
                confs.append(conf)
        if not votes:
            return None
        # Prefer prediction with highest total confidence when tied
        counter = Counter(votes)
        best_letter, count = counter.most_common(1)[0]
        if count < min_votes:
            return None
        # If we have confidence, among ties pick the one with higher total confidence
        if confs and hasattr(self.ml_model, 'predict_proba'):
            tied = [l for l, c in counter.most_common() if c == count]
            if len(tied) > 1:
                sum_conf = {}
                for i, letter in enumerate(votes):
                    sum_conf[letter] = sum_conf.get(letter, 0) + confs[i]
                best_letter = max(tied, key=lambda l: sum_conf.get(l, 0))
        return best_letter

    def process_frame_with_confidence(self, frame):
        """Return (letter, confidence) or (None, 0). Used for debugging."""
        if not self.detector or not self.ml_model:
            return None, 0.0
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            detection_result = self.detector.detect(mp_image)
            if not detection_result.hand_landmarks:
                return None, 0.0
            landmarks = detection_result.hand_landmarks[0]
            keypoints = np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten().astype(np.float64)
            if keypoints.size != 63:
                return None, 0.0
            keypoints = normalize_landmarks(keypoints)
            pred = self.ml_model.predict([keypoints])[0]
            conf = 0.0
            if hasattr(self.ml_model, 'predict_proba'):
                probs = self.ml_model.predict_proba([keypoints])[0]
                conf = float(np.max(probs))
            letter = None
            if hasattr(self.ml_model, 'classes_') and isinstance(pred, (int, np.integer)):
                idx = int(pred)
                if 0 <= idx < len(self.ml_model.classes_):
                    letter = str(self.ml_model.classes_[idx])
            else:
                letter = str(pred) if pred is not None else None
            return letter, conf
        except Exception as e:
            print(f"Error: {e}")
            return None, 0.0
