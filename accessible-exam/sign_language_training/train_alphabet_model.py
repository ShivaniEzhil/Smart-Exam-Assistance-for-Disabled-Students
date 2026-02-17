"""
Train the alphabet (finger-spelling) model for long answers.
Uses SINGLE FRAME (63 hand landmarks) per sample so the model matches
gesture_model.py at runtime (Hand Landmarker → 63 features → predict).

Data: MP_Data/<Class>/<sequence>/0.npy (and 1.npy ... 29.npy are duplicates).
We use one frame per sequence to get (n_samples, 63).

Improvements for accuracy:
- Data augmentation (Gaussian noise) to increase effective training data
- Deeper/wider MLP with more iterations
- L2 regularization to reduce overfitting
- Early stopping on validation set

Run from any directory; paths are relative to this script's project root (accessible-exam).
Output: action.pkl in accessible-exam/ (next to gesture_model.py).
"""

from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import os
import pickle
import argparse

# Project root = accessible-exam (parent of sign_language_training)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_PATH = os.path.join(PROJECT_ROOT, "MP_Data")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "action.pkl")
EXPECTED_FEATURES = 21 * 3  # 63

def _normalize(kp):
    """Same normalization as at inference (gesture_model.py)."""
    import sys
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    from landmark_utils import normalize_landmarks
    return normalize_landmarks(kp)


def augment_sample(kp, n_augment=4, noise_std=0.008):
    """Create augmented copies with small Gaussian noise. Returns list of (n_augment+1) arrays."""
    out = [kp]
    for _ in range(n_augment):
        noisy = kp + np.random.randn(kp.size).astype(np.float64) * noise_std
        out.append(noisy)
    return out


def load_single_frame_data(augment=True, augment_factor=4):
    """Load one frame per sequence from MP_Data. Optionally augment. Returns X, y, class_names."""
    if not os.path.isdir(DATA_PATH):
        raise FileNotFoundError(f"MP_Data not found at {DATA_PATH}. Run process_images.py first.")

    classes = sorted([d for d in os.listdir(DATA_PATH)
                      if os.path.isdir(os.path.join(DATA_PATH, d))])
    if not classes:
        raise FileNotFoundError(f"No class folders in {DATA_PATH}.")

    X_list, y_list = [], []

    for cls in classes:
        class_path = os.path.join(DATA_PATH, cls)
        seq_dirs = [d for d in os.listdir(class_path)
                    if os.path.isdir(os.path.join(class_path, d)) and d.isdigit()]
        for seq_id in sorted(seq_dirs, key=lambda x: int(x) if x.isdigit() else 0):
            frame_path = os.path.join(class_path, seq_id, "0.npy")
            if not os.path.isfile(frame_path):
                continue
            arr = np.load(frame_path)
            if arr.size != EXPECTED_FEATURES:
                continue
            if arr.ndim > 1:
                arr = arr.flatten()
            arr = _normalize(arr)
            if augment:
                for aug in augment_sample(arr, n_augment=augment_factor):
                    X_list.append(_normalize(aug))
                    y_list.append(cls)
            else:
                X_list.append(arr)
                y_list.append(cls)

    X = np.array(X_list, dtype=np.float64)
    y = np.array(y_list)
    return X, y, np.array(classes)


def main():
    parser = argparse.ArgumentParser(description="Train letter (A–Z, 1–9) model for finger-spelling")
    parser.add_argument("--no-augment", action="store_true", help="Disable data augmentation")
    parser.add_argument("--augment-factor", type=int, default=5, help="Augmented copies per sample (default 5)")
    args = parser.parse_args()

    print("Loading alphabet data (one frame per sequence, 63 features)...")
    X, y, class_names = load_single_frame_data(
        augment=not args.no_augment,
        augment_factor=args.augment_factor
    )
    print(f"  Samples: {len(X)}, Classes: {list(class_names)}, Feature dim: {X.shape[1]}")

    if X.shape[1] != EXPECTED_FEATURES:
        print(f"  ERROR: Expected {EXPECTED_FEATURES} features (hand only). Got {X.shape[1]}. Use process_images.py data.")
        return

    # Stratify requires at least 2 samples per class
    min_per_class = min(Counter(y).values()) if len(y) else 0
    if min_per_class >= 2:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.15, random_state=42, stratify=y
        )
    else:
        print("  Note: Few samples per class — using 80/20 split without stratification")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    print("Training MLPClassifier (single-frame alphabet, optimized for accuracy)...")
    model = MLPClassifier(
        hidden_layer_sizes=(512, 256, 128),
        max_iter=2000,
        activation="relu",
        solver="adam",
        alpha=0.0005,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=50,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"  Test accuracy: {acc*100:.2f}%")
    print(classification_report(y_test, y_pred, zero_division=0))

    os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"Saved model to {OUTPUT_PATH}")
    print("Restart the app; gesture_model.py will load this for A–Z (and 1–9) finger-spelling.")
    print("Tip: Add more images per letter in wlasl_alphabet_images/<Letter>/ for better accuracy.")


if __name__ == "__main__":
    main()
