"""
Train alphabet (finger-spelling) model using a 1D CNN on hand landmarks.
Uses the same MP_Data as train_alphabet_model.py but learns spatial patterns
across the 21 landmarks (shape 21 x 3) for better letter prediction.

Output: action_cnn.keras + action_classes.pkl in accessible-exam/
gesture_model.py will load these when available (fallback: action.pkl).

Run from accessible-exam:  python sign_language_training/train_alphabet_cnn.py
"""

from collections import Counter
import numpy as np
import os
import pickle
import argparse

# Project root = accessible-exam
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_PATH = os.path.join(PROJECT_ROOT, "MP_Data")
MODEL_PATH = os.path.join(PROJECT_ROOT, "action_cnn.keras")
CLASSES_PATH = os.path.join(PROJECT_ROOT, "action_classes.pkl")
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
    """Load one frame per sequence from MP_Data. Returns X (n, 21, 3), y, class_names."""
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
            # Reshape to (21, 3) for CNN
            arr_2d = arr.reshape(21, 3).astype(np.float32)
            if augment:
                for aug in augment_sample(arr, n_augment=augment_factor):
                    aug = _normalize(aug)
                    aug_2d = aug.reshape(21, 3).astype(np.float32)
                    X_list.append(aug_2d)
                    y_list.append(cls)
            else:
                X_list.append(arr_2d)
                y_list.append(cls)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list)
    return X, y, np.array(classes)


def build_cnn_model(num_classes, input_shape=(21, 3)):
    """Build 1D CNN: convolutions over the 21 landmarks, 3 channels (x,y,z)."""
    try:
        from tensorflow import keras
        from tensorflow.keras import layers
    except ImportError:
        raise ImportError("TensorFlow is required. Install with: pip install tensorflow")

    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv1D(64, 3, activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.Dropout(0.25),
        layers.Conv1D(128, 3, activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Dropout(0.25),
        layers.Conv1D(128, 3, activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling1D(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    parser = argparse.ArgumentParser(description="Train letter model with 1D CNN")
    parser.add_argument("--no-augment", action="store_true", help="Disable data augmentation")
    parser.add_argument("--augment-factor", type=int, default=5, help="Augmented copies per sample")
    parser.add_argument("--epochs", type=int, default=120, help="Max training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    args = parser.parse_args()

    print("Loading alphabet data (one frame per sequence, reshape to 21x3)...")
    X, y, class_names = load_single_frame_data(
        augment=not args.no_augment,
        augment_factor=args.augment_factor,
    )
    print(f"  Samples: {len(X)}, Classes: {list(class_names)}, Shape: {X.shape}")

    if X.shape[1] != 21 or X.shape[2] != 3:
        print("  ERROR: Expected shape (n, 21, 3). Use process_images.py data.")
        return

    # Class indices
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    y_idx = np.array([class_to_idx[c] for c in y], dtype=np.int32)

    # Train/validation split
    from sklearn.model_selection import train_test_split
    min_per_class = min(Counter(y).values()) if len(y) else 0
    if min_per_class >= 2:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y_idx, test_size=0.15, random_state=42,
            stratify=y_idx,
        )
    else:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y_idx, test_size=0.2, random_state=42,
        )

    print("Building 1D CNN model...")
    model = build_cnn_model(num_classes=len(class_names))

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=25,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            verbose=1,
        ),
    ]
    try:
        from tensorflow import keras as k
        callbacks.append(k.callbacks.TerminateOnNaN())
    except Exception:
        pass

    print("Training CNN...")
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    # Evaluate
    loss, acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"  Validation accuracy: {acc*100:.2f}%")

    os.makedirs(os.path.dirname(MODEL_PATH) or ".", exist_ok=True)
    model.save(MODEL_PATH)
    with open(CLASSES_PATH, "wb") as f:
        pickle.dump(class_names.tolist(), f)
    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved class names to {CLASSES_PATH}")
    print("Restart the app; gesture_model.py will use the CNN for letter prediction.")
    print("Tip: Add more images per letter in wlasl_alphabet_images/<Letter>/ for better accuracy.")


if __name__ == "__main__":
    main()
