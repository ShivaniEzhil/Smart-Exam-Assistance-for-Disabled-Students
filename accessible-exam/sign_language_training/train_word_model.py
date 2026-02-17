"""
Train word-level sign model from hand sequences.
Input: WLASL_Hand_Data or Custom_Hand_Data with <gloss>/<id>.npy (each shape SEQ_LEN x 63).
Output: word_model.pkl + word_classes.json in accessible-exam/ (for word_model.py).

Run after extract_wlasl_hand_sequences.py or extract_custom_word_sequences.py.
"""

import argparse
import json
import os
import pickle

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT, "WLASL_Hand_Data")
OUTPUT_PKL = os.path.join(PROJECT_ROOT, "word_model.pkl")
OUTPUT_JSON = os.path.join(PROJECT_ROOT, "word_classes.json")
SEQ_LEN = 30
FEAT_DIM = 63


def load_data(data_dir):
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}. Run extract script first.")
    X_list, y_list = [], []
    classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    if not classes:
        raise FileNotFoundError(f"No gloss folders in {data_dir}")
    for gloss in classes:
        gloss_path = os.path.join(data_dir, gloss)
        for fname in os.listdir(gloss_path):
            if not fname.endswith(".npy"):
                continue
            path = os.path.join(gloss_path, fname)
            arr = np.load(path)
            if arr.ndim == 2 and arr.shape[0] == SEQ_LEN and arr.shape[1] == FEAT_DIM:
                X_list.append(arr.flatten())
                y_list.append(gloss)
    return np.array(X_list, dtype=np.float64), np.array(y_list), np.array(classes)


def main():
    parser = argparse.ArgumentParser(description="Train word model from hand sequences")
    parser.add_argument("--data-dir", default=None, help="Data directory (default: WLASL_Hand_Data)")
    args = parser.parse_args()
    data_dir = os.path.join(PROJECT_ROOT, args.data_dir) if args.data_dir else DEFAULT_DATA_DIR

    print("Loading hand sequences from", data_dir, "...")
    X, y, class_names = load_data(data_dir)
    print(f"  Samples: {len(X)}, Classes: {len(class_names)}, Feature dim: {X.shape[1]}")

    # Stratify requires at least 2 samples per class; otherwise train on all data
    from collections import Counter
    min_per_class = min(Counter(y).values()) if len(y) else 0
    if min_per_class >= 2:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.15, random_state=42, stratify=y
        )
    else:
        print("  Note: Few samples per class — training on all data (add more videos for better accuracy)")
        X_train, y_train = X, y
        X_test, y_test = np.array([]), np.array([])

    print("Training MLPClassifier (word-level)...")
    model = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        max_iter=400,
        activation="relu",
        solver="adam",
        random_state=42,
    )
    model.fit(X_train, y_train)
    if len(y_test) > 0:
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"  Accuracy: {acc*100:.2f}%")
        print(classification_report(y_test, y_pred, zero_division=0))
    else:
        print("  (No test split — train accuracy on all samples)")

    with open(OUTPUT_PKL, "wb") as f:
        pickle.dump(model, f)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(list(class_names), f)
    print(f"Saved {OUTPUT_PKL} and {OUTPUT_JSON}. Word model ready for deaf exam.")


if __name__ == "__main__":
    main()
