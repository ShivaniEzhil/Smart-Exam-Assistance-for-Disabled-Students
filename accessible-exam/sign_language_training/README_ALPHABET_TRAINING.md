# Train Letter (A–Z, 1–9) Recognition Accurately

Letter detection often fails because the model was trained on limited or mismatched data. **Retraining with a good dataset** fixes this.

## 1. Get Training Data

### Option A: Download proven ASL dataset (best accuracy)

```bash
pip install kaggle
# Get API key from https://www.kaggle.com/settings -> Create New Token
# Put kaggle.json in ~/.kaggle/

cd accessible-exam
python sign_language_training/download_asl_dataset.py
```

Or manual download: `python sign_language_training/download_asl_dataset.py --manual`

### Option B: Record with your webcam

```bash
cd accessible-exam
python sign_language_training/record_letters.py
```

- Enter a letter (A, B, H, 1, etc.), show the sign, press **SPACE** to capture
- Record **30–50 images per letter** for best accuracy
- Use good lighting and a plain background
- Press ESC when done with a letter, Q to quit

Images are saved to `wlasl_alphabet_images/<Letter>/`.

### Option C: Use existing images

Place images in `sign_language_training/wlasl_alphabet_images/<Class>/`:
- `A/`, `B/`, … `Z/` for letters
- `1/`, `2/`, … `9/` for digits
- Use `.png`, `.jpg`, or `.jpeg`

## 2. Extract Hand Keypoints

```bash
cd accessible-exam
python sign_language_training/process_images.py
```

Use more images per class for better accuracy:

```bash
python sign_language_training/process_images.py --max-per-class 200
```

Output: `MP_Data/<Class>/<id>/0.npy` (63-dim hand landmarks).

## 3. Train the Model

### Option A: CNN (recommended for better accuracy)

Uses a 1D CNN on hand landmarks (21×3) to learn spatial patterns. Requires **TensorFlow** and **Python 3.9–3.12** (TensorFlow does not support 3.13+ yet).

```bash
cd accessible-exam
pip install tensorflow   # if not already installed (Python 3.9–3.12 only)
python sign_language_training/train_alphabet_cnn.py
```

Or run the helper script (checks Python version):

```bash
cd accessible-exam
bash run_cnn_training.sh
```

If you have Python 3.14, use a venv with 3.12: `python3.12 -m venv .venv_cnn` then `source .venv_cnn/bin/activate`, then the commands above.

This produces `action_cnn.keras` and `action_classes.pkl`. The app will use the CNN automatically if these files exist.

Optional: `--epochs 150`, `--batch-size 32`, `--no-augment` to disable augmentation.

### Option B: MLP (original)

```bash
cd accessible-exam
python sign_language_training/train_alphabet_model.py
```

**Training improvements (for accuracy):**
- **Data augmentation**: 5× synthetic copies per sample (Gaussian noise)
- **Deeper model**: (512, 256, 128) hidden layers
- **Early stopping**: Prevents overfitting
- **L2 regularization**: Reduces overfitting

To disable augmentation (e.g. if you have lots of real data):

```bash
python sign_language_training/train_alphabet_model.py --no-augment
```

Produces `action.pkl`. Used by the app when no CNN model is present.

## 4. Restart the App

Restart Flask so it loads the new model (`action_cnn.keras` or `action.pkl`). The deaf exam will use the updated letter model.

## Tips for Best Accuracy

1. **More data per letter**: 50+ samples per class is ideal
2. **Variety**: Vary hand position, distance from camera, slight rotation
3. **Lighting**: Consistent, avoid strong shadows
4. **Background**: Plain works best
