#!/bin/bash
# Train the CNN letter model and use it for gesture recognition.
# Requires: Python 3.9–3.12 (TensorFlow does not support 3.14 yet).

set -e
cd "$(dirname "$0")"

echo "Checking Python..."
PYVER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || true)
if [[ "$PYVER" == "3.14" ]] || [[ "$PYVER" == "3.13" ]]; then
  echo "Your Python is $PYVER. TensorFlow needs 3.9–3.12."
  echo ""
  echo "Option 1: Create a venv with Python 3.12 (if installed):"
  echo "  python3.12 -m venv .venv_cnn"
  echo "  source .venv_cnn/bin/activate"
  echo "  pip install tensorflow"
  echo "  python sign_language_training/train_alphabet_cnn.py"
  echo ""
  echo "Option 2: Use MLP instead (no TensorFlow):"
  echo "  python3 sign_language_training/train_alphabet_model.py"
  exit 1
fi

echo "Installing TensorFlow if needed..."
pip install tensorflow -q 2>/dev/null || pip install tensorflow

echo "Training CNN letter model..."
python3 sign_language_training/train_alphabet_cnn.py

echo ""
echo "Done. Restart the app (python3 app.py) to use the CNN."
