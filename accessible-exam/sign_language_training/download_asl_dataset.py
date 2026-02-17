"""
Download ASL Alphabet dataset for better letter recognition.
Uses Kaggle API (recommended) or manual instructions.

The model needs 50-200+ images per letter (A-Z, 1-9) for accurate recognition.
This script helps you get a proven dataset.

Run from accessible-exam:
  pip install kaggle   # if not installed
  # Set up Kaggle API: https://www.kaggle.com/docs/api
  # Put kaggle.json in ~/.kaggle/
  python sign_language_training/download_asl_dataset.py
"""
import os
import sys
import shutil
import zipfile
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
OUTPUT_BASE = os.path.join(SCRIPT_DIR, "wlasl_alphabet_images")
# Kaggle dataset: asl alphabet - several options
KAGGLE_DATASET = "grassknoted/asl-alphabet"  # 87k images, A-Z + 3 extra


def run_kaggle_download():
    """Download via Kaggle API."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        download_dir = os.path.join(SCRIPT_DIR, "asl_download")
        os.makedirs(download_dir, exist_ok=True)
        print(f"Downloading {KAGGLE_DATASET}... (this may take a few minutes)")
        api.dataset_download_files(KAGGLE_DATASET, path=download_dir, unzip=True)
        return download_dir
    except Exception as e:
        print(f"Kaggle download failed: {e}")
        return None


def organize_dataset(download_dir):
    """Move images into wlasl_alphabet_images/<Letter>/ structure."""
    os.makedirs(OUTPUT_BASE, exist_ok=True)
    seen = set()
    max_per_letter = 150
    for root, dirs, files in os.walk(download_dir):
        for d in dirs:
            if len(d) == 1 and d.isalpha() and d.upper() not in seen:
                src = os.path.join(root, d)
                dst = os.path.join(OUTPUT_BASE, d.upper())
                if os.path.isdir(src):
                    imgs = [f for f in os.listdir(src) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                    if imgs:
                        seen.add(d.upper())
                        os.makedirs(dst, exist_ok=True)
                        for i, f in enumerate(imgs[:max_per_letter]):
                            shutil.copy2(os.path.join(src, f), os.path.join(dst, f))
                        print(f"  {d.upper()}: {min(len(imgs), max_per_letter)} images")
            elif len(d) == 1 and d.isdigit() and d not in seen:
                src = os.path.join(root, d)
                dst = os.path.join(OUTPUT_BASE, d)
                if os.path.isdir(src):
                    imgs = [f for f in os.listdir(src) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                    if imgs:
                        seen.add(d)
                        os.makedirs(dst, exist_ok=True)
                        for f in imgs[:100]:
                            shutil.copy2(os.path.join(src, f), os.path.join(dst, f))
                        print(f"  {d}: {min(len(imgs), 100)} images")


def main():
    parser = argparse.ArgumentParser(description="Download ASL alphabet dataset")
    parser.add_argument("--manual", action="store_true", help="Show manual download instructions only")
    args = parser.parse_args()

    if args.manual:
        print("""
MANUAL DOWNLOAD INSTRUCTIONS
===========================
1. Go to: https://www.kaggle.com/datasets/grassknoted/asl-alphabet
2. Click "Download" (requires free Kaggle account)
3. Unzip the downloaded file
4. Copy/link the train folder structure to:
   sign_language_training/wlasl_alphabet_images/

   Expected structure:
   wlasl_alphabet_images/
     A/   <- images of letter A
     B/
     ...
     Z/

5. Run:
   python sign_language_training/process_images.py
   python sign_language_training/train_alphabet_model.py
""")
        return

    print("Attempting Kaggle download...")
    dl_dir = run_kaggle_download()
    if dl_dir:
        print("Organizing into wlasl_alphabet_images/...")
        organize_dataset(dl_dir)
        print(f"\nDone! Images in {OUTPUT_BASE}")
        print("Next: python sign_language_training/process_images.py")
        print("Then: python sign_language_training/train_alphabet_model.py")
    else:
        print("\nKaggle API not set up. Options:")
        print("  pip install kaggle")
        print("  Get API key from https://www.kaggle.com/settings")
        print("  Put kaggle.json in ~/.kaggle/")
        print("\nOr run with --manual for manual download instructions.")


if __name__ == "__main__":
    main()
