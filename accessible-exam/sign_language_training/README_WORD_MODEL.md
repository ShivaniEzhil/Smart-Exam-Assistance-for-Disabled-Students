# Word-level sign recognition (long answer) — WLASL integration

This adds **whole-word** sign recognition to the deaf/mute exam so students can sign words (e.g. "water", "help", "yes") instead of only spelling letter-by-letter.

**If the app only detects letters, not words:** the word model is not loaded. You need to train it once (see below). After `word_model.pkl` exists in `accessible-exam/`, restart the app and the deaf exam will show "Word + letter recognition enabled" and will detect both words and letters.

## Steps to enable the word model

### 1. Get WLASL videos

- In `sign_language_training/WLASL-master/start_kit/` run:
  - `python video_downloader.py` (requires `yt-dlp` and network)
  - `python preprocess.py`
- Clips will appear in `start_kit/videos/` as `<video_id>.mp4`.  
  If many links are broken, use the [WLASL request form](https://dxli94.github.io/WLASL/) to get missing/preprocessed videos.

### 2. Extract hand sequences

From the **accessible-exam** folder:

```bash
python sign_language_training/extract_wlasl_hand_sequences.py --max-glosses 100
```

Optional: `--videos path/to/videos` if your clips are elsewhere.  
Output: `WLASL_Hand_Data/<gloss>/<video_id>.npy` (each shape 30×63).

### 3. Train the word model

```bash
python sign_language_training/train_word_model.py
```

This creates `word_model.pkl` and `word_classes.json` in **accessible-exam/**.

### 4. Restart the app

Restart Flask so it loads the word model. In the deaf exam, after turning on the camera, use **"Record word (2.5s)"**: sign one word over 2.5 seconds, then the recognized word is inserted into your answer.

## Quick path (no WLASL word videos, use alphabet data only)

If you already have **MP_Data** (from process_images.py on WLASL-derived alphabet images in `wlasl_alphabet_images/`) and cannot run the WLASL video extractor (e.g. MediaPipe fails or no videos):

```bash
cd accessible-exam
python sign_language_training/build_wlasl_from_mpdata.py
python sign_language_training/train_word_model.py
```

Then restart the app. **Record word** will recognize letters (one per 2.5s clip), same vocabulary as Insert letter.

## Custom dataset (your own recordings)

Record your own sign-language word videos and train the word model on them for better recognition of your vocabulary.

### Step 1: Record word clips

From the **accessible-exam** folder:

```bash
python sign_language_training/record_word_videos.py
```

- Enter a **word label** (e.g. `water`, `help`, `yes`) and press Enter.
- Press **Space** to start recording; the script records for 2.5 seconds and saves automatically.
- Repeat for the same word (e.g. 20–30 clips per word) or enter another word. Press **Q** to quit.

Videos are saved under **`custom_word_videos/<word>/001.mp4`**, `002.mp4`, etc. Record multiple clips per word for better training.

Optional: `--out-dir path/to/folder` to use a different output directory.

### Step 2: Extract hand sequences

From **accessible-exam**:

```bash
python sign_language_training/extract_custom_word_sequences.py --videos custom_word_videos --output Custom_Hand_Data
```

- **`--videos`**: folder containing your word subfolders (default layout: `custom_word_videos/water/`, `custom_word_videos/help/`, …).
- **`--output`**: where to write `.npy` sequences (default: `Custom_Hand_Data`). Each file has shape 30×63.

Requires **`hand_landmarker.task`** in `accessible-exam/`. If MediaPipe fails (e.g. GPU/OpenGL error), run this step on another machine and copy the `Custom_Hand_Data` folder back.

### Step 3: Train the word model

```bash
python sign_language_training/train_word_model.py --data-dir Custom_Hand_Data
```

This creates **`word_model.pkl`** and **`word_classes.json`** in **accessible-exam/** using only your custom data.

### Step 4: Restart the app

Restart Flask so it loads the new word model. **Record word (2.5s)** in the deaf exam will then recognize the words you recorded and trained on.

---

## If you skip everything

The exam still works: **Insert letter** (A–Z, 1–9) and typing are always available. **Record word** will return "Word model not loaded" until you complete one of the pipelines above.
