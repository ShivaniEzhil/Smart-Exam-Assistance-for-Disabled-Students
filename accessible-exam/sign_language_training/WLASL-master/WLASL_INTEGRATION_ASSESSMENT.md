# Will WLASL Help for Words? — Assessment

**Short answer: Yes.** WLASL is a **word-level** American Sign Language dataset (2,000 words). It is exactly the kind of resource that can help you add **whole-word** recognition to your exam app, so students can sign words instead of only spelling letter-by-letter.

---

## What WLASL Gives You

| Item | Description |
|------|-------------|
| **WLASL_v0.3.json** | Metadata for ~2,000 glosses (words): `gloss`, `video_id`, `url`, `frame_start`, `frame_end`, `split` (train/val/test). |
| **Videos** | Not included in the repo. You must **download** them using `start_kit/video_downloader.py`, then run `preprocess.py` to get clips under `videos/`. |
| **Subsets** | WLASL100, WLASL300, WLASL1000, WLASL2000 (top-K most frequent words). Good to start with **WLASL100** (e.g. book, drink, computer, help, yes, no, water, etc.). |
| **Training code** | **I3D**: video CNN (raw video frames). **TGCN**: graph network on **pre-extracted pose/keypoints** (body + hands in OpenPose-style JSON). |

So: WLASL is **word labels + video clips** (or keypoints derived from them). That is what you need for word recognition.

---

## How It Can Help Your App

- **Current app:** Single-frame **hand** landmarks (63 dims) → alphabet model → **letters/digits** (finger-spelling).
- **With WLASL:** **Video sequence** (or sequence of keypoints) → word model → **whole words** (e.g. "water", "help", "yes").

So yes — WLASL **will help for words** if you add a **sequence model** that takes a short video (or keypoint sequence) and outputs a word label.

---

## What’s Different From Your Current Pipeline

| Your current pipeline | WLASL / their code |
|-----------------------|--------------------|
| **Input** | Single frame → 63 hand keypoints (MediaPipe Hand Landmarker). |
| **Input** | Full **video clip** (I3D) or **sequence of body+hand keypoints** (TGCN, OpenPose-style). |
| **Output** | One label per frame (letter/digit). |
| **Output** | One label per **clip** (word). |
| **Run where** | Your Flask backend: decode image → hand landmarks → `action.pkl` → letter. |
| **Run where** | Need to send a **short video** (or keypoint sequence) to the backend, then run I3D or TGCN. |

So you have two main integration options.

---

## Option A: Use WLASL Videos + Your Own Hand Features (Recommended for your stack)

1. **Get WLASL videos**  
   - Run `start_kit/video_downloader.py` (needs `yt-dlp`), then `preprocess.py`.  
   - Use a subset (e.g. WLASL100) so you have a manageable word list.

2. **Extract features that match your app**  
   - For each video clip, run **MediaPipe Hand Landmarker** frame-by-frame (same as in `gesture_model.py`).  
   - Save a **sequence** of 63-dim vectors per clip (e.g. pad/trim to fixed length, e.g. 30–60 frames).

3. **Train a sequence model**  
   - Input: `(batch, num_frames, 63)` (or flattened per clip).  
   - Use an **LSTM** or **1D CNN** (or small Transformer) in PyTorch/TensorFlow.  
   - Output: word class (one of WLASL100, etc.).  
   - This keeps your **runtime** the same: capture a short buffer of frames in the app, extract 63-dim hand features, run your sequence model.

4. **Integrate in the app**  
   - In the exam UI, add a “Record word” (e.g. 2–3 seconds) that sends a short video or a sequence of frames.  
   - Backend: run Hand Landmarker on each frame → build sequence → run word model → return word (e.g. "water").  
   - Insert that word into the answer text (like you do with letters now).

This way WLASL **helps for words** by providing **labeled video data**; you keep using **your** 63-dim hand pipeline and add a **temporal** model on top.

---

## Option B: Use Their I3D or TGCN Directly

- **I3D**: Input is **raw video**. You’d need to send a short video from the browser, run I3D in the backend (PyTorch, pretrained weights from their repo), map output to WLASL glosses.  
- **TGCN**: Input is **pre-extracted pose** (body + hands, OpenPose format). You’d need to either run OpenPose on WLASL videos (or on your camera stream) to get that format, then run their TGCN.

So WLASL **does** help (same dataset and word list), but integration is heavier: different input format (video or OpenPose keypoints), PyTorch, and their preprocessing.

---

## Practical Next Steps (Using WLASL for Words)

1. **Decide word list**  
   - Start with **WLASL100** (see `code/I3D/preprocess/wlasl_class_list.txt` or `nslt_100.json`). Many are useful for exams (e.g. help, yes, no, water, book, go, want, work, etc.).

2. **Get the data**  
   - From `sign_language_training/WLASL-master/start_kit/`:  
     - `python video_downloader.py`  # download (requires yt-dlp and network)  
     - `python preprocess.py`       # extract clips to `videos/`  
   - If links are broken, use their form to request missing/preprocessed videos (see README).

3. **Build a WLASL→MediaPipe pipeline**  
   - Script that: for each WLASL clip in `videos/`, load video → run Hand Landmarker per frame → save `(T, 63)` per word instance.  
   - Organize as e.g. `WLASL_Hand_Data/<gloss>/<video_id>_<instance>.npy`.

4. **Train a word model**  
   - LSTM/1D-CNN: input `(num_frames, 63)`, output one of 100 (or 300) words.  
   - Save this model (e.g. PyTorch/TF) and load it in the Flask app next to `gesture_model.py`.

5. **Backend API**  
   - New endpoint, e.g. `POST /process_word`: receive a short video or list of frames → extract 63-dim sequence → run word model → return `{ "word": "water" }`.

6. **Frontend**  
   - “Sign word” button: record 2–3 seconds → send to `/process_word` → insert returned word into the answer box.

---

## Summary

| Question | Answer |
|----------|--------|
| **Will WLASL help for words?** | **Yes.** It provides word-level labels and video clips (or you derive keypoints from them) for training a **word** recognizer. |
| **Can you use it with your current alphabet setup?** | **Yes**, by extracting **hand landmarks (63-dim)** from WLASL videos with MediaPipe, then training a **sequence** model (LSTM/CNN) for words. Your existing letter model stays as-is; you add a separate word model and a “record word” flow. |
| **What’s in the repo vs what you need to do?** | Repo: metadata (JSON), download/preprocess scripts, I3D/TGCN code. You still need to **download videos**, optionally **re-extract 63-dim hand features**, and **train + integrate** a word model that fits your stack. |

If you want, the next step can be a **concrete script outline** (e.g. `extract_wlasl_hand_sequences.py` and a minimal LSTM training script) that reads WLASL clips and produces `(T, 63)` data for your word model.
