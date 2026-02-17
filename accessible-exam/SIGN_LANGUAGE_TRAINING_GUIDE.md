# How to Train a Sign Language to Text Model for Long Answers

This project uses **WLASL (American Sign Language) only** for both alphabet (finger-spelling) and word recognition. Indian or other sign-language datasets are not used.

## Implemented: Alphabet (Finger-Spelling) Model for Long Answers

The app now supports **long answers** via finger-spelling using a **single-frame** alphabet model (A–Z and 1–9).

- **Data**: Use **WLASL only** (American Sign Language). For alphabet: place images per class in `sign_language_training/wlasl_alphabet_images/<Class>/` (e.g. `A/`, `B/`, …, `Z/`, `1/`, …), e.g. by exporting frames from WLASL finger-spelling or ASL letter videos. Run `process_images.py` to build `MP_Data` with 63-dim hand keypoints (Hand Landmarker).
- **Train**: From the `accessible-exam` folder run:
  ```bash
  python3 sign_language_training/train_alphabet_model.py
  ```
  This loads one frame per sequence (63 features), trains an MLP, and saves **`action.pkl`** in `accessible-exam/` for `gesture_model.py`.
- **Optional**: Add classes **Space** and **Backspace** (folders `Space/`, `Backspace/` with images) and retrain; the UI will insert a space or delete the last character when those signs are detected.

---

Currently, the system uses a **Static Gesture Recognition** approach. It looks at a single frame and determines if the hand shape matches "A", "B", "C", etc.

For **Long Answers**, you need **Continuous Sign Language Recognition (SLR)**. This is much more complex because it involves movement over time (dynamic gestures) and grammar.

Here is a roadmap to implement a system that can recognize full sentences or phrases.

## Approach: LSTM + MediaPipe Landmarks

The most feasible approach for a student project is to use **MediaPipe** to extract hand skeleton data and then train a **Recurrent Neural Network (LSTM)** to recognize sequences of movements.

### Step 1: Define Your Vocabulary
You cannot easily train a model to understand *any* sentence immediately. Start with a fixed vocabulary relevant to your exam subject (e.g., Science).

**Example Vocabulary:**
- "Photosynthesis"
- "Plants"
- "Water"
- "Sunlight"
- "Process"
- "Food"
- "Make"

### Step 2: Data Collection
You need to record data for *each* word in your vocabulary.

1.  **Create a script** to capture webcam frames.
2.  **Record 30-50 videos** for *each word*.
3.  **Process** these videos with MediaPipe to extract keypoints.
    - Instead of saving the video, save the **Landmark Coordinates** (x, y, z) for every frame.
    - Save these as NumPy arrays (`.npy` files).

**Data Structure:**
- `MP_Data/`
  - `Photosynthesis/`
    - `0.npy` (Sequence of 30 frames of landmarks)
    - `1.npy`
    - ...
  - `Water/`
    - `0.npy`
    - ...

### Step 3: Build the LSTM Model
Use TensorFlow/Keras to build a model that takes a *sequence* of landmarks as input.

```python
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 1662))) # 30 frames, 1662 keypoints
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(len(actions), activation='softmax')) # Output layer
```

### Step 4: Training
1.  Load your `.npy` data.
2.  Split into Training and Testing sets.
3.  Train the model using `model.fit()`.
4.  Save the trained weights (`action.h5`).

### Step 5: Integration
1.  Update `gesture_model.py` to collect a **buffer of 30 frames**.
2.  Pass this buffer to your LSTM model.
3.  If prediction confidence > 0.8, append the word to the sentence.

---

## Alternative: Finger Spelling (Character Level)

If training for full words is too hard, you can extend your current "Static" model to recognize **all 26 letters** (A-Z).

1.  **Train a Classifier**: Instead of heuristic `if/else` logic (which I wrote in `gesture_model.py`), collect images of A-Z.
2.  **Train a CNN** or Random Forest on the landmark data for A-Z.
3.  **Implementation**:
    - The student spells out words: "P-L-A-N-T-S".
    - You implement a "Space" gesture to separate words.
    - You implement a "Backspace" gesture.

This is slower for the user but much easier to build than a full sentence recognition system.

## Recommended Tools
- **Python Libraries**: `opencv-python`, `mediapipe`, `tensorflow`, `scikit-learn`.
- **Tutorial Reference**: Search for *"Sign Language Detection using LSTM and MediaPipe"* on YouTube (Nicholas Renotte has a simplified tutorial on this exact method).
