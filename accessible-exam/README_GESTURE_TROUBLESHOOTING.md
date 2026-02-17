# Gesture / alphabet / word not detecting — troubleshooting

## Fixes already in the app

1. **Alphabet (letters)**  
   The server was unpacking the gesture result incorrectly (`_, gesture = ...`). This is fixed: the backend now returns the detected letter (e.g. "A", "1") correctly.

2. **Camera and frame size**  
   The frontend only sends frames when the video has valid dimensions and starts the gesture loop after the video is ready, so the server is not given empty or invalid images.

3. **Hand detection sensitivity**  
   Hand Landmarker uses `min_hand_detection_confidence=0.3` so hands are detected more easily.

4. **CPU delegate**  
   MediaPipe is configured to use the CPU delegate to reduce GPU-related issues.

---

## If you see "Show hand to camera" or "Waiting..." all the time

- **Use one hand** in frame, with **good lighting** and a **plain background**.
- Hold the sign **steady** for about a second so the model can classify it.
- Make sure the **hand is clearly visible** (palm or back of hand facing the camera, not too small in frame).

---

## If the server fails with "Could not create an NSOpenGLPixelFormat" or "kGpuService"

MediaPipe is trying to use the GPU and failing (common on macOS, especially over SSH or without a display).

**What to do:**

1. **Run the app in a normal desktop session**  
   Start Flask (e.g. `python3 app.py`) from a terminal on the same Mac where you use the browser (not over SSH without a display).

2. **Use a different MediaPipe build (e.g. on Apple Silicon)**  
   Try `mediapipe-silicon` if you are on M1/M2/M3:
   ```bash
   pip uninstall mediapipe
   pip install mediapipe-silicon
   ```
   Then run the app again.

3. **Run with a virtual display (advanced)**  
   On a headless Linux server you can use `xvfb-run python3 app.py` so the process has a virtual display. On macOS this is less common.

---

## Quick test (alphabet pipeline)

From the `accessible-exam` folder:

```bash
python3 test_gesture_pipeline.py
```

- If you see "Gesture model loaded OK" and "Result: None" with the dummy image, the model and detector load correctly; "None" is expected when no hand is in the image.
- If you see "FAIL: Hand Landmarker not loaded", the MediaPipe Hand Landmarker failed to start (often the OpenGL/GPU error above).
- To test with a real hand image:  
  `python3 test_gesture_pipeline.py path/to/photo_of_hand.jpg`

---

## Word model

Word detection needs `word_model.pkl` (and optionally `word_classes.json`) from training on WLASL data. If you have not run the WLASL extraction and training, the "Record word" button will return "Word model not loaded" — that is expected. Alphabet (letters) and typing still work.
