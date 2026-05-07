# Viva Preparation Guide

## Project Title
Smart Exam Assistance for Disabled Students (Accessible Exam Tool)

## 1) One-Line Project Definition
This project is a Flask-based accessible examination system that allows blind students to take exams using voice interaction and deaf/mute students to answer using typing plus sign-language recognition, with all answers stored as text for fair evaluation.

## 2) Problem Statement (What challenge you solved)
Traditional exams are not equally accessible for students with disabilities:
- Blind students depend on scribes to read questions and write answers.
- Deaf/mute students cannot use voice-based systems and need alternative interaction.
- Manual support reduces independence and may affect privacy and consistency.

This project solves that by providing a disability-adaptive exam platform.

## 3) Core Objective
- Build one web platform that supports multiple accessibility modes.
- Enable exam participation with minimal human dependency.
- Convert all answers to text so evaluation is standardized.

## 4) Technology Stack
- Backend: Flask (Python)
- Frontend: HTML, CSS, JavaScript (Jinja templates)
- OCR: pytesseract + Pillow
- PDF text extraction: pypdf
- Sign recognition pipeline: OpenCV + MediaPipe + ML model (pickle/CNN)
- Word-level sign model: custom model loaded from `word_model.pkl`
- Storage: JSON files on disk (`data/`, `exam_data/`)
- Browser speech: Web Speech API (SpeechSynthesis + SpeechRecognition)

## 4A) Libraries Used + Why They Are Used (Technical)

### Backend / Web
- **Flask** (`flask`)
  - **Where used**: `app.py` (routes, templates, session, JSON APIs)
  - **Why**: lightweight Python web framework; easy routing + file uploads + session management; ideal for prototype.
- **Werkzeug** (`werkzeug.utils.secure_filename`) *(comes with Flask)*
  - **Where used**: `app.py` (safe file naming on upload)
  - **Why**: prevents unsafe filenames and path traversal issues when saving uploads.

### OCR / Document Processing
- **pytesseract** (`pytesseract`)
  - **Where used**: `app.py` -> `extract_text_from_image()`
  - **Why**: performs OCR on scanned image question papers.
  - **Important note**: requires **Tesseract OCR** installed on the OS (system dependency).
- **Pillow** (`PIL.Image`, `ImageEnhance`, `ImageFilter`)
  - **Where used**: `app.py` -> `preprocess_image()` and image loading
  - **Why**: image preprocessing (grayscale/contrast/sharpen) improves OCR accuracy.
- **pypdf** (`pypdf.PdfReader`)
  - **Where used**: `app.py` -> `extract_text_from_pdf()`
  - **Why**: extracts selectable text from PDFs; also used to detect embedded images/diagrams per page to warn blind students.

### Sign Language / Computer Vision / ML Inference
- **OpenCV** (`opencv-python-headless` / `cv2`)
  - **Where used**: `app.py` endpoints (`/process_gesture`, `/process_sign`, `/process_word`) to decode base64 JPEG frames and convert them to images.
  - **Why**: fast image decoding and preprocessing for backend inference.
  - **Note**: `opencv-python-headless` is used to avoid GUI dependencies on servers.
- **NumPy** (`numpy`)
  - **Where used**: `gesture_model.py`, `word_model.py`, `app.py`
  - **Why**: stores and transforms landmark arrays; reshaping sequences; numerical operations.
- **MediaPipe** (`mediapipe`)
  - **Where used**:
    - `gesture_model.py`: `vision.HandLandmarker` loads `hand_landmarker.task` and extracts 21 hand landmarks
    - `word_model.py`: same detector for sequences
  - **Why**: robust, real-time hand landmark detection (21 points) which becomes the input features for ML classification.
- **protobuf**
  - **Where used**: indirect dependency required by MediaPipe.
  - **Why**: MediaPipe model/config serialization; version pinned to avoid compatibility issues.
- **pickle / json** *(Python standard library)*
  - **Where used**:
    - `gesture_model.py`: loads `action.pkl` (trained classifier) and classes
    - `word_model.py`: loads `word_model.pkl` + `word_classes.json`
    - `app.py`: stores papers/users/exam sessions to JSON
  - **Why**: simple persistence mechanism for prototype and fast model loading.
- **scikit-learn** (`sklearn`)
  - **Where used**: training scripts in `sign_language_training/` and for loading `action.pkl` / `word_model.pkl` if they were trained with sklearn.
  - **Why**: quick baseline models (MLP/Linear models), probability outputs (`predict_proba`) for confidence thresholding.
- **TensorFlow / Keras** (`tensorflow`, `keras`) *(optional)*
  - **Where used**: `gesture_model.py` loads `action_cnn.keras` if present; training via `sign_language_training/train_alphabet_cnn.py`
  - **Why**: CNN-based landmark classifier can improve letter recognition accuracy over simple MLP.
  - **Note**: TensorFlow support depends on Python version; use as optional enhancement.

### Browser-side Speech (No server Python libs needed)
- **Web Speech API** *(built into Chrome/Edge)*
  - **Where used**:
    - `templates/blind_exam.html`: `speechSynthesis` (TTS), `SpeechRecognition` (STT)
    - `templates/completed.html`, `templates/student_dashboard.html`: TTS prompts
  - **Why**: real-time interaction, minimal server load, easier deployment than server-side audio stacks.
  - **Limitations**: SpeechRecognition may require internet and is browser-dependent (Chrome recommended).

### Optional / Legacy (present in requirements but not required for the current browser-speech design)
- **pyttsx3**, **SpeechRecognition**, **pyaudio**
  - **Why they appear**: older/alternative server-side speech approach.
  - **In this build**: voice exam primarily uses browser Web Speech API; these libs are not essential unless you re-enable server speech features.

## 4B) Installation Guide (Full Setup)

### Prerequisites
- **Python**: 3.9+ recommended (some ML stacks like TensorFlow may require specific versions)
- **Browser**: Google Chrome recommended (best Web Speech support)
- **Microphone** permission: required for blind mode STT
- **Camera** permission: required for deaf/mute mode sign capture
- **Tesseract OCR (system dependency)**: required for OCR on images

### Step 1 — Create virtual environment (recommended)

```bash
cd "accessible-exam"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### Step 2 — Install Python packages

```bash
pip install -r requirements.txt
```

### Step 3 — Install Tesseract OCR (OS-level)

#### macOS (Homebrew)
```bash
brew install tesseract
```

#### Ubuntu / Debian
```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr
```

#### Windows
- Install “Tesseract OCR” and ensure the install folder is added to **PATH**.
- Then restart the terminal/IDE so `pytesseract` can call it.

### Step 4 — MediaPipe model file (`hand_landmarker.task`)
This project expects `hand_landmarker.task` in the `accessible-exam/` directory.

- If it’s missing, run:
```bash
python download_hand_landmarker.py
```

### Step 5 — Run the application

```bash
python app.py
```

Open the site at `http://127.0.0.1:5000`.

### Step 6 — Enable sign recognition models (optional training)
The app can run immediately, but for accurate sign recognition you need trained model files:
- Letter model: `action.pkl` (baseline) or `action_cnn.keras` (better)
- Word model: `word_model.pkl` + `word_classes.json` (optional)

Recommended letter model training (high level):
```bash
python sign_language_training/record_letters.py
python sign_language_training/process_images.py
python sign_language_training/train_alphabet_model.py
```

For word model, see `sign_language_training/README_WORD_MODEL.md`.

## 4C) Technical “What happens internally” (Library-to-Flow Mapping)

### Teacher upload → question extraction
1. Flask receives file: `request.files` (Flask/Werkzeug)
2. File stored: `secure_filename()` (Werkzeug)
3. If PDF:
   - parse pages and text: `pypdf.PdfReader`
4. If image:
   - read and preprocess: `Pillow`
   - OCR: `pytesseract` → plain text
5. Split into instructions/questions: Python regex (`re`)
6. Persist:
   - exam JSON in `exam_data/` (json)
   - paper list in `data/papers.json` (json)

### Blind exam → voice interaction
1. Page loads questions (Jinja → JSON blob)
2. TTS reads question: `speechSynthesis` (Web Speech API)
3. Student speaks answer: `SpeechRecognition` (Web Speech API)
4. Answer is saved to server: `fetch('/api/save-answer')` (JS → Flask JSON API)
5. Flask updates student session JSON.

### Deaf exam → sign recognition
1. Webcam stream: `getUserMedia()` (browser)
2. Frames captured as JPEG base64 (canvas)
3. Backend decodes to image: `cv2.imdecode()` (OpenCV) + `numpy`
4. Hand landmarks: MediaPipe HandLandmarker (21×3 = 63 features)
5. Feature normalization: `landmark_utils.py`
6. Classification:
   - letter model: sklearn pickle or Keras CNN (`gesture_model.py`)
   - optional word model: sklearn pickle (`word_model.py`)
7. Answer saved: `/submit_deaf_answer` updates session JSON.

## 5) High-Level Architecture
1. Teacher uploads question paper (PDF/image).
2. Backend extracts and structures questions.
3. Student logs in and chooses exam mode (voice or visual).
4. System loads exam in mode-specific interface:
   - Voice mode for blind students.
   - Visual/sign mode for deaf/mute students.
5. Answers are autosaved to server.
6. Completion page shows answered count and review.

## 6) Folder & File Roles (Important for viva)
- `app.py`: Main Flask app, routes, auth logic, upload pipeline, APIs, exam flow.
- `templates/index.html`: Teacher and student login.
- `templates/teacher_dashboard.html`: Upload and paper management UI.
- `templates/student_dashboard.html`: Student paper listing and exam launch.
- `templates/blind_exam.html`: Voice-first exam UI (TTS/STT, commands, shortcuts).
- `templates/deaf_exam.html`: Visual/sign exam UI (camera capture + typed input).
- `templates/completed.html`: Final answer review and completion summary.
- `gesture_model.py`: Letter/digit sign inference model + confidence/voting logic.
- `word_model.py`: Sequence-based word recognition from multiple frames.
- `sign_language_training/`: scripts for dataset prep, feature extraction, and training.
- `exam_data/`: per-student exam session JSON files.
- `data/users.json`, `data/papers.json`: login data and paper metadata.

## 7) Detailed Build Explanation (Module by Module)

### 7.1 Authentication and role separation
`app.py` uses session-based login with two roles:
- Teacher
- Student

Teacher can upload/manage papers. Student can only see active papers matching their mode.

### 7.2 Question paper processing pipeline
When teacher uploads:
1. File extension is validated.
2. File is saved in `uploads/`.
3. Text extraction:
   - PDF -> `extract_text_from_pdf()`
   - Image -> preprocess + `extract_text_from_image()`
4. `split_sections()` separates instruction lines and numbered questions.
5. Visual references are flagged by regex (`figure`, `diagram`, etc.).
6. Data is saved to `exam_data/<exam_id>.json`.
7. Paper metadata is saved in `data/papers.json`.

### 7.3 Student exam session creation
When student starts an exam:
1. Selected paper is validated as active.
2. Master exam JSON is loaded.
3. A new per-student session copy is created via `save_exam_data(...)`.
4. Session stores the new `exam_id`.
5. Student is redirected based on mode:
   - `voice` -> `/blind-exam`
   - `visual` -> `/deaf-exam`

This design isolates each student’s answers from the master paper.

### 7.4 Blind mode (Voice exam)
Implemented in `templates/blind_exam.html`:
- Reads instructions/questions using browser TTS.
- Captures spoken answers using browser SpeechRecognition.
- Supports:
  - voice commands (`answer`, `submit`, `next`, `repeat`, etc.)
  - keyboard shortcuts for accessibility fallback
  - speed control, mute/unmute, help panel
- Each answer is posted to `/api/save-answer`.

Backend `/api/save-answer` updates only answer fields inside student exam JSON.

### 7.5 Deaf/mute mode (Visual exam)
Implemented in `templates/deaf_exam.html`:
- Student can type answers directly.
- Optional camera-based sign capture:
  - Frontend records webcam frames.
  - Frames are sent as base64 to backend endpoints.

Backend inference endpoints:
- `/process_gesture`: single/multi-frame letter prediction.
- `/process_sign`: unified mode-aware sign detection (letter/number/general).
- `/process_word`: sequence-level word prediction (if word model loaded).
- `/word_suggestions`: autocomplete suggestions for spell-building workflow.

Answers are saved through `/submit_deaf_answer`.

### 7.6 Completion and review
`/completed` renders:
- answered count vs total
- full question-wise answer review
- optional read-all via TTS

## 8) Data Storage Design
This project intentionally uses JSON files for prototype simplicity:
- Easy to debug and inspect.
- No DB setup needed for demo environments.
- Suitable for final-year prototype scale.

Limitations (for viva honesty):
- Not ideal for large concurrent deployments.
- Needs migration to SQL/NoSQL for production.

## 9) Sign Model Pipeline (Training + Inference)

### 9.1 Letter model
Training flow:
1. Collect images (`record_letters.py` or external dataset).
2. Extract landmarks (`process_images.py`) -> `MP_Data/`.
3. Train model (`train_alphabet_model.py` or `train_alphabet_cnn.py`).
4. Save model (`action.pkl` or `action_cnn.keras`).

Runtime flow:
1. Frame -> MediaPipe hand landmarks (21 points x 3 dims = 63 features).
2. Normalize landmarks.
3. Model predicts class.
4. Confidence threshold and frame voting reduce false positives.

### 9.2 Word model
Training flow:
1. Prepare short sign video sequences (WLASL/custom).
2. Convert to 30-frame landmark sequences.
3. Train sequence classifier (`train_word_model.py`).
4. Save `word_model.pkl` and `word_classes.json`.

Runtime:
- Multi-frame clip from frontend is converted to keypoint sequence and classified.

## 10) Why this design is technically strong
- Accessibility-first UX, not accessibility added later.
- Mode-specific interfaces reduce cognitive/interaction barriers.
- Frontend speech APIs reduce backend complexity and latency.
- Per-student exam session cloning avoids data collision.
- Progressive fallback:
  - if word model unavailable -> letter mode still works
  - if speech unsupported -> keyboard shortcuts still allow exam flow

## 11) Known Limitations
- Hardcoded prototype credentials and Flask secret key (not production-safe).
- JSON persistence instead of transactional DB.
- STT accuracy depends on browser, internet, accent, and noise.
- Sign accuracy depends on training quality, lighting, and camera conditions.
- No advanced invigilation/anti-cheating module yet.

## 12) Future Scope
- Secure authentication with hashed passwords and role-based admin panel.
- Migrate to SQLite/PostgreSQL.
- Timer, autosubmit, and proctoring aids.
- Multi-language voice support.
- Improved word-level sign vocabulary with larger datasets.
- Teacher submissions analytics/export.

## 13) 2-Minute Viva Script
We developed a Flask-based accessible exam system for disabled students. The platform supports two adaptive modes: voice mode for blind students and visual-sign mode for deaf and mute students. Teachers upload question papers in PDF or image format, and the backend extracts text using pypdf or OCR, then separates instructions and questions.

When a student starts an exam, the system creates an independent exam session for that student. In blind mode, questions are read aloud using browser text-to-speech, and answers are captured through speech recognition with voice commands and keyboard shortcuts. In deaf/mute mode, students can type answers and also use camera-based sign capture. The backend processes sign frames with OpenCV and MediaPipe landmarks, then applies trained ML models for letters, numbers, and optional word recognition.

All responses are converted and saved as text in structured exam sessions, and a completion page provides summary and review. The system improves independence, accessibility, and fairness in exams while keeping the architecture practical for deployment in educational institutions.

## 14) Typical Viva Questions with Model Answers

### Q1: Why did you choose Flask?
Flask is lightweight, easy to structure for prototypes, and integrates well with Python libraries for OCR, ML inference, and file/session handling.

### Q2: How do you process both PDF and image question papers?
We validate file type first, then use pypdf for text PDFs and pytesseract with image preprocessing for image-based papers.

### Q3: How are instructions separated from questions?
We use regex-based parsing to detect numbered question patterns and classify text before first question as instructions.

### Q4: How do blind students interact without mouse?
Through automatic TTS reading, voice commands, and complete keyboard shortcuts for every major action.

### Q5: Why use browser speech APIs instead of Python speech libraries?
Browser APIs reduce server load, avoid blocking calls, and give low-latency interaction. It also simplifies deployment for this architecture.

### Q6: How does sign recognition work technically?
Camera frames are captured in frontend, sent to backend, hand landmarks are extracted using MediaPipe, then ML model predicts sign class. Multi-frame voting improves reliability.

### Q7: What if word model is not available?
The system continues with letter/number-level capture, so exam functionality is preserved.

### Q8: How do you prevent one student’s answers mixing with another’s?
Each exam start creates a unique exam session file keyed by generated exam ID and linked to student identity.

### Q9: Why did you store data in JSON instead of database?
JSON was chosen for rapid prototyping, easy inspection, and low setup overhead. We acknowledge DB migration for production.

### Q10: What are the major limitations?
Security hardening, scalable storage, and model accuracy in diverse real-world conditions are current limitations and planned improvements.

### Q11: What is the innovation in your project?
A single integrated platform supports multiple disability-specific interaction modalities in a complete exam workflow, not just isolated assistive demos.

### Q12: How is fairness improved?
Students independently access the same paper through suitable interfaces, and all outputs are normalized to text for uniform evaluation.

## 15) Demo Order for Viva (Recommended)
1. Show login page and role split.
2. Teacher uploads paper and show extracted question count.
3. Student login in voice mode, answer one question by speech.
4. Student login in visual mode, answer one question using sign/typing.
5. Show completion page and saved answers summary.
6. Briefly show `exam_data/*.json` to prove persistence.

## 16) Short Closing Line
This project demonstrates a practical, AI-assisted accessible exam architecture that can be scaled into a production system with stronger security, database integration, and richer sign-language models.

