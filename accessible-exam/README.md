# Accessible Exam Tool

A web-based accessible examination system designed for **visually impaired (blind) students**, enabling them to take exams independently using voice interaction. Built as a final-year project.

---

## Problem Statement

Visually impaired students face significant barriers during written examinations. They typically depend on scribes (human writers) to read questions and write answers, which compromises independence, privacy, and accuracy. There is a need for a technology-driven solution that allows blind students to take exams on their own.

---

## Solution Overview

The **Accessible Exam Tool** is a Flask-based web application where:

1. **Teachers** log in, upload question papers (PDF or image), and assign them to **Voice** (blind) or **Visual** (deaf) or both.
2. The system **extracts** questions using OCR (images) or PDF text parsing.
3. **Students** log in with roll number and DOB, choose exam type, and start an exam:
   - **Voice exam (blind):** Questions read aloud (TTS), answers by speech (STT), full keyboard navigation.
   - **Visual/Deaf exam:** Answer by sign language (camera) and/or typing, with word/phrase suggestions.
4. Answers are saved (SQLite + JSON). **Teachers** can view submissions per paper and **download answer sheets as CSV** (or view/print) for evaluation.

---

## Key Features Implemented

### 1. Question Paper Upload & Text Extraction
- Accepts **PDF** and **image** formats (PNG, JPG, JPEG, WebP, BMP, TIFF, GIF).
- **OCR** (Optical Character Recognition) via `pytesseract` + `Pillow` for image-based papers.
- **PDF text extraction** via `pypdf` for text-based PDFs.
- **Image preprocessing** (contrast enhancement, sharpening) for better OCR accuracy.
- Automatic **separation of instructions** from numbered questions using regex.

### 2. Diagram/Image Detection & Warnings
- Detects embedded images inside PDFs and flags them with spoken warnings.
- Scans question text for visual keywords (figure, diagram, graph, chart, table, etc.).
- Alerts the student: *"This question refers to a visual element. Ask your invigilator to describe it."*

### 3. Text-to-Speech (TTS) — Questions Read Aloud
- Uses the **Web Speech API** (`SpeechSynthesis`) — runs entirely in the browser.
- Questions are **automatically read aloud** one at a time when displayed.
- **Speed control**: Students can increase/decrease speech speed using `+` and `-` keys.
- **Mute/Unmute** toggle with a speaker button (keyboard shortcut: `M`).

### 4. Speech-to-Text (STT) — Voice Answers
- Uses the **Web Speech API** (`SpeechRecognition`) for real-time voice capture.
- Students press `A` or click "Answer" to start recording.
- The spoken answer is transcribed and displayed on screen.
- Students can **re-record**, **clear**, or **submit** their answer.

### 5. Complete Keyboard Navigation (11 Shortcuts)
No mouse required. All actions are accessible via keyboard:

| Key | Action |
|-----|--------|
| `R` | Repeat current question |
| `A` | Start voice answer |
| `C` | Clear current answer |
| `Enter` | Submit answer & go to next |
| `N` / `→` | Skip to next question |
| `P` / `←` | Go to previous question |
| `V` | Review all answers |
| `M` | Mute / Unmute audio |
| `H` | Toggle help panel |
| `+` / `-` | Increase / Decrease speech speed |
| `Esc` | Stop audio / Cancel recording |

### 6. Exam State Management
- Each exam session generates a **unique ID** (UUID).
- Exam data (questions, instructions, answers) stored as **JSON files on disk** — not in cookies.
- Only the 8-character exam ID is stored in the Flask session cookie (avoids the 4KB cookie size limit).
- Answers are **auto-saved** to the server via AJAX after each submission.

### 7. Visual Question Grid & Progress Tracking
- Right-side panel shows a **numbered grid** of all questions.
- Color-coded: **purple** = current, **green** = answered, **white** = unanswered.
- Students can **click any question** to jump to it directly.
- A **question counter** ("Question 5 of 40") is always visible.

### 8. Completion & Review Page
- Shows a summary: *"You answered X of Y questions."*
- Lists all questions with their recorded answers.
- **Read All Answers** button reads everything aloud (shortcut: `R`).
- Option to start a new exam.

### 9. Accessibility Standards
- **ARIA attributes** on all interactive elements (`aria-label`, `aria-live`, `role`).
- **Skip-to-content** link on every page for screen reader users.
- **Focus-visible** outlines (orange 3px) on all focusable elements.
- **`prefers-reduced-motion`** support — disables animations for users who need it.
- **`prefers-contrast: high`** support — stronger borders for high-contrast mode.
- **Semantic HTML5** structure (`<main>`, `<nav>`, `<section>`).

### 10. Fully Responsive Design
Adapts to all screen sizes with 4 breakpoints:

| Screen Size | Breakpoint | Behavior |
|------------|-----------|----------|
| Large desktop | 1300px+ | Full two-column layout, wide exam area |
| Desktop | 768px – 1299px | Standard layout |
| Tablet | 481px – 768px | Columns stack vertically, adjusted spacing |
| Mobile | ≤ 480px | Compact layout, smaller buttons/fonts |
| Small mobile | ≤ 360px | Ultra-compact, full-width buttons |

---

## Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Backend** | Flask (Python) | Web server, routing, auth, file handling |
| **OCR** | pytesseract + Pillow | Extract text from images of question papers |
| **PDF Parsing** | pypdf | Extract text from PDF question papers |
| **Sign language** | OpenCV, MediaPipe, NumPy | Hand detection & gesture/word recognition (Deaf mode) |
| **TTS** | Web Speech API (SpeechSynthesis) | Read questions aloud in the browser |
| **STT** | Web Speech API (SpeechRecognition) | Capture voice answers in the browser |
| **Frontend** | HTML5, CSS3, Vanilla JavaScript | UI, keyboard navigation, accessibility |
| **Data Storage** | SQLite + JSON | Exam sessions & answers (DB); papers & users (JSON) |
| **Templating** | Jinja2 | Server-side HTML rendering |

---

## Requirements (Python dependencies)

All dependencies are listed in **`requirements.txt`**. Brief usage:

| Package | Used for |
|--------|----------|
| **flask** | Web app: routes, sessions, auth, file uploads, APIs. |
| **pypdf** | Extracting text from PDF question papers. |
| **Pillow** | Opening and preprocessing images before OCR. |
| **pytesseract** | OCR: turning question-paper images into text. *(Requires Tesseract installed on the system, e.g. `brew install tesseract` on macOS.)* |
| **opencv-python-headless** | Processing video frames for sign-language hand detection (Deaf mode). |
| **numpy** | Array handling for frames and landmark data (gesture/word models). |
| **mediapipe** | Hand landmark detection for sign recognition (letters and words). |
| **protobuf** | Used by MediaPipe; keep version &lt;4 for compatibility. |
| **scikit-learn** | *Optional.* Only needed if you run `train_alphabet_model.py` or `train_word_model.py` to train sign-language models. |

**Not required:** TTS and STT run in the browser via the Web Speech API, so **pyttsx3**, **SpeechRecognition**, and **PyAudio** are not in `requirements.txt`.

---

## Project Architecture

```
accessible-exam/
├── app.py                    # Flask backend (routes, OCR, PDF, sign APIs, DB)
├── requirements.txt          # Python dependencies (see Requirements section above)
├── README.md                 # This file
├── gesture_model.py          # Letter/digit sign recognition (MediaPipe + ML)
├── word_model.py             # Word-level sign recognition (WLASL-style)
├── landmark_utils.py         # Hand landmark normalization for models
│
├── static/
│   └── style.css             # Styles (responsive, accessible)
│
├── templates/
│   ├── index.html            # Login — Teacher or Student (voice / visual)
│   ├── teacher_dashboard.html   # Upload papers, view submissions
│   ├── teacher_submissions.html  # List submissions per paper; View / Download CSV
│   ├── teacher_view_answer_sheet.html  # Printable answer sheet
│   ├── student_dashboard.html    # Available papers, Start Exam
│   ├── blind_exam.html       # Voice exam (TTS/STT, keyboard)
│   ├── deaf_exam.html        # Deaf mode — sign + type answers
│   ├── deaf.html             # Deaf mode entry
│   └── completed.html        # Completion — answer review, Read All
│
├── data/                     # JSON + SQLite (auto-created)
│   ├── users.json            # Teachers, blind_students, deaf_students
│   ├── papers.json           # Uploaded paper metadata
│   ├── phrase_suggestions.json # Words + phrases for Deaf mode autocomplete
│   ├── phrase_suggestions.json
│   └── exam.db               # SQLite: exam_sessions, answers (for teacher download)
│
├── uploads/                  # Uploaded question papers (auto-created)
├── exam_data/                # JSON exam session files (auto-created)
└── sign_language_training/   # Scripts to train letter/word models (optional)
```

---

## Datasets and data files

Data used by the app and by the optional sign-language training pipeline:

### Application data (used at runtime)

| Dataset / file | Location | Used by | Purpose |
|----------------|----------|---------|---------|
| **users.json** | `data/users.json` | `app.py` (`get_users`, auth) | Teachers and students (blind_students, deaf_students): usernames, passwords, DOB, names. |
| **papers.json** | `data/papers.json` | `app.py` (dashboards, start exam) | Uploaded paper metadata: subject, exam_id, teacher, exam_mode, num_questions. |
| **exam.db** | `data/exam.db` | `app.py` (DB helpers, teacher submissions) | SQLite: exam_sessions, answers. Used for teacher submission list and CSV download. |
| **exam_data/*.json** | `exam_data/<exam_id>.json` | `app.py` (load/save exam, answers) | Per-session exam content: instructions, questions, answers; paper_id, student_id. |
| **phrase_suggestions.json** | `data/phrase_suggestions.json` | `app.py` (`/word_suggestions`) | **words**: prefix-based autocomplete list; **phrases**: context-based next-word suggestions (e.g. "help" → "me"). |
| **word_classes.json** | `word_classes.json` (project root) | `word_model.py` | Class names for the word sign model (output of `train_word_model.py`). |
| **word_model.pkl** | Project root | `word_model.py` | Trained word-level sign classifier (WLASL-style). Optional; Deaf mode works with letters only without it. |
| **action.pkl** / **action_cnn.keras** + **action_classes.pkl** | Project root | `gesture_model.py` | Letter/digit sign model(s). Optional; from `train_alphabet_model.py` or `train_alphabet_cnn.py`. |
| **hand_landmarker.task** | Project root (or sign_language_training/) | `gesture_model.py`, `word_model.py`, extraction scripts | MediaPipe hand-landmark model; required for sign recognition and for building training data. |

### Sign-language training data (optional)

| Dataset / folder | Location | Used by | Purpose |
|------------------|----------|---------|---------|
| **WLASL (metadata)** | `sign_language_training/WLASL-master/start_kit/WLASL_v0.3.json` | `extract_wlasl_hand_sequences.py`, `download_wlasl_subset.py`, WLASL start_kit scripts | Word-level ASL dataset index: glosses, video_id, url, frame ranges, split. |
| **WLASL videos** | `sign_language_training/WLASL-master/start_kit/videos/` (after download) | `extract_wlasl_hand_sequences.py` | Video clips per gloss; converted to hand sequences. |
| **WLASL_Hand_Data** | `WLASL_Hand_Data/<gloss>/<id>.npy` | `train_word_model.py` | Extracted hand sequences (30×63) per word; input for word model training. |
| **Custom_Hand_Data** | Optional folder from `extract_custom_word_sequences.py` | `train_word_model.py` (e.g. `--data-dir Custom_Hand_Data`) | Same format as WLASL_Hand_Data for custom word videos. |
| **MP_Data** | `MP_Data/<Class>/<sequence>/0.npy` (etc.) | `train_alphabet_model.py`, `train_alphabet_cnn.py`, `build_wlasl_from_mpdata.py` | Per-class hand keypoints (63-d or 21×3) for letter/digit training; produced by `process_images.py`. |
| **wlasl_alphabet_images** | `sign_language_training/wlasl_alphabet_images/<Class>/` | `process_images.py` | Input images per letter/digit; script writes keypoints to `MP_Data`. |
| **ASL Alphabet (Kaggle)** | Downloaded via `download_asl_dataset.py` | Optional; organize into class folders for `process_images.py` | External ASL letter images for building MP_Data. |

### File → usage summary

- **app.py:** `data/users.json`, `data/papers.json`, `data/exam.db`, `data/phrase_suggestions.json` (words + phrases), `exam_data/*.json`.
- **gesture_model.py:** `action.pkl` or `action_cnn.keras` + `action_classes.pkl`, `hand_landmarker.task`.
- **word_model.py:** `word_model.pkl`, `word_classes.json`, `hand_landmarker.task`.
- **train_alphabet_model.py:** `MP_Data` → produces `action.pkl`.
- **train_alphabet_cnn.py:** `MP_Data` → produces `action_cnn.keras`, `action_classes.pkl`.
- **train_word_model.py:** `WLASL_Hand_Data` or `Custom_Hand_Data` → produces `word_model.pkl`, `word_classes.json`.
- **extract_wlasl_hand_sequences.py:** WLASL videos + `WLASL_v0.3.json` → writes `WLASL_Hand_Data`.
- **extract_custom_word_sequences.py:** Custom video folder → writes `WLASL_Hand_Data` or `Custom_Hand_Data`.
- **process_images.py:** `wlasl_alphabet_images` (or similar) + `hand_landmarker.task` → writes `MP_Data`.
- **build_wlasl_from_mpdata.py:** `MP_Data` → builds `WLASL_Hand_Data` when WLASL videos are not available.

---

## How It Works — Step by Step

### Step 1: Teacher Uploads Question Paper
- Teacher logs in, goes to dashboard, and uploads a PDF or image (subject name, assign to Voice/Visual/Both).
- The system extracts text (OCR or PDF), splits instructions and questions, and creates an exam session.
- Paper appears in the teacher’s list; a copy is stored for students who match the assigned mode.

### Step 2: Student Enters Exam
- Student logs in with roll number and DOB, selects Voice or Visual exam type.
- Dashboard shows available papers; student clicks **Start Exam** for a paper.
- For voice: redirect to `/blind-exam`. For deaf: redirect to `/deaf-exam`.
- If instructions exist (voice), they are read aloud first; then student starts questions.

### Step 3: Answering Questions
- Each question is displayed one at a time and **read aloud automatically**.
- The student presses `A` to activate the microphone and speaks their answer.
- The answer appears on screen. They can:
  - Press `Enter` to **submit and move to the next question**.
  - Press `A` again to **re-record**.
  - Press `C` to **clear** the answer.
  - Press `N` to **skip**.
  - Press `P` to go **back**.

### Step 4: Completion
- After the last question, the student sees a completion summary.
- They can review all answers (read aloud with `V` or `R`).
- Clicking "Finish Exam" navigates to the completion page with the full summary.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|---------|-------------|
| GET | `/` | Login — Teacher or Student (voice/visual) |
| GET/POST | `/teacher/login` | Teacher login |
| GET | `/teacher/dashboard` | Upload papers, list submissions |
| POST | `/teacher/upload` | Upload question paper (PDF/image) |
| GET | `/teacher/paper/<paper_id>/submissions` | List submitted answer sheets |
| GET | `/teacher/view-answer-sheet/<exam_id>` | View answer sheet (printable) |
| GET | `/teacher/download-answer-sheet/<exam_id>.csv` | Download CSV for evaluation |
| GET/POST | `/student/login` | Student login (roll, DOB, exam type) |
| GET | `/student/dashboard` | Available papers, Start Exam |
| GET | `/student/start-exam/<paper_id>` | Start exam (voice or deaf) |
| GET | `/blind-exam` | Voice exam page |
| GET | `/deaf-exam` | Deaf mode exam page |
| POST | `/api/save-answer` | AJAX — save answer (blind) |
| POST | `/submit_deaf_answer` | AJAX — save answer (deaf) |
| GET | `/api/get-answers` | AJAX — get saved answers |
| POST | `/process_gesture`, `/process_word`, `/process_sign` | Sign-language APIs |
| GET | `/completed` | Completion — answer summary |
| GET | `/logout` | Logout |
| GET | `/reset` | Clear exam session |

---

## How to Run

### Prerequisites
- Python 3.9+
- Tesseract OCR installed (`brew install tesseract` on macOS)
- Google Chrome (recommended for Speech Recognition support)

### Installation

```bash
cd accessible-exam
pip3 install -r requirements.txt
python3 app.py
```

Then open **http://127.0.0.1:5000** in your browser.  
See the **Requirements** section above for what each dependency is used for.

### Using a Virtual Environment (optional)

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
```

---

## Important Notes

- **Data:** Exam sessions and answers are stored in **SQLite** (`data/exam.db`). Teachers see submissions per paper and can **View** or **Download CSV** for evaluation. JSON in `exam_data/` and `data/` is still used for session data and config.
- **uploads/** stores uploaded question papers; **exam_data/** holds per-session JSON; both can be cleared periodically.
- **Best browser:** Google Chrome (full Web Speech API for TTS and STT).
- **Microphone:** Speech Recognition needs microphone permission when the student uses voice answer.
- TTS works offline (browser voices). STT on Chrome typically needs an internet connection.

---

## Future Enhancements (Scope for Extension)

- **Timer support** — Configurable exam duration with audio countdown.
- **Multi-language support** — TTS/STT in regional languages.
- **Answer export as PDF** — In addition to CSV.
- **Stronger authentication** — e.g. roll number verification, password for students.

---

See **BUILD_PLAN.md** for the step-by-step development plan.
