# Accessible Exam Tool

A web-based accessible examination system designed for **visually impaired (blind) students**, enabling them to take exams independently using voice interaction. Built as a final-year project.

---

## Problem Statement

Visually impaired students face significant barriers during written examinations. They typically depend on scribes (human writers) to read questions and write answers, which compromises independence, privacy, and accuracy. There is a need for a technology-driven solution that allows blind students to take exams on their own.

---

## Solution Overview

The **Accessible Exam Tool** is a Flask-based web application where:

1. **Staff/invigilators** upload the question paper (as a PDF or image).
2. The system **automatically extracts** all questions using OCR (for images) or PDF text parsing.
3. **Blind students** interact with the exam entirely through **voice**:
   - Questions are **read aloud** using Text-to-Speech (TTS).
   - Students **speak their answers** using Speech-to-Text (STT).
   - Full **keyboard navigation** is available — no mouse needed.
4. Answers are saved and a **completion summary** is shown at the end.

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
| **Backend** | Flask (Python) | Web server, routing, file handling |
| **OCR** | pytesseract + Pillow | Extract text from images of question papers |
| **PDF Parsing** | pypdf | Extract text from PDF question papers |
| **TTS** | Web Speech API (SpeechSynthesis) | Read questions aloud in the browser |
| **STT** | Web Speech API (SpeechRecognition) | Capture voice answers in the browser |
| **Frontend** | HTML5, CSS3, Vanilla JavaScript | UI, keyboard navigation, accessibility |
| **Data Storage** | JSON files on disk | Per-session exam data (questions + answers) |
| **Templating** | Jinja2 | Server-side HTML rendering |

---

## Project Architecture

```
accessible-exam/
├── app.py                    # Flask backend (routes, OCR, PDF, API)
├── requirements.txt          # Python dependencies
├── README.md                 # This file
│
├── static/
│   ├── style.css             # All styles (responsive, accessible)
│   └── image.png             # Background illustration
│
├── templates/
│   ├── index.html            # Home page — mode selection
│   ├── blind.html            # Upload page — staff uploads question paper
│   ├── blind_exam.html       # Exam page — student takes exam by voice
│   ├── completed.html        # Completion — answer review & summary
│   └── deaf.html             # Deaf mode (placeholder)
│
├── uploads/                  # Uploaded question papers (auto-created)
├── exam_data/                # JSON exam sessions (auto-created)
└── BUILD_PLAN.md             # Step-by-step development plan
```

---

## How It Works — Step by Step

### Step 1: Staff Uploads Question Paper
- Staff navigates to `/blind` and uploads a PDF or image of the question paper.
- The system extracts text using OCR (for images) or PDF parsing.
- Instructions and questions are automatically separated.
- Data is saved as a JSON file on disk.

### Step 2: Student Enters Exam
- The student clicks "Proceed to Exam" and is taken to `/blind-exam`.
- If instructions exist, they are read aloud first.
- The student presses Enter or clicks "Start Questions" to begin.

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
| GET | `/` | Home page — mode selection |
| GET/POST | `/blind` | Upload page — handles file upload & OCR |
| GET | `/blind-exam` | Exam page — loads questions from stored data |
| POST | `/api/save-answer` | AJAX — saves a single answer `{ question_index, answer }` |
| GET | `/api/get-answers` | AJAX — retrieves all saved answers |
| GET | `/completed` | Completion page — shows answer summary |
| GET | `/reset` | Clears session and redirects to upload page |

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

### Using a Virtual Environment (optional)

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
```

---

## Important Notes

- The `exam_data/` folder contains JSON files that are automatically created per exam session. They store the extracted questions, instructions, and student answers. These files can be safely deleted after use.
- The `uploads/` folder stores uploaded question paper files (images/PDFs). These can also be cleared periodically.
- **Best browser**: Google Chrome (full Web Speech API support for both TTS and STT).
- Speech Recognition requires **microphone permission** — the browser will prompt for it.
- TTS works offline using built-in browser voices. STT requires an internet connection on Chrome.

---

## Future Enhancements (Scope for Extension)

- **Deaf & Mute Mode** — Sign language recognition using camera + ML models.
- **Timer support** — Configurable exam duration with audio countdown warnings.
- **Multi-language support** — TTS/STT in regional languages.
- **Admin dashboard** — For staff to manage question papers and view submitted answers.
- **Database integration** — Replace JSON files with a proper database (SQLite/PostgreSQL).
- **Answer export** — Download answers as PDF for evaluation.
- **Authentication** — Student login with roll number verification.

---

See **BUILD_PLAN.md** for the step-by-step development plan.
