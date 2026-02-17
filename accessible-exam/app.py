"""
Accessible Exam Tool — Flask backend
Teacher/Student login, paper management, and voice exam with OCR + TTS/STT
"""

from flask import Flask, render_template, request, redirect, url_for, session, jsonify, flash, Response
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
from pypdf import PdfReader
import os
import re
import json
import uuid
from datetime import datetime
from functools import wraps
from werkzeug.utils import secure_filename
import cv2
try:
    from gesture_model import GestureModel
except ImportError:
    GestureModel = None
try:
    from word_model import WordModel
except ImportError:
    WordModel = None


app = Flask(__name__)
app.secret_key = "accessible-exam-secret-key-change-in-production"

UPLOAD_FOLDER = "uploads"
EXAM_FOLDER = "exam_data"
DATA_FOLDER = "data"
IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp", "tiff", "gif"}
PDF_EXTENSIONS = {"pdf"}
ALLOWED_EXTENSIONS = IMAGE_EXTENSIONS | PDF_EXTENSIONS

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(EXAM_FOLDER, exist_ok=True)
os.makedirs(DATA_FOLDER, exist_ok=True)

# -------------------------------------------------
# USER & PAPER DATA  (JSON files on disk)
# -------------------------------------------------
USERS_FILE = os.path.join(DATA_FOLDER, "users.json")
PAPERS_FILE = os.path.join(DATA_FOLDER, "papers.json")


def _load_json(path, default=None):
    if not os.path.exists(path):
        return default if default is not None else {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_users():
    return _load_json(USERS_FILE, default={"teachers": {}, "students": {}})


def save_users(users):
    _save_json(USERS_FILE, users)


def get_papers():
    return _load_json(PAPERS_FILE, default=[])


def save_papers(papers):
    _save_json(PAPERS_FILE, papers)


# Create default users if none exist (auto-generated on startup)
def init_default_users():
    users = get_users()
    if not users.get("teachers") or not users.get("blind_students") or not users.get("deaf_students"):
        users = {
            "teachers": {
                "teacher1": {"password": "teach123", "name": "Mrs. Priya"},
                "teacher2": {"password": "teach456", "name": "Mr. Kumar"},
            },
            "blind_students": {
                "CSE001": {"dob": "2004-05-15", "name": "Ananya"},
                "CSE002": {"dob": "2004-08-22", "name": "Rahul"},
            },
            "deaf_students": {
                "CSE003": {"dob": "2004-01-10", "name": "Divya"},
                "CSE004": {"dob": "2003-12-25", "name": "Karthik"},
            },
        }
        save_users(users)
    return users


init_default_users()


# -------------------------------------------------
# AUTH DECORATORS
# -------------------------------------------------
def teacher_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if session.get("role") != "teacher":
            flash("Please log in as a teacher.", "error")
            return redirect(url_for("teacher_login"))
        return f(*args, **kwargs)
    return decorated


def student_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if session.get("role") != "student":
            flash("Please log in as a student.", "error")
            return redirect(url_for("student_login"))
        return f(*args, **kwargs)
    return decorated


# -------------------------------------------------
# EXAM DATA STORAGE  (JSON files on disk)
# -------------------------------------------------
def save_exam_data(instructions, questions, answers=None, paper_id=None, student_id=None):
    """Save exam data to a JSON file and return its ID."""
    exam_id = str(uuid.uuid4())[:8]
    data = {
        "instructions": instructions,
        "questions": questions,
        "answers": answers or {},
        "paper_id": paper_id,
        "student_id": student_id,
        "started_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    path = os.path.join(EXAM_FOLDER, f"{exam_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    return exam_id


def load_exam_data(exam_id):
    """Load exam data from disk. Returns dict or None."""
    if not exam_id:
        return None
    path = os.path.join(EXAM_FOLDER, f"{exam_id}.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def update_exam_answers(exam_id, answers):
    """Update only the answers in an existing exam file."""
    data = load_exam_data(exam_id)
    if data is None:
        return False
    data["answers"] = answers
    path = os.path.join(EXAM_FOLDER, f"{exam_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    return True


# -------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def get_file_ext(filename):
    return filename.rsplit(".", 1)[1].lower() if "." in filename else ""


def preprocess_image(image):
    image = image.convert("L")
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2.0)
    image = image.filter(ImageFilter.SHARPEN)
    return image


def extract_text_from_image(image_path):
    img = Image.open(image_path)
    img = preprocess_image(img)
    text = pytesseract.image_to_string(img)
    return text.strip()


def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    pages_text = []
    for page_num, page in enumerate(reader.pages, start=1):
        page_text = (page.extract_text() or "").strip()
        has_images = len(page.images) > 0 if hasattr(page, "images") else False
        if has_images:
            page_text += (
                "\n[NOTE: This page contains a diagram or image that cannot "
                "be read aloud. Please ask your invigilator to describe it.]"
            )
        if page_text:
            pages_text.append(page_text)
    return "\n".join(pages_text).strip()


def extract_text(file_path):
    ext = get_file_ext(file_path)
    if ext in PDF_EXTENSIONS:
        return extract_text_from_pdf(file_path)
    return extract_text_from_image(file_path)


IMAGE_KEYWORDS = re.compile(
    r"\b(figure|fig\.|diagram|graph|chart|table|image|picture|drawing|"
    r"illustration|map|sketch|refer\s+to\s+the\s+(figure|diagram|image))\b",
    re.IGNORECASE,
)


def flag_visual_references(text):
    if IMAGE_KEYWORDS.search(text):
        return (
            text
            + " [NOTE: This question refers to a visual element such as a "
            "diagram or figure. Please ask your invigilator to describe it.]"
        )
    return text


def split_sections(text):
    lines = text.split("\n")
    instructions = []
    questions = []
    is_instruction = True
    current_question = ""

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if re.match(r"^\d+[\.\)\:\s]", line):
            is_instruction = False
            if current_question:
                questions.append(flag_visual_references(current_question.strip()))
            current_question = line
        elif not is_instruction:
            current_question += " " + line
        else:
            instructions.append(line)

    if current_question:
        questions.append(flag_visual_references(current_question.strip()))

    return instructions, questions


# =====================================================
# ROUTES — HOME
# =====================================================
@app.route("/")
def index():
    # If already logged in, redirect to appropriate dashboard
    if session.get("role") == "teacher":
        return redirect(url_for("teacher_dashboard"))
    if session.get("role") == "student":
        return redirect(url_for("student_dashboard"))
    return render_template("index.html")


# =====================================================
# ROUTES — TEACHER
# =====================================================
@app.route("/teacher/login", methods=["GET", "POST"])
def teacher_login():
    if session.get("role") == "teacher":
        return redirect(url_for("teacher_dashboard"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        users = get_users()
        teacher = users["teachers"].get(username)

        if teacher and teacher["password"] == password:
            session["role"] = "teacher"
            session["username"] = username
            session["name"] = teacher["name"]
            return redirect(url_for("teacher_dashboard"))
        else:
            flash("Invalid teacher username or password.", "error")

    return redirect(url_for("index"))


@app.route("/teacher/dashboard")
@teacher_required
def teacher_dashboard():
    papers = get_papers()
    # Show only papers uploaded by this teacher
    my_papers = [p for p in papers if p.get("teacher") == session["username"]]
    return render_template("teacher_dashboard.html", papers=my_papers)


@app.route("/teacher/upload", methods=["POST"])
@teacher_required
def teacher_upload():
    if "question_paper" not in request.files:
        flash("No file uploaded.", "error")
        return redirect(url_for("teacher_dashboard"))

    file = request.files["question_paper"]
    subject = request.form.get("subject", "").strip() or "Untitled"
    exam_mode = request.form.get("exam_mode", "voice").strip()

    if file.filename == "":
        flash("No file selected.", "error")
        return redirect(url_for("teacher_dashboard"))

    if not allowed_file(file.filename):
        flash(
            "Unsupported file type. Allowed: PDF, "
            + ", ".join(sorted(IMAGE_EXTENSIONS)).upper()
            + ".",
            "error",
        )
        return redirect(url_for("teacher_dashboard"))

    filename = secure_filename(file.filename)
    path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(path)

    try:
        extracted_text = extract_text(path)
        if not extracted_text:
            flash("Could not extract any text. Try a clearer image or text-based PDF.", "error")
            return redirect(url_for("teacher_dashboard"))

        instructions, questions = split_sections(extracted_text)
        num_instructions = len(instructions)
        num_questions = len(questions)

        paper_id = str(uuid.uuid4())[:8]
        exam_id = save_exam_data(instructions, questions, paper_id=paper_id)

        paper_entry = {
            "paper_id": paper_id,
            "exam_id": exam_id,
            "subject": subject,
            "exam_mode": exam_mode,
            "filename": filename,
            "teacher": session["username"],
            "teacher_name": session.get("name", ""),
            "num_questions": num_questions,
            "num_instructions": num_instructions,
            "uploaded_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "status": "active",
        }

        papers = get_papers()
        papers.append(paper_entry)
        save_papers(papers)

        flash(f"Paper uploaded! {num_questions} questions extracted for {subject}.", "success")

    except Exception as e:
        flash(f"Error processing file: {e}", "error")

    return redirect(url_for("teacher_dashboard"))


@app.route("/teacher/delete-paper/<paper_id>", methods=["POST"])
@teacher_required
def delete_paper(paper_id):
    papers = get_papers()
    papers = [p for p in papers if not (p["paper_id"] == paper_id and p["teacher"] == session["username"])]
    save_papers(papers)
    flash("Paper deleted.", "success")
    return redirect(url_for("teacher_dashboard"))


# =====================================================
# ROUTES — STUDENT
# =====================================================
@app.route("/student/login", methods=["GET", "POST"])
def student_login():
    if session.get("role") == "student":
        return redirect(url_for("student_dashboard"))

    if request.method == "POST":
        roll_number = request.form.get("roll_number", "").strip().upper()
        dob = request.form.get("dob", "").strip()
        exam_type = request.form.get("exam_type", "voice").strip()
        users = get_users()

        # Look up in the correct student group
        if exam_type == "voice":
            student = users.get("blind_students", {}).get(roll_number)
        else:
            student = users.get("deaf_students", {}).get(roll_number)

        if student and student["dob"] == dob:
            session["role"] = "student"
            session["username"] = roll_number
            session["name"] = student["name"]
            session["roll"] = roll_number
            session["student_mode"] = exam_type
            return redirect(url_for("student_dashboard"))
        else:
            if exam_type == "voice":
                flash("Invalid roll number or date of birth for Blind Students.", "error")
            else:
                flash("Invalid roll number or date of birth for Deaf & Mute Students.", "error")

    return redirect(url_for("index"))


@app.route("/student/dashboard")
@student_required
def student_dashboard():
    papers = get_papers()
    student_mode = session.get("student_mode", "voice")
    # Show only active papers that match this student's mode (or "both")
    active_papers = [
        p for p in papers
        if p.get("status") == "active"
        and (p.get("exam_mode") == student_mode or p.get("exam_mode") == "both")
    ]
    return render_template("student_dashboard.html", papers=active_papers, student_mode=student_mode)


@app.route("/student/start-exam/<paper_id>")
@student_required
def start_exam(paper_id):
    papers = get_papers()
    paper = next((p for p in papers if p["paper_id"] == paper_id and p.get("status") == "active"), None)

    if not paper:
        flash("Paper not found or no longer available.", "error")
        return redirect(url_for("student_dashboard"))

    # Load the master exam data (from teacher upload)
    master_data = load_exam_data(paper["exam_id"])
    if master_data is None:
        flash("Exam data not found.", "error")
        return redirect(url_for("student_dashboard"))

    # Create a new exam session for this student
    student_exam_id = save_exam_data(
        instructions=master_data["instructions"],
        questions=master_data["questions"],
        paper_id=paper_id,
        student_id=session["username"],
    )

    session["exam_id"] = student_exam_id
    session["current_paper_subject"] = paper["subject"]

    if session.get("student_mode") == "visual":
         return redirect(url_for("deaf_exam"))
    
    return redirect(url_for("blind_exam"))


# =====================================================
# ROUTES — EXAM (voice exam pages)
# =====================================================
@app.route("/blind-exam")
@student_required
def blind_exam():
    exam_id = session.get("exam_id")
    data = load_exam_data(exam_id)

    if data is None:
        return redirect(url_for("student_dashboard"))

    return render_template(
        "blind_exam.html",
        instructions=data["instructions"],
        questions=data["questions"],
        answers=data.get("answers", {}),
        total_questions=len(data["questions"]),
        subject=session.get("current_paper_subject", "Exam"),
    )


# ---- API: Save a single answer (AJAX) ----------
@app.route("/api/save-answer", methods=["POST"])
def save_answer():
    req = request.get_json()
    if not req:
        return jsonify({"error": "No data provided"}), 400

    question_index = req.get("question_index")
    answer_text = req.get("answer")

    if question_index is None or answer_text is None:
        return jsonify({"error": "Missing question_index or answer"}), 400

    exam_id = session.get("exam_id")
    data = load_exam_data(exam_id)
    if data is None:
        return jsonify({"error": "Exam not found"}), 404

    data["answers"][str(question_index)] = answer_text
    update_exam_answers(exam_id, data["answers"])

    return jsonify(
        {
            "success": True,
            "answered": len(data["answers"]),
            "total": len(data["questions"]),
        }
    )


# ---- API: Retrieve all saved answers -----------
@app.route("/api/get-answers")
def get_answers():
    exam_id = session.get("exam_id")
    data = load_exam_data(exam_id)
    if data is None:
        return jsonify({"answers": {}, "questions": []})
    return jsonify(
        {"answers": data.get("answers", {}), "questions": data["questions"]}
    )


# ---- Completion Page ----------------------------
@app.route("/completed")
@student_required
def completed():
    exam_id = session.get("exam_id")
    data = load_exam_data(exam_id)
    if data is None:
        return redirect(url_for("student_dashboard"))
    return render_template(
        "completed.html",
        answers=data.get("answers", {}),
        questions=data["questions"],
        total=len(data["questions"]),
        subject=session.get("current_paper_subject", "Exam"),
    )


# ---- Logout ------------------------------------
@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))


# ---- Legacy reset (now goes to student dashboard) ----
@app.route("/reset")
def reset():
    exam_id = session.get("exam_id")
    if exam_id:
        session.pop("exam_id", None)
    if session.get("role") == "student":
        return redirect(url_for("student_dashboard"))
    return redirect(url_for("index"))


# ---- Deaf & Mute Mode (placeholder) ------------
@app.route("/deaf")
def deaf():
    return render_template("deaf.html")


# =====================================================
# ROUTES — DEAF EXAM (Sign Language Logic)
# =====================================================
import base64
import numpy as np

gesture_recognizer = None
word_model_recognizer = None

def get_gesture_model():
    global gesture_recognizer
    if gesture_recognizer is None and GestureModel:
        gesture_recognizer = GestureModel()
    return gesture_recognizer

def get_word_model():
    global word_model_recognizer
    if word_model_recognizer is None and WordModel:
        word_model_recognizer = WordModel()
        # If word model's hand detector failed to init, reuse gesture model's (same MediaPipe)
        gm = get_gesture_model()
        if gm and getattr(gm, "detector", None) and getattr(word_model_recognizer, "detector", None) is None:
            word_model_recognizer.set_detector(gm.detector)
    return word_model_recognizer

@app.route('/sign_models_status', methods=['GET'])
def sign_models_status():
    """Return whether word model is loaded so the UI can show 'words + letters' vs 'letters only'."""
    wm = get_word_model()
    word_loaded = wm is not None and getattr(wm, 'ml_model', None) is not None
    return jsonify({'word_model_loaded': word_loaded})


# Word suggestions for spell-and-autocomplete (keyboard auto-correct style)
_WORD_SUGGESTIONS_CACHE = None
_PHRASE_SUGGESTIONS_CACHE = None

def _get_word_suggestions_vocab():
    global _WORD_SUGGESTIONS_CACHE
    if _WORD_SUGGESTIONS_CACHE is None:
        path = os.path.join(DATA_FOLDER, "word_suggestions.json")
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                _WORD_SUGGESTIONS_CACHE = json.load(f)
        else:
            _WORD_SUGGESTIONS_CACHE = []
    return _WORD_SUGGESTIONS_CACHE


def _get_phrase_suggestions():
    global _PHRASE_SUGGESTIONS_CACHE
    if _PHRASE_SUGGESTIONS_CACHE is None:
        path = os.path.join(DATA_FOLDER, "phrase_suggestions.json")
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                _PHRASE_SUGGESTIONS_CACHE = json.load(f)
        else:
            _PHRASE_SUGGESTIONS_CACHE = {}
    return _PHRASE_SUGGESTIONS_CACHE


@app.route('/word_suggestions')
def word_suggestions():
    """Return suggestions. ?prefix=hel -> prefix match. ?context=help+me -> next-word suggestions."""
    prefix = (request.args.get("prefix") or "").lower().strip()
    context = (request.args.get("context") or "").lower().strip()

    # Context-based: last 2-3 words -> suggest next words
    if context:
        words = context.split()
        phrase_data = _get_phrase_suggestions()
        # Try longest match first: "help me" then "help" then ""
        for n in range(min(3, len(words)), 0, -1):
            phrase = " ".join(words[-n:])
            if phrase in phrase_data:
                return jsonify({"suggestions": phrase_data[phrase][:8]})
        return jsonify({"suggestions": []})

    # Prefix-based: current word being spelled
    if not prefix:
        return jsonify({"suggestions": []})
    vocab = _get_word_suggestions_vocab()
    matches = [w for w in vocab if w.startswith(prefix)][:8]
    return jsonify({"suggestions": matches})


@app.route('/process_gesture', methods=['POST'])
def process_gesture():
    """
    Receives base64 image(s) from the client, detects gestures.
    Single image: returns one prediction.
    Multiple images: returns majority vote (more accurate for letter capture).
    """
    model = get_gesture_model()
    if not model:
        return jsonify({'gesture': None, 'error': 'Model not loaded'})

    data = request.json or {}
    images_b64 = data.get('images') or ([data.get('image')] if data.get('image') else [])

    if not images_b64:
        return jsonify({'gesture': None})

    try:
        frames = []
        for item in images_b64[:9]:
            if not item:
                continue
            s = item if isinstance(item, str) else ''
            if ',' in s:
                s = s.split(',', 1)[1]
            raw = base64.b64decode(s)
            arr = np.frombuffer(raw, np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is not None:
                frames.append(frame)

        if not frames:
            return jsonify({'gesture': None, 'error': 'Invalid image(s)'})

        if len(frames) >= 5:
            gesture = model.process_frames_vote(frames, min_votes=3)
        elif len(frames) >= 2:
            gesture = model.process_frames_vote(frames, min_votes=2)
        else:
            gesture = model.process_frame(frames[0])
        letter, conf = model.process_frame_with_confidence(frames[0])
        out = {'gesture': gesture, 'confidence': round(conf, 3)}
        return jsonify(out)
    except Exception as e:
        print(f"Error processing gesture: {e}")
        return jsonify({'gesture': None})


@app.route('/process_word', methods=['POST'])
def process_word():
    """
    Receives a list of base64 images (short video as frames). Extracts hand sequences,
    runs word model (WLASL-style), returns recognized word or null.
    Body: { "frames": [ "data:image/jpeg;base64,...", ... ] }
    """
    model = get_word_model()
    if not model or not getattr(model, 'ml_model', None):
        return jsonify({'word': None, 'error': 'Word model not loaded. Run train_word_model.py after WLASL extraction.'})

    data = request.json
    frames_b64 = data.get('frames') if data else None
    if not frames_b64 or not isinstance(frames_b64, list) or len(frames_b64) < 3:
        return jsonify({'word': None})

    frames = []
    for item in frames_b64[:60]:  # cap at 60 frames
        try:
            s = item if isinstance(item, str) else ''
            if ',' in s:
                s = s.split(',', 1)[1]
            raw = base64.b64decode(s)
            arr = np.frombuffer(raw, np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is not None:
                frames.append(frame)
        except Exception:
            continue
    if len(frames) < 3:
        return jsonify({'word': None})
    try:
        word = model.process_sequence(frames)
        return jsonify({'word': word})
    except Exception as e:
        print('Error processing word:', e)
        return jsonify({'word': None})


@app.route('/process_sign', methods=['POST'])
def process_sign():
    """
    Unified capture: accepts frames from a short recording.
    Body: { "frames": [...], "mode": "letter" | "number" | null }
    - mode "letter": only return A–Z (single letter).
    - mode "number": only return 0–9 (single digit).
    - no mode: try word model first, then letter/digit.
    """
    data = request.json or {}
    frames_b64 = data.get('frames')
    mode = data.get('mode')  # 'letter' | 'number' | None

    if not frames_b64 or not isinstance(frames_b64, list):
        return jsonify({'sign': None})

    frames = []
    for item in frames_b64[:60]:
        try:
            s = item if isinstance(item, str) else ''
            if ',' in s:
                s = s.split(',', 1)[1]
            raw = base64.b64decode(s)
            arr = np.frombuffer(raw, np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is not None:
                frames.append(frame)
        except Exception:
            continue

    # Word model only when not restricting to letter/number
    if not mode and len(frames) >= 3:
        word_model = get_word_model()
        if word_model and getattr(word_model, 'ml_model', None):
            try:
                word = word_model.process_sequence(frames)
                if word:
                    return jsonify({'sign': word, 'source': 'word'})
            except Exception as e:
                print('Word model error:', e)

    gesture_model = get_gesture_model()
    if not gesture_model or not frames:
        return jsonify({'sign': None})

    if len(frames) >= 5:
        gesture = gesture_model.process_frames_vote(frames, min_votes=3)
    elif len(frames) >= 2:
        gesture = gesture_model.process_frames_vote(frames, min_votes=2)
    else:
        gesture = gesture_model.process_frame(frames[0])

    if not gesture:
        return jsonify({'sign': None})

    # Filter by mode: letter -> only A–Z, number -> only 0–9
    if mode == 'letter':
        if len(gesture) == 1 and gesture.isalpha():
            return jsonify({'sign': gesture.upper(), 'source': 'letter'})
        return jsonify({'sign': None})
    if mode == 'number':
        if len(gesture) == 1 and gesture.isdigit():
            return jsonify({'sign': gesture, 'source': 'number'})
        return jsonify({'sign': None})

    return jsonify({'sign': gesture, 'source': 'letter'})


@app.route("/deaf-exam")
@student_required
def deaf_exam():
    exam_id = session.get("exam_id")
    data = load_exam_data(exam_id)

    if data is None:
        return redirect(url_for("student_dashboard"))

    # Convert question strings to objects for the template
    # Template expects q.id and q.text
    raw_questions = data["questions"]
    question_objs = []
    for i, q_text in enumerate(raw_questions):
        question_objs.append({"id": i + 1, "text": q_text})

    return render_template(
        "deaf_exam.html",
        questions=question_objs,
        answers=data.get("answers", {}),
        subject=session.get("current_paper_subject", "Exam"),
    )

@app.route("/submit_deaf_answer", methods=["POST"])
def submit_deaf_answer():
    req = request.get_json()
    q_id = req.get("question_id") # 1-based index from template
    ans = req.get("answer")
    
    if not q_id or not ans:
        return jsonify({'status': 'error', 'msg': 'Missing data'}), 400

    exam_id = session.get("exam_id")
    data = load_exam_data(exam_id)
    if not data:
        return jsonify({'status': 'error', 'msg': 'Exam not found'}), 404

    # Convert 1-based ID to 0-based index string
    idx_str = str(int(q_id) - 1)
    
    data["answers"][idx_str] = ans
    update_exam_answers(exam_id, data["answers"])
    
    return jsonify({'status': 'success'})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
