"""
Accessible Exam Tool — Flask backend
Blind mode with OCR, file-based exam storage, and client-side TTS/STT
"""

from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
from pypdf import PdfReader
import os
import re
import json
import uuid
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = "accessible-exam-secret-key-change-in-production"

UPLOAD_FOLDER = "uploads"
EXAM_FOLDER = "exam_data"
IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp", "tiff", "gif"}
PDF_EXTENSIONS = {"pdf"}
ALLOWED_EXTENSIONS = IMAGE_EXTENSIONS | PDF_EXTENSIONS

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(EXAM_FOLDER, exist_ok=True)


# -------------------------------------------------
# EXAM DATA STORAGE  (JSON files on disk)
# -------------------------------------------------
def save_exam_data(instructions, questions, answers=None):
    """Save exam data to a JSON file and return its ID."""
    exam_id = str(uuid.uuid4())[:8]
    data = {
        "instructions": instructions,
        "questions": questions,
        "answers": answers or {},
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
    """Check if the file has an allowed extension (image or PDF)."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def get_file_ext(filename):
    """Return the lowercase file extension."""
    return filename.rsplit(".", 1)[1].lower() if "." in filename else ""


def preprocess_image(image):
    """Enhance image for better OCR accuracy."""
    image = image.convert("L")
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2.0)
    image = image.filter(ImageFilter.SHARPEN)
    return image


def extract_text_from_image(image_path):
    """Extract text from an image with preprocessing."""
    img = Image.open(image_path)
    img = preprocess_image(img)
    text = pytesseract.image_to_string(img)
    return text.strip()


def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file.

    When a page contains embedded images a warning is inserted.
    """
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
    """Extract text from either an image or a PDF."""
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
    """If a question mentions a figure / diagram, append a spoken warning."""
    if IMAGE_KEYWORDS.search(text):
        return (
            text
            + " [NOTE: This question refers to a visual element such as a "
            "diagram or figure. Please ask your invigilator to describe it.]"
        )
    return text


def split_sections(text):
    """Separate instructions from numbered questions."""
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


# -------------------------------------------------
# ROUTES
# -------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


# ---- Blind Mode: Upload + OCR ------------------
@app.route("/blind", methods=["GET", "POST"])
def blind():
    extracted_text = None
    error = None
    num_instructions = 0
    num_questions = 0

    if request.method == "POST" and "question_paper" in request.files:
        file = request.files["question_paper"]

        if file.filename == "":
            error = "No file selected."
        elif not allowed_file(file.filename):
            error = (
                "Unsupported file type. Allowed: PDF, "
                + ", ".join(sorted(IMAGE_EXTENSIONS)).upper()
                + "."
            )
        else:
            filename = secure_filename(file.filename)
            path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(path)

            try:
                extracted_text = extract_text(path)
                if not extracted_text:
                    error = "Could not extract any text. Try a clearer image or text-based PDF."
                else:
                    instructions, questions = split_sections(extracted_text)
                    num_instructions = len(instructions)
                    num_questions = len(questions)

                    # Save to disk (not cookie) and store only the ID in session
                    exam_id = save_exam_data(instructions, questions)
                    session["exam_id"] = exam_id
                    session["num_instructions"] = num_instructions
                    session["num_questions"] = num_questions
            except Exception as e:
                error = f"Error processing file: {e}"

    return render_template(
        "blind.html",
        extracted_text=extracted_text,
        error=error,
        num_instructions=num_instructions or session.get("num_instructions", 0),
        num_questions=num_questions or session.get("num_questions", 0),
    )


# ---- Blind Mode: Exam Page ---------------------
@app.route("/blind-exam")
def blind_exam():
    exam_id = session.get("exam_id")
    data = load_exam_data(exam_id)

    if data is None:
        return redirect(url_for("blind"))

    return render_template(
        "blind_exam.html",
        instructions=data["instructions"],
        questions=data["questions"],
        answers=data.get("answers", {}),
        total_questions=len(data["questions"]),
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
def completed():
    exam_id = session.get("exam_id")
    data = load_exam_data(exam_id)
    if data is None:
        return redirect(url_for("blind"))
    return render_template(
        "completed.html",
        answers=data.get("answers", {}),
        questions=data["questions"],
        total=len(data["questions"]),
    )


# ---- Reset / New Exam --------------------------
@app.route("/reset")
def reset():
    session.clear()
    return redirect(url_for("blind"))


# ---- Deaf & Mute Mode (placeholder) ------------
@app.route("/deaf")
def deaf():
    return render_template("deaf.html")


if __name__ == "__main__":
    app.run(debug=True, port=5000)
