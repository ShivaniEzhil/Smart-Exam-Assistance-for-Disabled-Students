# Accessible Exam Tool — Step-by-Step Build Plan

## Big picture

**What we are building:** An AI-based Accessible Exam Tool where:
- Teacher uploads questions
- Student selects accessibility mode
- System adapts exam
- Student answers using voice / sign
- Answers are saved as text

**Focus for now:** Blind mode + Deaf & Mute mode only. (Autism mode can be added later.)

We build a **basic working prototype first**, then improve it.

---

## Step 1: Platform choice

**Use a Web Application.**

| Why | Reason |
|-----|--------|
| Works on laptop, tablet, PC | One codebase, everywhere |
| Easy to use camera & mic | Browser APIs |
| Easy to demo in exams | No install for students |

**Tech stack:**
- **Frontend:** HTML, CSS, JavaScript  
- **Backend:** Python (Flask)  
- **AI:** Python libraries  

---

## Step 2: Project structure

```
accessible-exam/
├── app.py                 # Backend (Flask)
├── templates/
│   ├── index.html         # Mode selection page
│   ├── blind.html
│   └── deaf.html
├── static/
│   └── style.css
```
*(autism.html can be added later)*

At this stage: **no AI, just pages.**

---

## Step 3: Mode selection page

- **index.html** → 2 buttons (for now):
  - Blind Mode  
  - Deaf & Mute Mode  
- Clicking a button opens the respective exam page.  
- **Goal:** Navigation working.  
- *(Autism mode: add later if needed.)*

---

## Step 4: Blind mode (Text → Audio, Voice → Text)

**What happens:**
- Question is read aloud  
- Student speaks answer  
- Voice is converted to text  

**How:**
- Store question as text  
- **Text-to-Speech** → pyttsx3  
- **Speech-to-Text** → SpeechRecognition  

**Result:** Blind student listens to question, answers by speaking, system saves answer as text.  
**First AI feature done.**

---

## Step 5: Deaf & Mute mode (Sign → Text)

**Prototype approach:**
- Use webcam  
- Detect hand gestures  
- Convert gestures → text (basic demo)  

**How:**
- Webcam: OpenCV  
- Hand detection: MediaPipe  
- If hand detected → “Sign captured”  
- Map a few gestures to words (demo)  

**For prototype:**  
- No full sign language needed  
- 3–5 gestures enough  
- Example: Open palm → “Yes”, Closed fist → “No”  

**Goal:** Prove sign-to-text concept.

---

## Step 6: All answers → TEXT

No matter the mode (voice, sign, typing), **final output must be TEXT.**

- One function: `process_answer(input) → text`  
- Makes evaluation and storage simple.

---

## Step 7: Store answers

**Prototype:** Save in `.txt` or `.csv`  

**Later (optional):** MySQL / SQLite  

**Example columns:** Student Name | Mode | Answer  

---

## Step 8: Teacher view (optional)

Teacher can:
- View submitted answers  
- See which mode each student used  

Makes the project complete and professional.

---

## Step 9: Test the system

- Test each mode: Blind → audio + voice; Deaf → camera + sign  
- Fix small issues  

---

## How everything works together

```
Teacher uploads exam
        ↓
Student selects accessibility mode
        ↓
AI adapts exam interface
        ↓
Student answers using voice/sign
        ↓
AI converts answer to text
        ↓
Answer stored & evaluated
```

---

## Viva / pitch line

*“We built a web-based AI examination tool that adapts itself based on student disability. Blind students hear questions and answer by voice; deaf students read questions and answer through sign language. All responses are converted into text for fair evaluation.”*
