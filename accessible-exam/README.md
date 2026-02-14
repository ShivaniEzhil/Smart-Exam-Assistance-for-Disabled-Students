# Accessible Exam Tool

## How to run

On macOS, use `pip3` and `python3`:

```bash
cd accessible-exam
pip3 install -r requirements.txt
python3 app.py
```

Then open **http://127.0.0.1:5000** in your browser.

### Using a virtual environment (optional)

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
```

### Notes

- The `exam_data/` folder contains JSON files that are automatically created per exam session. They store the extracted questions, instructions, and student answers. These files can be safely deleted after use.
- The `uploads/` folder stores uploaded question paper files (images/PDFs). These can also be cleared periodically.

See **BUILD_PLAN.md** for the step-by-step build plan.
