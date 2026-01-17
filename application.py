import os
import time
import json
from uuid import uuid4

import cv2
from deepface import DeepFace

import requests
from flask import Flask, render_template, request, redirect, url_for, session, flash, send_from_directory
from werkzeug.utils import secure_filename


try:
    import ollama
except Exception:
    ollama = None


# Ρυθμίζουμε Flask ώστε:
# - templates να διαβάζονται από τον φάκελο "templates"
# - static αρχεία (CSS) να σερβίρονται από τον φάκελο "css"
# Το static_url_path="/css" σημαίνει ότι το CSS θα φορτώνεται από /css/...
app = Flask(__name__, template_folder="templates", static_folder="css", static_url_path="/css")

# Καλύτερα να έρχεται από env (Render Environment Variables)
app.secret_key = os.getenv("SECRET_KEY", "change_this_secret_key")

app.config["UPLOAD_FOLDER"] = os.path.join(os.getcwd(), "uploads")
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10MB

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)


# -----------------------------
# Ρυθμίσεις LLM Provider
# -----------------------------
# Στο Render: LLM_PROVIDER=groq
# Τοπικά: LLM_PROVIDER=ollama (αν υπάρχει Ollama)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").strip().lower()

# Groq (OpenAI-compatible)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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").strip()
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant").strip() 
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1").strip() 

# Ollama (local)
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").strip()
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3").strip()
ollama_client = None
if LLM_PROVIDER == "ollama" and ollama is not None:
    ollama_client = ollama.Client(host=OLLAMA_HOST)


def allowed_file(filename: str) -> bool:
    # Ελέγχουμε την κατάληξη του αρχείου
    if "." not in filename:
        return False
    ext = filename.rsplit(".", 1)[1].lower().strip()
    return ext in ALLOWED_EXTENSIONS


def cleanup_old_uploads(folder: str, max_age_sec: int = 3600):
    # Σβήνουμε παλιά uploads για να μη γεμίζει ο δίσκος
    # Δεν σβήνουμε αμέσως τα νέα αρχεία, γιατί τα χρειάζεται το chat για προβολή
    now = time.time()
    try:
        for name in os.listdir(folder):
            path = os.path.join(folder, name)
            if os.path.isfile(path):
                age = now - os.path.getmtime(path)
                if age > max_age_sec:
                    os.remove(path)
    except Exception:
        pass


def build_detected_objects_payload(detections):
    # Φτιάχνουμε μικρό JSON payload για να μην βαραίνει το prompt
    objects = []
    for d in detections:
        x1, y1, x2, y2 = d.get("bbox", (0, 0, 0, 0))
        objects.append({
            "type": "face",
            "emotion": d.get("label", "unknown"),
            "confidence": round(float(d.get("confidence", 0.0)), 3),
            "bbox": [int(x1), int(y1), int(x2), int(y2)]
        })
    return objects


def build_prompt(objects, prompt_style="focus_emotion"):
    # Prompts στα Αγγλικά για να ταιριάζουν με labels τύπου happy/surprise/etc
    objects_json = json.dumps(objects, ensure_ascii=False)

    if prompt_style == "bullet_then_summary":
        return f"""
I will give you detections.
First write 2 bullet points with facts (e.g., number of faces, dominant emotion).
Then write ONE short summary sentence in English.
No guesses. No background. No extra objects.
Repeat emotion labels exactly as given.

Detections:
{objects_json}
""".strip()

    if prompt_style == "focus_emotion":
        return f"""
You will receive detections from computer vision.
Write a short image description (2-3 sentences) in English with emphasis on emotion.

Rules:
- Use ONLY the provided detections.
- Do NOT invent background, context, or extra objects.
- Do NOT mention technical terms (bbox, json, confidence).
- Mention how many faces are present.
- Mention the dominant emotion(s) and repeat emotion labels EXACTLY as given (e.g., "surprise", "happy").
- Do not translate the emotion labels into other words.

Detections:
{objects_json}
""".strip()

    return f"""
You are an image description assistant.
I will give you computer vision detections (objects).
Write ONE short paragraph (2-3 sentences) in English.

Rules:
- Use ONLY the provided detections.
- Do NOT invent background, context, or extra objects.
- Do NOT mention technical terms (bbox, json, confidence).
- You may mention how many faces are present and the dominant emotion(s).
- If an emotion is provided, repeat it EXACTLY as given (e.g., "surprise", "happy").

Detected objects:
{objects_json}
""".strip()


def run_image_mode(image_path, detector_backend="opencv"):
    # Κάνει DeepFace emotion analysis και επιστρέφει detections
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Δεν ήταν δυνατή η ανάγνωση της εικόνας.")

    result = DeepFace.analyze(
        img_path=img,
        actions=["emotion"],
        detector_backend=detector_backend,
        enforce_detection=False
    )

    faces = result if isinstance(result, list) else [result]
    detections = []

    for f in faces:
        region = f.get("region", {}) or {}
        x = int(region.get("x", 0))
        y = int(region.get("y", 0))
        w = int(region.get("w", 0))
        h = int(region.get("h", 0))

        emotions = f.get("emotion", {}) or {}
        if emotions:
            label = max(emotions, key=emotions.get)
            confidence = float(emotions.get(label, 0.0)) / 100.0
        else:
            label = "unknown"
            confidence = 0.0

        detections.append({
            "bbox": (x, y, x + w, y + h),
            "label": label,
            "confidence": confidence
        })

    return detections


def groq_generate(prompt: str) -> str:
    # Καλεί Groq OpenAI-compatible endpoint: /chat/completions
    if not GROQ_API_KEY:
        return "GROQ_API_KEY is missing on the server."

    url = f"{GROQ_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3,
        "max_tokens": 160
    }

    try:
        r = requests.post(url, headers=headers, json=payload, timeout=60)
    except Exception as e:
        return f"Groq request failed: {str(e)}"

    if r.status_code != 200:
        # Κόβουμε το μήνυμα για να μην έχουμε ογκώδεις απαντήσεις
        return f"Groq API error ({r.status_code}): {r.text[:250]}"

    data = r.json()
    try:
        text = data["choices"][0]["message"]["content"].strip()
        return text if text else "The LLM returned an empty response."
    except Exception:
        return "Unexpected Groq response format."


def llm_generate_description(detections, prompt_style="focus_emotion"):
    # Ενιαία συνάρτηση LLM: Groq (deployment) ή Ollama (local)
    objects = build_detected_objects_payload(detections)

    if len(objects) == 0:
        return "No reliable detections were found, so I cannot produce a confident description."

    prompt = build_prompt(objects, prompt_style=prompt_style)

    if LLM_PROVIDER == "ollama":
        if ollama_client is None:
            return "LLM_PROVIDER=ollama but Ollama client is not available on this server."
        resp = ollama_client.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.3, "num_predict": 160},
            stream=False
        )
        text = (resp.get("message", {}) or {}).get("content", "").strip()
        return text if text else "The LLM returned an empty response."

    # Default: Groq
    return groq_generate(prompt)


def push_message(role, content, detections=None, image_url=None, latency=None):
    # Κρατάμε chat ιστορικό στο session
    if "chat" not in session:
        session["chat"] = []

    session["chat"].append({
        "role": role,
        "content": content,
        "detections": detections or [],
        "image_url": image_url,
        "latency": latency
    })
    session.modified = True


@app.route("/", methods=["GET"])
def index():
    # Το template μας λέγεται webpage.html
    chat_hist = session.get("chat", [])
    return render_template("webpage.html", chat=chat_hist)


@app.route("/clear", methods=["POST"])
def clear_chat():
    # Καθαρισμός ιστορικού
    session["chat"] = []
    session.modified = True
    return redirect(url_for("index"))


@app.route("/chat", methods=["POST"])
def chat():
    # Upload και μετά redirect σε loading screen
    prompt_style = request.form.get("prompt_style", "focus_emotion").strip()
    detector_backend = request.form.get("detector_backend", "opencv").strip()

    file = request.files.get("image")
    if file is None or file.filename is None or file.filename.strip() == "":
        flash("Δεν επιλέχθηκε αρχείο εικόνας.", "error")
        return redirect(url_for("index"))

    if not allowed_file(file.filename):
        flash("Μη έγκυρος τύπος αρχείου. Δεκτά: png/jpg/jpeg/webp.", "error")
        return redirect(url_for("index"))

    safe_name = secure_filename(file.filename)
    unique_name = f"{uuid4().hex}_{safe_name}"
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)

    try:
        file.save(save_path)
    except Exception as e:
        flash(f"Αποτυχία αποθήκευσης αρχείου: {e}", "error")
        return redirect(url_for("index"))

    # Κάνουμε cleanup παλιών uploads
    cleanup_old_uploads(app.config["UPLOAD_FOLDER"], max_age_sec=3600)

    # Προσθέτουμε user message στο chat (μόνο filename)
    img_url = url_for("uploaded_file", filename=unique_name)
    push_message(role="user", content=safe_name, image_url=img_url)

    # Αποθηκεύουμε “pending job” στο session ώστε να το επεξεργαστεί το /process
    session["pending_upload"] = {
        "filename": unique_name,
        "prompt_style": prompt_style,
        "detector_backend": detector_backend
    }
    session.modified = True

    return redirect(url_for("loading"))


@app.route("/loading", methods=["GET"])
def loading():
    # Εμφανίζει σελίδα φόρτωσης (οπτικό feedback)
    if not session.get("pending_upload"):
        return redirect(url_for("index"))
    return render_template("loading.html")


@app.route("/process", methods=["GET"])
def process():
    # Τρέχει inference + LLM μετά το loading screen
    pending = session.get("pending_upload")
    if not pending:
        return redirect(url_for("index"))

    t0 = time.time()

    prompt_style = pending.get("prompt_style", "focus_emotion")
    detector_backend = pending.get("detector_backend", "opencv")
    filename = pending.get("filename")

    if not filename:
        session.pop("pending_upload", None)
        session.modified = True
        flash("Δεν βρέθηκε το αρχείο προς επεξεργασία.", "error")
        return redirect(url_for("index"))

    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)

    try:
        detections = run_image_mode(save_path, detector_backend=detector_backend)
        llm_text = llm_generate_description(detections, prompt_style=prompt_style)
        latency = round(float(time.time() - t0), 2)

        push_message(role="bot", content=llm_text, detections=detections, latency=latency)

        if latency > 25:
            flash("Σημείωση: Αργή επεξεργασία. Δοκίμασε μικρότερη εικόνα ή άλλο detector.", "warn")

    except Exception as e:
        push_message(role="bot", content=f"Processing error: {str(e)}")
        flash("Παρουσιάστηκε σφάλμα στην επεξεργασία της εικόνας.", "error")

    finally:
        session.pop("pending_upload", None)
        session.modified = True

    return redirect(url_for("index"))


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    # Σερβίρουμε την ανεβασμένη εικόνα για να φαίνεται στο chat
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    print("RUNNING FROM:", os.getcwd())
    app.run(host="0.0.0.0", port=port, debug=True)

