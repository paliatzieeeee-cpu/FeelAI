# FeelAI — Emotion-Aware Image Description

FeelAI is a small Flask web app that looks at a photo, detects the faces in it,
recognises each face's emotion, and asks a Large Language Model to describe the
image in plain English based only on what was detected.

It combines two parts of AI in one pipeline:

```
uploaded image
   -> face detection (OpenCV, MTCNN or RetinaFace)
   -> emotion recognition per face (DeepFace, TensorFlow)
   -> structured detections (emotion label, confidence, bounding box)
   -> LLM prompt that restricts the model to those detections
   -> short description shown in a chat-style interface
```

## Features

- **Upload an image** (PNG, JPG, JPEG or WEBP, up to 10 MB) through a chat-style web page.
- **Choose the face detector:** `opencv` (fastest), `mtcnn` or `retinaface`.
- **Choose the prompt style:**
  - `focus_emotion`: 2–3 sentences focused on the emotions present
  - `bullet_then_summary`: two factual bullet points, then a one-sentence summary
  - `concise`: one short paragraph
- **Grounded descriptions:** the prompts tell the LLM to use only the detections, not to
  invent background or objects, and to repeat emotion labels exactly as detected.
- **Two LLM providers:**
  - **Groq** (default): Llama 3.1 8B Instant through Groq's OpenAI-compatible API
  - **Ollama**: a local model (default `gemma3`) for running everything offline
- **Chat history** per browser session, with detected emotions and confidence shown as
  tags and the processing time for each image.
- **Housekeeping:** a loading screen while the image is processed, and automatic removal
  of uploads older than one hour.

## Tech stack

Python 3.10 · Flask · DeepFace · TensorFlow · OpenCV · MTCNN · RetinaFace · Groq API ·
Ollama · HTML/CSS (Jinja templates) · Gunicorn

## Run it locally

1. Create and activate a virtual environment with **Python 3.10**:

   ```bash
   python -m venv venv
   venv\Scripts\activate        # Windows
   source venv/bin/activate     # macOS / Linux
   ```

2. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   For a local LLM, also install the Ollama client (`pip install ollama`) and
   [Ollama](https://ollama.com) itself, then pull a model, e.g. `ollama pull gemma3`.

3. Create a `.env` file in the project folder:

   ```env
   SECRET_KEY=any-long-random-string

   # Option A: Groq (hosted)
   LLM_PROVIDER=groq
   GROQ_API_KEY=your-groq-key
   GROQ_MODEL=llama-3.1-8b-instant

   # Option B: Ollama (local)
   # LLM_PROVIDER=ollama
   # OLLAMA_HOST=http://127.0.0.1:11434
   # OLLAMA_MODEL=gemma3
   ```

4. Start the app:

   ```bash
   python application.py
   ```

   and open <http://127.0.0.1:5000>. The first analysis is slower, because DeepFace
   downloads its emotion model weights on first use.

For a server deployment, the app can be started with Gunicorn
(`gunicorn application:app`); set the same variables as environment variables and the
`PORT` variable if your host requires it.

## Project structure

```
application.py        Flask app: upload, emotion detection, LLM call, chat history
templates/
  webpage.html        chat interface
  loading.html        loading screen shown while an image is processed
css/
  webpage.css         interface styles
  loading.css         loading screen styles
requirements.txt      Python dependencies
PythonProject3/       earlier development version and exploration notebook (part1.ipynb)
```

## Limitations

- Emotion recognition from a single photo is an estimate; lighting, angle and occlusion
  affect the result, and facial expressions do not always reflect how someone feels.
- The description only covers faces and emotions. Other objects and the scene are not
  analysed, by design, so the LLM has nothing to invent from.
- Uploaded images are stored temporarily on the server (up to one hour). Use images you
  have the right to process.

## Acknowledgements

Face detection and emotion recognition use the open-source
[DeepFace](https://github.com/serengil/deepface) library by Sefik Ilkin Serengil (MIT licence).
