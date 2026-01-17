# FeelAI — Flask Emotion Chat (DeepFace + LLM)

FeelAI is a simple **chat-like Flask web app** where the user uploads an image and the system:
- detects face emotion(s) using **DeepFace**
- sends the detected results to an **LLM**
- generates a short natural-language description
- shows both detections and LLM output inside a chat UI (HTML/CSS only)

## Features
- Chat-style interface (session history)
- Image upload + preview in chat
- Emotion detection (DeepFace)
- LLM text generation (default: **Groq API** for deployment)
- Loading screen for long processing times (no JS) using `/loading → /process`

## Tech Stack
- Python, Flask
- OpenCV + DeepFace (emotion analysis)
- LLM Provider:
  - **Groq API** (recommended for public deployment)
  - optional local mode with **Ollama** (for development)

---

## Project Structure

