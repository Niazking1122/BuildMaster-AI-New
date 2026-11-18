
import os
import json
import urllib.request
import urllib.error
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI(title="BuildMaster AI", version="2.1.0")

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
preferences = []

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class ChatRequest(BaseModel):
    message: str


class MemoryRequest(BaseModel):
    preferences: list[str]


@app.get("/")
def home():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"status": "online"}


@app.get("/api/health")
def health():
    return {
        "status": "online",
        "app": "BuildMaster AI",
        "backend": "working",
        "ai_configured": bool(os.getenv("GEMINI_API_KEY"))
    }


@app.get("/api/info")
def info():
    return {
        "name": "BuildMaster AI",
        "purpose": "Personal AI Software Builder"
    }


@app.post("/api/chat")
def chat(data: ChatRequest):
    message = data.message.strip()

    if not message:
        return {"reply": "Please tell me your idea."}

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        return {"reply": "GEMINI_API_KEY is missing in Vercel."}

    memory = "\n".join(preferences)

    prompt = (
        "You are BuildMaster AI, a personal software architect "
        "and app builder. Help users plan, design, code, debug "
        "and improve websites and apps. Give complete working "
        "code when requested. Explain steps simply.\n\n"
        "Saved user preferences:\n"
        + (memory if memory else "No saved preferences.")
        + "\n\nUser request:\n"
        + message
    )

    url = (
        "https://generativelanguage.googleapis.com/v1beta/"
        "models/gemini-3.5-flash-lite:generateContent"
    )

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ]
    }

    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": api_key
        },
        method="POST"
    )

    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            result = json.loads(response.read().decode("utf-8"))

        candidates = result.get("candidates", [])

        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            reply = "\n".join(
                part.get("text", "")
                for part in parts
                if part.get("text")
            )

            if reply:
                return {"reply": reply}

        return {"reply": "Gemini returned no text. Please try again."}

    except urllib.error.HTTPError as error:
        details = error.read().decode("utf-8", errors="replace")
        return {"reply": "Gemini API error: " + details[:1200]}

    except Exception as error:
        return {"reply": "Connection error: " + str(error)}


@app.get("/api/memory")
def get_memory():
    return {"preferences": preferences}


@app.post("/api/memory")
def save_memory(data: MemoryRequest):
    preferences.clear()

    for item in data.preferences:
        item = item.strip()
        if item and item not in preferences:
            preferences.append(item)

    return {"status": "saved", "preferences": preferences}
