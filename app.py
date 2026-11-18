
import os
import json
import urllib.request
import urllib.error
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI(title="BuildMaster AI", version="2.0.0")

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
        return {"reply": "Gemini API key is missing. Please check Vercel Environment Variables."}

    memory = "\n".join(preferences)

    system_instruction = (
        "You are BuildMaster AI, a helpful personal software architect "
        "and app builder. Help users plan, design, build, debug and improve "
        "websites and applications. Give complete practical code when asked. "
        "Explain steps simply. Remember the user's preferences supplied "
        "in the conversation context."
    )

    prompt = (
        system_instruction
        + "\n\nSaved user preferences:\n"
        + (memory if memory else "No saved preferences yet.")
        + "\n\nUser request:\n"
        + message
    )

    url = "https://generativelanguage.googleapis.com/v1beta/interactions"

    payload = {
        "model": "gemini-3.7-flash",
        "input": prompt
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

        reply = result.get("output_text", "")

        if not reply:
            for step in result.get("steps", []):
                for part in step.get("content", {}).get("parts", []):
                    if part.get("text"):
                        reply += part["text"]

        if not reply:
            reply = "Gemini responded, but no text was returned."

        return {"reply": reply}

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
