
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path

app = FastAPI(title="BuildMaster AI", version="1.1.0")

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
        "backend": "working"
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

    reply = (
        "Welcome to BuildMaster AI!\n\n"
        "I received your idea:\n"
        + message
        + "\n\nLet's build it step by step:\n"
        "1. Understand your idea.\n"
        "2. Plan the design.\n"
        "3. Prepare the code.\n"
        "4. Test the project.\n\n"
        "This is a starter response. Real AI connection is coming next."
    )

    if preferences:
        reply += "\n\nYour saved preferences: " + "; ".join(preferences)

    return {"reply": reply}


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
