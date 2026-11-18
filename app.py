
import os
import io
import re
import json
import zipfile
import urllib.request
import urllib.error
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI(title="BuildMaster AI", version="3.0")

BASE = Path(__file__).resolve().parent
STATIC = BASE / "static"
preferences = []

if STATIC.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


class ChatRequest(BaseModel):
    message: str


class MemoryRequest(BaseModel):
    preferences: list[str]


def ask_gemini(prompt, json_mode=False):
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("Gemini API key is missing.")

    url = (
        "https://generativelanguage.googleapis.com/v1beta/"
        "models/gemini-3.5-flash-lite:generateContent"
    )

    config = {"maxOutputTokens": 16000}
    if json_mode:
        config["responseMimeType"] = "application/json"

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": config
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": key
        },
        method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            result = json.loads(response.read().decode())
    except urllib.error.HTTPError as error:
        raise RuntimeError(
            "Gemini API error: " +
            error.read().decode(errors="replace")[:1000]
        )

    candidates = result.get("candidates", [])
    if not candidates:
        raise RuntimeError("Gemini returned no response.")

    parts = candidates[0].get("content", {}).get("parts", [])
    reply = "\n".join(
        p.get("text", "") for p in parts if p.get("text")
    )

    if not reply:
        raise RuntimeError("Empty Gemini response.")

    return reply


@app.get("/")
def home():
    file = STATIC / "index.html"
    if file.exists():
        return FileResponse(file)
    return {"status": "online"}


@app.get("/api/health")
def health():
    return {
        "status": "online",
        "app": "BuildMaster AI",
        "ai_configured": bool(os.getenv("GEMINI_API_KEY"))
    }


@app.post("/api/chat")
def chat(data: ChatRequest):
    message = data.message.strip()
    if not message:
        return {"reply": "Please describe your idea."}

    prompt = (
        "You are BuildMaster AI, a personal software architect. "
        "Help users plan, design, code and debug complete applications. "
        "Give practical code and simple explanations. "
        "Be honest about prototypes and limitations.\n\n"
        "Saved preferences:\n" + "\n".join(preferences) +
        "\n\nUser request:\n" + message
    )

    try:
        return {"reply": ask_gemini(prompt)}
    except Exception as error:
        return {"reply": str(error)}


@app.post("/api/build")
def build_project(data: ChatRequest):
    message = data.message.strip()

    if not message:
        return JSONResponse(
            {"error": "Describe your application first."},
            status_code=400
        )

    prompt = (
        "You are an expert full-stack developer. Build a runnable "
        "starter project for the user's application. Return ONLY valid "
        "JSON in this exact format:\n"
        '{"project_name":"my-app","files":['
        '{"path":"index.html","content":"complete source code"}]}\n'
        "Include all necessary files and a README with instructions. "
        "For backend apps include requirements.txt or package.json. "
        "Do not include passwords, API keys or secrets. "
        "Use relative file paths only. Do not use ../ or absolute paths. "
        "Maximum 20 files. Make the project practical and runnable.\n\n"
        "User preferences:\n" + "\n".join(preferences) +
        "\n\nApplication request:\n" + message
    )

    try:
        raw = ask_gemini(prompt, json_mode=True).strip()

        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)

        project = json.loads(raw)
        files = project.get("files", [])

        if not isinstance(files, list) or not files:
            raise ValueError("No project files generated.")

        buffer = io.BytesIO()
        count = 0

        with zipfile.ZipFile(
            buffer, "w", zipfile.ZIP_DEFLATED
        ) as archive:

            for item in files[:20]:
                path = str(item.get("path", "")).replace("\\", "/")
                content = item.get("content", "")

                if (
                    not path or path.startswith("/")
                    or ".." in Path(path).parts
                    or not isinstance(content, str)
                    or len(content.encode("utf-8")) > 300000
                ):
                    continue

                archive.writestr(path, content)
                count += 1

        if count == 0:
            raise ValueError("No valid files were generated.")

        buffer.seek(0)

        name = re.sub(
            r"[^a-zA-Z0-9_-]+",
            "-",
            str(project.get("project_name", "my-project"))
        ).strip("-") or "my-project"

        return StreamingResponse(
            buffer,
            media_type="application/zip",
            headers={
                "Content-Disposition":
                f'attachment; filename="{name}.zip"'
            }
        )

    except Exception as error:
        return JSONResponse(
            {"error": str(error)},
            status_code=502
        )


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
