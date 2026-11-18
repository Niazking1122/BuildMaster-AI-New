
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

app = FastAPI(title="BuildMaster AI", version="4.0")

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
        "You are BuildMaster AI, a senior full-stack software architect "
        "and developer. Generate a complete, runnable multi-file project, "
        "not just a demo snippet. Choose a suitable technology stack based "
        "on the user's request and keep setup simple.\n"
        "Return ONLY valid JSON in this exact format: "
        '{"project_name":"my-app","files":['
        '{"path":"src/main.py","content":"complete source code"}]}\n'
        "Project requirements:\n"
        "1. Create a sensible folder structure and separate files by purpose. "
        "Include frontend, backend, and database/schema files when the app needs them. "
        "2. Implement the requested features end-to-end. If the user asks for "
        "accounts, login, roles, or saved records, build real backend routes and "
        "persistent database integration; do not fake these with browser-only state. "
        "3. Use secure password hashing and server-side authorization for login apps. "
        "Never store plaintext passwords. Include input validation and safe defaults. "
        "4. Include a .env.example with placeholder variable names only, never real secrets. "
        "5. Include dependency files such as requirements.txt or package.json, "
        "database setup/schema or migrations where appropriate, and a detailed README "
        "with installation, environment setup, database setup, run, and test instructions. "
        "6. Add useful error handling and a basic test or health-check when appropriate. "
        "7. If a feature requires an external service or credential, explain setup in README "
        "and use environment variables. Do not claim that an external service is already connected. "
        "8. Do not include passwords, API keys, tokens, or other secrets. "
        "Use relative file paths only; never use absolute paths or ../. "
        "Maximum 20 files, each file must be under 300 KB. "
        "Make the result coherent: imports, routes, filenames, and README commands must match. "
        "Do not output markdown fences or explanations outside the JSON.\n\n"
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
