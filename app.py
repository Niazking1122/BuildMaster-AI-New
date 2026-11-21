import os
import io
import re
import json
import time
import random
import zipfile
import urllib.request
import urllib.error
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI(title="BuildMaster AI", version="5.1")

BASE = Path(__file__).resolve().parent
STATIC = BASE / "static"
preferences = []

GEMINI_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash"
]
RETRY_STATUSES = {429, 500, 502, 503, 504}


if STATIC.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


class ChatRequest(BaseModel):
    message: str


class MemoryRequest(BaseModel):
    preferences: list[str]



def supabase_config():
    url = os.getenv("SUPABASE_URL", "").strip().rstrip("/")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
    if not url or not key:
        return None, None
    return url, key



def supabase_request(path, method="GET", payload=None):
    base_url, key = supabase_config()
    if not base_url or not key:
        raise RuntimeError("Supabase environment variables are not configured.")
    url = base_url + "/rest/v1/" + path.lstrip("/")
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "apikey": key,
            "Authorization": "Bearer " + key,
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Prefer": "return=minimal"
        },
        method=method
    )
    with urllib.request.urlopen(req, timeout=20) as response:
        response_body = response.read().decode("utf-8")
        if response_body:
            return json.loads(response_body)
        return None



def load_preferences():
    base_url, key = supabase_config()
    if base_url and key:
        try:
            rows = supabase_request("bm_shared_memories?select=content&order=id.asc")
            items = [row.get("content", "").strip() for row in rows if row.get("content", "").strip()]
            preferences.clear()
            preferences.extend(items)
            return items
        except Exception:
            pass
    return list(preferences)



def persist_preferences(items):
    base_url, key = supabase_config()
    if not base_url or not key:
        return False
    # This table stores shared preferences for this prototype (not private user data).
    supabase_request("bm_shared_memories?id=gt.0", method="DELETE")
    if items:
        supabase_request("bm_shared_memories", method="POST", payload=[{"content": item} for item in items])
    return True



def ask_gemini(prompt, json_mode=False):
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("Gemini API key is missing in Vercel.")
    config = {"maxOutputTokens": 16000}
    if json_mode:
        config["responseMimeType"] = "application/json"
    payload = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": config}
    last_error = "No Gemini model responded."
    for model in GEMINI_MODELS:
        url = "https://generativelanguage.googleapis.com/v1beta/" + f"models/{model}:generateContent"
        for attempt in range(2):
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json", "x-goog-api-key": key},
                method="POST"
            )
            try:
                with urllib.request.urlopen(req, timeout=120) as response:
                    result = json.loads(response.read().decode("utf-8"))
                candidates = result.get("candidates", [])
                if not candidates:
                    last_error = f"{model}: Gemini returned no response."
                    break
                parts = candidates[0].get("content", {}).get("parts", [])
                reply = "\n".join(part.get("text", "") for part in parts if part.get("text"))
                if reply.strip():
                    return reply
                last_error = f"{model}: Empty Gemini response."
                break
            except urllib.error.HTTPError as error:
                body = error.read().decode(errors="replace")
                last_error = f"{model}: HTTP {error.code}: {body[:700]}"
                if error.code in RETRY_STATUSES:
                    if attempt == 0:
                        time.sleep(1.5 + random.random())
                        continue
                    break
                if error.code == 404:
                    break
                raise RuntimeError(last_error)
            except (urllib.error.URLError, TimeoutError, OSError) as error:
                last_error = f"{model}: Connection error: {error}"
                if attempt == 0:
                    time.sleep(1.5 + random.random())
                    continue
                break
            except Exception as error:
                last_error = f"{model}: {str(error)}"
                break
    raise RuntimeError("All Gemini models failed. Please try again later. " + last_error)


@app.get("/")
def home():
    file = STATIC / "index.html"
    if file.exists():
        return FileResponse(file)
    return {"status": "online"}


@app.get("/api/health")
def health():
    supabase_url, supabase_key = supabase_config()
    return {
        "status": "online",
        "app": "BuildMaster AI",
        "version": "5.1",
        "ai_configured": bool(os.getenv("GEMINI_API_KEY")),
        "memory_configured": bool(supabase_url and supabase_key),
        "models": GEMINI_MODELS
    }


@app.post("/api/chat")
def chat(data: ChatRequest):
    message = data.message.strip()
    if not message:
        return {"reply": "Please describe your idea."}
    saved_preferences = load_preferences()
    prompt = (
        "You are BuildMaster AI, a personal software architect. "
        "Help users plan, design, code and debug complete applications. "
        "Give practical code and simple explanations. Be honest about prototypes and limitations.\n\n"
        "Saved preferences:\n" + "\n".join(saved_preferences) + "\n\nUser request:\n" + message
    )
    try:
        return {"reply": ask_gemini(prompt)}
    except Exception as error:
        return {"reply": str(error)}


@app.post("/api/build")
def build_project(data: ChatRequest):
    message = data.message.strip()
    if not message:
        return JSONResponse({"error": "Describe your application first."}, status_code=400)
    saved_preferences = load_preferences()
    prompt = (
        "You are BuildMaster AI, a senior full-stack software architect and developer. "
        "Generate a complete, runnable multi-file project, not just a demo snippet. "
        "Choose a suitable technology stack based on the user's request and keep setup simple.\n"
        "Return ONLY valid JSON in this exact format: "
        '{"project_name":"my-app","files":[{"path":"src/main.py","content":"complete source code"}]}\n'
        "Project requirements:\n"
        "1. Create a sensible folder structure and separate files by purpose. Include frontend, backend, and database/schema files when needed.\n"
        "2. Implement requested features end-to-end. If the user asks for accounts, login, roles, or saved records, build real backend routes and persistent database integration; do not fake these with browser-only state.\n"
        "3. Use secure password hashing and server-side authorization. Never store plaintext passwords. Include input validation and safe defaults.\n"
        "4. Include a .env.example with placeholder variable names only, never real secrets.\n"
        "5. Include requirements.txt or package.json, database setup/schema or migrations where appropriate, and a detailed README with installation, environment setup, database setup, run, and test instructions.\n"
        "6. Add useful error handling and a basic test or health-check when appropriate.\n"
        "7. Explain external service setup in README and use environment variables. Do not claim an external service is already connected.\n"
        "8. Do not include passwords, API keys, tokens, or other secrets. Use relative file paths only; never use absolute paths or ../.\n"
        "Maximum 20 files, each file under 300 KB. Make imports, routes, filenames, and README commands match.\n"
        "Do not output markdown fences or explanations outside the JSON.\n\n"
        "User preferences:\n" + "\n".join(saved_preferences) + "\n\nApplication request:\n" + message
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
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            for item in files[:20]:
                if not isinstance(item, dict):
                    continue
                path = str(item.get("path", "")).replace("\\", "/")
                content = item.get("content", "")
                if (not path or path.startswith("/") or ".." in Path(path).parts
                    or not isinstance(content, str) or len(content.encode("utf-8")) > 300000):
                    continue
                archive.writestr(path, content)
                count += 1
        if count == 0:
            raise ValueError("No valid project files were generated.")
        buffer.seek(0)
        name = re.sub(r"[^a-zA-Z0-9_-]+", "-", str(project.get("project_name", "my-project"))).strip("-") or "my-project"
        return StreamingResponse(buffer, media_type="application/zip", headers={"Content-Disposition": f'attachment; filename="{name}.zip"'})
    except Exception as error:
        return JSONResponse({"error": str(error)}, status_code=502)


@app.get("/api/memory")
def get_memory():
    return {"preferences": load_preferences()}


@app.post("/api/memory")
def save_memory(data: MemoryRequest):
    cleaned = []
    for item in data.preferences:
        if not isinstance(item, str):
            continue
        item = item.strip()
        if item and item not in cleaned:
            cleaned.append(item)
    try:
        persistent = persist_preferences(cleaned)
        preferences.clear()
        preferences.extend(cleaned)
        return {"status": "saved", "persistent": persistent, "preferences": preferences}
    except Exception as error:
        return JSONResponse({"error": "Could not save shared memory to Supabase: " + str(error)}, status_code=502)
@app.get("/api/memory/test")
def test_memory():
    try:
        rows = supabase_request("bm_shared_memories?select=id,content")
        return {"status": "success", "data": rows}
    except Exception as e:
        return {"status": "error", "message": str(e)}
