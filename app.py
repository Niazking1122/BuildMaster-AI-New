
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

app = FastAPI(title="BuildMaster AI", version="5.2")

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


# -------------------------
# SUPABASE CONFIG
# -------------------------

def supabase_config():
    url = os.getenv("SUPABASE_URL", "").strip().rstrip("/")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()

    if not url or not key:
        return None, None

    return url, key


def supabase_request(path, method="GET", payload=None):
    base_url, key = supabase_config()

    if not base_url or not key:
        raise RuntimeError(
            "Supabase URL or secret key is missing."
        )

    url = base_url + "/rest/v1/" + path.lstrip("/")

    body = None
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "apikey": key,
            "Authorization": "Bearer " + key,
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Prefer": "return=representation"
        },
        method=method
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            response_body = response.read().decode("utf-8")
            if response_body:
                return json.loads(response_body)
            return None

    except urllib.error.HTTPError as error:
        error_body = error.read().decode(
            "utf-8", errors="replace"
        )
        raise RuntimeError(
            f"Supabase HTTP {error.code}: {error_body[:1000]}"
        )

    except urllib.error.URLError as error:
        raise RuntimeError(
            f"Supabase connection failed: {error}"
        )


# -------------------------
# MEMORY: LOAD
# -------------------------

def load_preferences():
    base_url, key = supabase_config()

    if base_url and key:
        try:
            rows = supabase_request(
                "bm_shared_memories?select=id,content&order=id.asc"
            )

            items = [
                row.get("content", "").strip()
                for row in rows
                if row.get("content", "").strip()
            ]

            preferences.clear()
            preferences.extend(items)

            return items

        except Exception as error:
            print("MEMORY LOAD ERROR:", str(error))

    return list(preferences)


# -------------------------
# MEMORY: SAVE
# -------------------------

def persist_preferences(items):
    base_url, key = supabase_config()

    if not base_url or not key:
        raise RuntimeError(
            "Supabase environment variables are missing."
        )

    # Clear old shared preferences
    supabase_request(
        "bm_shared_memories?id=gt.0",
        method="DELETE"
    )

    # Insert new preferences
    if items:
        supabase_request(
            "bm_shared_memories",
            method="POST",
            payload=[
                {"content": item}
                for item in items
            ]
        )

    return True


# -------------------------
# GEMINI AI
# -------------------------

def ask_gemini(prompt, json_mode=False):
    key = os.getenv("GEMINI_API_KEY")

    if not key:
        raise RuntimeError(
            "Gemini API key is missing in Vercel."
        )

    config = {"maxOutputTokens": 16000}

    if json_mode:
        config["responseMimeType"] = "application/json"

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ],
        "generationConfig": config
    }

    last_error = "No Gemini model responded."

    for model in GEMINI_MODELS:

        url = (
            "https://generativelanguage.googleapis.com/"
            "v1beta/models/"
            + model
            + ":generateContent"
        )

        for attempt in range(2):

            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "x-goog-api-key": key
                },
                method="POST"
            )

            try:
                with urllib.request.urlopen(
                    req, timeout=120
                ) as response:
                    result = json.loads(
                        response.read().decode("utf-8")
                    )

                candidates = result.get("candidates", [])

                if not candidates:
                    last_error = (
                        f"{model}: Gemini returned no response."
                    )
                    break

                parts = candidates[0].get(
                    "content", {}
                ).get("parts", [])

                reply = "\n".join(
                    part.get("text", "")
                    for part in parts
                    if part.get("text")
                )

                if reply.strip():
                    return reply

                last_error = f"{model}: Empty response."
                break

            except urllib.error.HTTPError as error:
                body = error.read().decode(
                    errors="replace"
                )

                last_error = (
                    f"{model}: HTTP {error.code}: "
                    + body[:700]
                )

                if error.code in RETRY_STATUSES:
                    if attempt == 0:
                        time.sleep(
                            1.5 + random.random()
                        )
                        continue
                    break

                if error.code == 404:
                    break

                raise RuntimeError(last_error)

            except (
                urllib.error.URLError,
                TimeoutError,
                OSError
            ) as error:

                last_error = (
                    f"{model}: Connection error: {error}"
                )

                if attempt == 0:
                    time.sleep(
                        1.5 + random.random()
                    )
                    continue

                break

            except Exception as error:
                last_error = f"{model}: {str(error)}"
                break

    raise RuntimeError(
        "All Gemini models failed. "
        "Please try again later. " + last_error
    )


# -------------------------
# HOME
# -------------------------

@app.get("/")
def home():
    file = STATIC / "index.html"

    if file.exists():
        return FileResponse(file)

    return {"status": "online"}


# -------------------------
# HEALTH CHECK
# -------------------------

@app.get("/api/health")
def health():
    supabase_url, supabase_key = supabase_config()

    return {
        "status": "online",
        "app": "BuildMaster AI",
        "version": "5.2",
        "ai_configured": bool(
            os.getenv("GEMINI_API_KEY")
        ),
        "memory_configured": bool(
            supabase_url and supabase_key
        ),
        "models": GEMINI_MODELS
    }


# -------------------------
# CHAT
# -------------------------

@app.post("/api/chat")
def chat(data: ChatRequest):
    message = data.message.strip()

    if not message:
        return {"reply": "Please describe your idea."}

    saved_preferences = load_preferences()

    prompt = (
        "You are BuildMaster AI, a personal software architect. "
        "Help users plan, design, code and debug complete applications. "
        "Give practical code and simple explanations. "
        "Be honest about prototypes and limitations.\n\n"
        "Saved preferences:\n"
        + "\n".join(saved_preferences)
        + "\n\nUser request:\n"
        + message
    )

    try:
        return {"reply": ask_gemini(prompt)}

    except Exception as error:
        return {"reply": str(error)}


# -------------------------
# PROJECT BUILDER
# -------------------------

@app.post("/api/build")
def build_project(data: ChatRequest):
    message = data.message.strip()

    if not message:
        return JSONResponse(
            {"error": "Describe your application first."},
            status_code=400
        )

    saved_preferences = load_preferences()

    prompt = (
        "You are BuildMaster AI, a senior full-stack software architect "
        "and developer. Generate a complete, runnable multi-file project, "
        "not just a demo snippet. Choose a suitable technology stack.\n"
        "Return ONLY valid JSON in this exact format: "
        '{"project_name":"my-app","files":[{"path":"src/main.py","content":"complete source code"}]}\n'
        "Requirements:\n"
        "1. Use a sensible folder structure.\n"
        "2. Implement requested features end-to-end.\n"
        "3. Use secure password hashing and server-side authorization.\n"
        "4. Never store plaintext passwords.\n"
        "5. Include .env.example with placeholders only.\n"
        "6. Include requirements.txt or package.json when needed.\n"
        "7. Include database setup/schema where appropriate.\n"
        "8. Include a detailed README with setup and run instructions.\n"
        "9. Add error handling and health checks when appropriate.\n"
        "10. Do not claim external services are connected.\n"
        "11. Never include secrets or API keys.\n"
        "12. Use relative paths only; never use ../ or absolute paths.\n"
        "Maximum 20 files, each under 300 KB.\n"
        "Return JSON only, without markdown fences.\n\n"
        "User preferences:\n"
        + "\n".join(saved_preferences)
        + "\n\nApplication request:\n"
        + message
    )

    try:
        raw = ask_gemini(
            prompt,
            json_mode=True
        ).strip()

        if raw.startswith("```"):
            raw = re.sub(
                r"^```(?:json)?\s*",
                "",
                raw
            )
            raw = re.sub(
                r"\s*```$",
                "",
                raw
            )

        project = json.loads(raw)

        files = project.get("files", [])

        if not isinstance(files, list) or not files:
            raise ValueError(
                "No project files generated."
            )

        buffer = io.BytesIO()
        count = 0

        with zipfile.ZipFile(
            buffer, "w", zipfile.ZIP_DEFLATED
        ) as archive:

            for item in files[:20]:

                if not isinstance(item, dict):
                    continue

                path = str(
                    item.get("path", "")
                ).replace("\\", "/")

                content = item.get("content", "")

                if (
                    not path
                    or path.startswith("/")
                    or ".." in Path(path).parts
                    or not isinstance(content, str)
                    or len(content.encode("utf-8")) > 300000
                ):
                    continue

                archive.writestr(path, content)
                count += 1

        if count == 0:
            raise ValueError(
                "No valid project files were generated."
            )

        buffer.seek(0)

        name = re.sub(
            r"[^a-zA-Z0-9_-]+",
            "-",
            str(
                project.get(
                    "project_name",
                    "my-project"
                )
            )
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


# -------------------------
# MEMORY: GET
# -------------------------

@app.get("/api/memory")
def get_memory():
    return {
        "preferences": load_preferences()
    }


# -------------------------
# MEMORY: POST / SAVE
# -------------------------

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
        persist_preferences(cleaned)

        preferences.clear()
        preferences.extend(cleaned)

        return {
            "status": "saved",
            "persistent": True,
            "preferences": preferences
        }

    except Exception as error:

        print("MEMORY SAVE ERROR:", str(error))

        return JSONResponse(
            {
                "status": "error",
                "persistent": False,
                "error": str(error)
            },
            status_code=502
        )


# -------------------------
# MEMORY: DATABASE TEST
# -------------------------

@app.get("/api/memory/test")
def test_memory():

    try:
        rows = supabase_request(
            "bm_shared_memories?select=id,content&order=id.asc"
        )

        return {
            "status": "success",
            "count": len(rows),
            "data": rows
        }

    except Exception as error:

        return JSONResponse(
            {
                "status": "error",
                "message": str(error)
            },
            status_code=502
        )
