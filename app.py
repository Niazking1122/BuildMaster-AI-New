from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

app = FastAPI(title="BuildMaster AI", version="1.0.0")

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/")
def home():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"name": "BuildMaster AI", "status": "online"}

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
