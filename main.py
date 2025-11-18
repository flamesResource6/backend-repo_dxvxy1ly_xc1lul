import os
import json
import subprocess
from typing import List, Optional, Dict, Any

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from database import db, create_document, get_documents

app = FastAPI(title="AI Tools Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------
# Models
# -----------------------------
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str = Field(..., description="Ollama model name, e.g., 'llama3.1' ")
    messages: List[ChatMessage]
    options: Optional[Dict[str, Any]] = None

class GenerateRequest(BaseModel):
    model: str
    prompt: str
    options: Optional[Dict[str, Any]] = None

class ScriptRunRequest(BaseModel):
    script_path: str = Field(..., description="Absolute path to your python script or a script in ./scripts")
    args: Optional[List[str]] = Field(default_factory=list)
    input_text: Optional[str] = None

class Research(BaseModel):
    topic: str
    summary: str

class PlanItem(BaseModel):
    day: str
    tasks: List[str]

class WeeklyPlan(BaseModel):
    title: str
    week_start: str
    items: List[PlanItem]


# -----------------------------
# Helpers
# -----------------------------
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

def ollama_available() -> bool:
    try:
        r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


# -----------------------------
# Basic
# -----------------------------
@app.get("/")
def read_root():
    return {"message": "AI Tools Backend running", "ollama": "available" if ollama_available() else "unavailable"}

@app.get("/api/health")
def health():
    return {"status": "ok", "ollama": ollama_available()}

@app.get("/test")
def test_database():
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": []
    }

    try:
        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
            response["database_name"] = getattr(db, 'name', None) or os.getenv("DATABASE_NAME") or "Unknown"
            try:
                collections = db.list_collection_names()
                response["collections"] = collections[:10]
                response["database"] = "✅ Connected & Working"
                response["connection_status"] = "Connected"
            except Exception as e:
                response["database"] = f"⚠️ Connected but Error: {str(e)[:80]}"
        else:
            response["database"] = "⚠️ Available but not initialized"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:80]}"

    return response


# -----------------------------
# Ollama proxy endpoints
# -----------------------------
@app.get("/api/ollama/status")
def ollama_status():
    return {"available": ollama_available(), "host": OLLAMA_HOST}

@app.get("/api/ollama/models")
def list_models():
    if not ollama_available():
        return {"models": []}
    try:
        r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=10)
        r.raise_for_status()
        data = r.json()
        models = [m.get("name") for m in data.get("models", [])]
        return {"models": models}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ollama/generate")
def ollama_generate(req: GenerateRequest):
    if not ollama_available():
        raise HTTPException(status_code=503, detail="Ollama server not reachable")
    payload = {"model": req.model, "prompt": req.prompt, "stream": True}
    if req.options:
        payload["options"] = req.options
    try:
        r = requests.post(f"{OLLAMA_HOST}/api/generate", json=payload, stream=True, timeout=120)
        r.raise_for_status()
        full_text = ""
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            try:
                part = json.loads(line)
                full_text += part.get("response", "")
                if part.get("done"):
                    break
            except json.JSONDecodeError:
                continue
        return {"text": full_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ollama/chat")
def ollama_chat(req: ChatRequest):
    if not ollama_available():
        raise HTTPException(status_code=503, detail="Ollama server not reachable")
    payload = {
        "model": req.model,
        "messages": [m.model_dump() for m in req.messages],
        "stream": True,
    }
    if req.options:
        payload["options"] = req.options
    try:
        r = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, stream=True, timeout=120)
        r.raise_for_status()
        full_text = ""
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            try:
                part = json.loads(line)
                msg = part.get("message", {})
                full_text += msg.get("content", "")
                if part.get("done"):
                    break
            except json.JSONDecodeError:
                continue
        return {"message": {"role": "assistant", "content": full_text}}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------
# Run custom Python script
# -----------------------------
@app.post("/api/run-script")
def run_script(req: ScriptRunRequest):
    # Resolve path: allow absolute or relative to ./scripts
    path = req.script_path
    if not os.path.isabs(path):
        maybe = os.path.join(os.getcwd(), "scripts", path)
        if os.path.exists(maybe):
            path = maybe
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"Script not found: {path}")

    try:
        cmd = ["python", path] + (req.args or [])
        proc = subprocess.run(
            cmd,
            input=(req.input_text or "").encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=300
        )
        return {
            "code": proc.returncode,
            "stdout": proc.stdout.decode("utf-8", errors="ignore"),
            "stderr": proc.stderr.decode("utf-8", errors="ignore"),
        }
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Script execution timed out")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------
# Simple persistence for planner and research
# -----------------------------
@app.post("/api/research")
def save_research(item: Research):
    if db is None:
        raise HTTPException(status_code=500, detail="Database not configured")
    _id = create_document("research", item.model_dump())
    return {"id": _id}

@app.get("/api/research")
def list_research(limit: int = 50):
    if db is None:
        raise HTTPException(status_code=500, detail="Database not configured")
    docs = get_documents("research", limit=limit)
    for d in docs:
        d["_id"] = str(d["_id"])  # stringify for JSON
    return {"items": docs}

@app.post("/api/plans")
def save_plan(plan: WeeklyPlan):
    if db is None:
        raise HTTPException(status_code=500, detail="Database not configured")
    payload = plan.model_dump()
    _id = create_document("weeklyplan", payload)
    return {"id": _id}

@app.get("/api/plans")
def list_plans(limit: int = 50):
    if db is None:
        raise HTTPException(status_code=500, detail="Database not configured")
    docs = get_documents("weeklyplan", limit=limit)
    for d in docs:
        d["_id"] = str(d["_id"])  # stringify
    return {"items": docs}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
