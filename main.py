import asyncio
import json
import os
import subprocess
import time
import uuid
from pathlib import Path

import httpx
from fastapi import FastAPI
from fastapi import UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"
OUTPUT_DIR = Path("/workspace/miniapp_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = "/workspace/models/fasth3"
FASTVIDEO_DIR = "/root/FastVideo"
PYTHON_BIN = "/workspace/venv_max/bin/python"
GPU_WORKER_T2VA_URL = "http://127.0.0.1:8091"
GPU_WORKER_REF2VA_URL = "http://127.0.0.1:8092"
UPLOAD_DIR = Path("/workspace/miniapp/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Tovi AI Mini App")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- in-memory job store + single-worker queue (one GPU) ----
JOBS: dict[str, dict] = {}
job_queue: "asyncio.Queue[str]" = asyncio.Queue()

ASPECT_MAP = {
    "9:16": (480, 832),  # width, height
    "16:9": (832, 480),
    "1:1": (640, 640),
}
DURATION_TO_FRAMES = {
    "5": 125,
    "10": 250,
}

STYLE_PREFIX = {
    "cinematic": "cinematic film still, dramatic lighting, ",
    "anime": "anime style, vibrant colors, studio ghibli inspired, ",
    "realistic": "photorealistic, natural lighting, highly detailed, ",
    "3d": "3d render, pixar style, octane render, ",
}


class GenerateRequest(BaseModel):
    prompt: str
    style: str = "cinematic"
    aspect: str = "9:16"
    duration: str = "5"
    init_data: str = ""
    photo_id: str = ""


@app.post("/api/generate")
async def generate(req: GenerateRequest):
    job_id = uuid.uuid4().hex[:12]
    width, height = ASPECT_MAP.get(req.aspect, ASPECT_MAP["9:16"])
    num_frames = DURATION_TO_FRAMES.get(req.duration, 125)
    full_prompt = STYLE_PREFIX.get(req.style, "") + req.prompt
    photo_path = ""
    if req.photo_id:
        matches = list(UPLOAD_DIR.glob(f"{req.photo_id}.*"))
        if matches:
            photo_path = str(matches[0])

    JOBS[job_id] = {
        "status": "queued",
        "prompt": req.prompt,
        "created": time.time(),
        "width": width,
        "height": height,
        "num_frames": num_frames,
        "full_prompt": full_prompt,
        "photo_path": photo_path,
    }
    await job_queue.put(job_id)
    return {"job_id": job_id, "status": "queued"}


@app.post("/api/upload_photo")
async def upload_photo(photo: UploadFile = File(...)):
    photo_id = uuid.uuid4().hex
    ext = os.path.splitext(photo.filename or "")[1] or ".jpg"
    dest = UPLOAD_DIR / f"{photo_id}{ext}"
    with open(dest, "wb") as f:
        f.write(await photo.read())
    return {"photo_id": photo_id}

@app.get("/api/status/{job_id}")
async def status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        return {"status": "error", "message": "not found"}
    resp = {"status": job["status"], "prompt": job["prompt"]}
    if job["status"] == "done":
        resp["result_url"] = f"/media/{job_id}.mp4"
    if job["status"] == "error":
        resp["message"] = job.get("error", "unknown error")
    return resp


@app.get("/media/{filename}")
async def media(filename: str):
    path = OUTPUT_DIR / filename
    return FileResponse(path, media_type="video/mp4")


async def worker_loop():
    async with httpx.AsyncClient(timeout=1800.0) as client:
        while True:
            job_id = await job_queue.get()
            job = JOBS[job_id]
            job["status"] = "running"
            try:
                final_path = OUTPUT_DIR / f"{job_id}.mp4"
                target_url = GPU_WORKER_REF2VA_URL if job["photo_path"] else GPU_WORKER_T2VA_URL
                resp = await client.post(
                    f"{target_url}/generate",
                    json={
                        "prompt": job["full_prompt"],
                        "output_path": str(final_path),
                        "num_frames": job["num_frames"],
                        "steps": 9,
                        "height": job["height"],
                        "width": job["width"],
                        "seed": 0,
                        "reference_image": job.get("photo_path", ""),
                    },
                )
                data = resp.json()
                if data.get("ok") and final_path.exists():
                    job["status"] = "done"
                else:
                    job["status"] = "error"
                    job["error"] = str(data.get("error", "unknown error"))[-4000:]
                    print(f"[job {job_id}] FAILED:\n{job['error']}")
            except Exception as e:
                job["status"] = "error"
                job["error"] = str(e)
                print(f"[job {job_id}] EXCEPTION: {e}")
            finally:
                job_queue.task_done()


@app.on_event("startup")
async def on_startup():
    asyncio.create_task(worker_loop())


app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static") 
