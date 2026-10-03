import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("FASTVIDEO_ATTENTION_BACKEND", "VIDEO_SPARSE_ATTN_H3")

import asyncio
import threading
import time
import traceback
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from fastvideo import VideoGenerator
from fastvideo.api import (
    ComponentConfig,
    EngineConfig,
    GenerationRequest,
    GeneratorConfig,
    InputConfig,
    OffloadConfig,
    OutputConfig,
    ParallelismConfig,
    PipelineSelection,
    SamplingConfig,
)
from fastvideo.pipelines.basic.minimax_h3.reference import MiniMaxH3Reference

MODEL_PATH = "/workspace/models/fasth3"
REF2VA_MODEL_PATH = "/workspace/models/minimax_h3"

WORKER_MODE = os.environ.get("GPU_WORKER_MODE", "t2va")
WORKER_PORT = int(os.environ.get("GPU_WORKER_PORT", "8091" if WORKER_MODE == "t2va" else "8092"))

assert WORKER_MODE in ("t2va", "ref2va"), f"bad GPU_WORKER_MODE: {WORKER_MODE}"

app = FastAPI(title=f"Tovi GPU Worker [{WORKER_MODE}]")

state = {"ready": False, "generator": None, "mode": WORKER_MODE}
gen_lock = threading.Lock()


class GenReq(BaseModel):
    prompt: str
    output_path: str
    num_frames: int = 125
    steps: int = 9
    height: int = 832
    width: int = 480
    seed: int = 0
    reference_image: str = ""


@app.get("/health")
async def health():
    return {"ready": state["ready"], "mode": state["mode"]}


def _load_t2va():
    return VideoGenerator.from_config(
        GeneratorConfig(
            model_path=MODEL_PATH,
            pipeline=PipelineSelection(experimental={}),
            engine=EngineConfig(
                num_gpus=1,
                execution_backend="mp",
                use_fsdp_inference=False,
                parallelism=ParallelismConfig(tp_size=1, sp_size=1),
            ),
        )
    )


def _load_ref2va():
    return VideoGenerator.from_config(
        GeneratorConfig(
            model_path=REF2VA_MODEL_PATH,
            engine=EngineConfig(
                num_gpus=1,
                use_fsdp_inference=False,
                parallelism=ParallelismConfig(tp_size=1, sp_size=1),
                offload=OffloadConfig(dit=False, dit_layerwise=False, text_encoder=True, vae=True, pin_cpu_memory=False),
            ),
            pipeline=PipelineSelection(
                workload_type="i2v",
                components=ComponentConfig(override_pipeline_cls_name="MiniMaxH3Ref2VAModularPipeline"),
            ),
        )
    )


def _run(req: GenReq):
    with gen_lock:
        if WORKER_MODE == "ref2va":
            references = [MiniMaxH3Reference(source=req.reference_image, media_type="image")]
            request = GenerationRequest(
                prompt=req.prompt,
                inputs=InputConfig(references=references),
                sampling=SamplingConfig(
                    height=req.height,
                    width=req.width,
                    num_frames=req.num_frames,
                    fps=24,
                    num_inference_steps=req.steps,
                    guidance_scale=1.0,
                    batch_cfg=False,
                    seed=req.seed,
                ),
                output=OutputConfig(
                    output_path=req.output_path,
                    save_video=True,
                    return_frames=False,
                ),
            )
        else:
            request = GenerationRequest(
                prompt=req.prompt,
                negative_prompt="",
                sampling=SamplingConfig(
                    height=req.height,
                    width=req.width,
                    num_frames=req.num_frames,
                    fps=24,
                    num_inference_steps=req.steps,
                    guidance_scale=1.0,
                    batch_cfg=False,
                    seed=req.seed,
                ),
                output=OutputConfig(
                    output_path=req.output_path,
                    save_video=True,
                    return_frames=False,
                ),
            )
        t0 = time.time()
        result = state["generator"].generate(request)
        return time.time() - t0, result


@app.post("/generate")
async def generate(req: GenReq):
    if not state["ready"]:
        return {"ok": False, "error": "model still loading"}

    loop = asyncio.get_event_loop()
    try:
        elapsed, result = await loop.run_in_executor(None, _run, req)
        ok = Path(req.output_path).exists()
        print(f"[gpu_worker:{WORKER_MODE}] generation done in {elapsed:.1f}s ok={ok} -> {req.output_path}", flush=True)
        return {"ok": ok, "video_path": req.output_path, "elapsed": elapsed}
    except Exception as e:
        err = f"{e}\n{traceback.format_exc()}"
        print(f"[gpu_worker:{WORKER_MODE}] generation FAILED: {err}", flush=True)
        return {"ok": False, "error": err}


def main():
    print(f"[gpu_worker:{WORKER_MODE}] loading VideoGenerator on main thread (one-time, takes several minutes)...", flush=True)
    t0 = time.time()
    state["generator"] = _load_t2va() if WORKER_MODE == "t2va" else _load_ref2va()
    state["ready"] = True
    print(f"[gpu_worker:{WORKER_MODE}] READY. model loaded in {time.time()-t0:.1f}s. starting HTTP server on 127.0.0.1:{WORKER_PORT}", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=WORKER_PORT)


if __name__ == "__main__":
    main()
