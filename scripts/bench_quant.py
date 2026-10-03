#!/usr/bin/env python
"""End-to-end bf16 vs quantized comparison through the real gpu_worker.py.

For every QUANT_MODE it starts gpu_worker.py on a spare port, waits for the
model, runs one warm-up plus the prompt x seed grid with fixed seeds, samples
GPU memory with nvidia-smi, then stops the worker. Afterwards each quantized
video is compared with the bf16 one for the same prompt/seed (PSNR/SSIM on
luma, log-spectrogram distance on audio) and a side-by-side mp4 is written.

Stop the production workers first: memory is read GPU-wide.

    python scripts/bench_quant.py --worker t2va --modes bf16,svdq,fp4 \
        --gpu-price-per-hour 1.89 --out /workspace/quant_bench/run1

Needs ffmpeg and numpy. Results: <out>/report.md, <out>/report.json, <out>/<mode>/*.mp4,
<out>/side_by_side/*.mp4, <out>/<mode>/worker.log.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
DEFAULT_PROMPTS = [
    "cinematic film still, dramatic lighting, a woman in a red coat walks through a rainy neon street at night, "
    "she turns to the camera and says hello",
    "photorealistic, natural lighting, highly detailed, a golden retriever runs along the beach at sunset, "
    "waves crashing, the dog barks happily",
    "anime style, vibrant colors, studio ghibli inspired, a boy rides a bicycle down a hill through a field of "
    "sunflowers, wind chimes ringing",
]
ASPECTS = {"9:16": (480, 832), "16:9": (832, 480), "1:1": (640, 640)}  # width, height (as in main.py)


# --------------------------------------------------------------------------- worker control


def _http_json(url: str, payload: dict | None = None, timeout: float = 10.0) -> dict:
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


class GpuMemSampler(threading.Thread):

    def __init__(self, interval: float = 0.5):
        super().__init__(daemon=True)
        self.interval, self.samples, self._stop_evt = interval, [], threading.Event()

    @staticmethod
    def read_mib() -> float:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", "0"], text=True)
        return float(out.strip().splitlines()[0])

    def run(self):
        while not self._stop_evt.is_set():
            try:
                self.samples.append((time.time(), self.read_mib()))
            except Exception:  # noqa: BLE001 - a missed sample is fine
                pass
            self._stop_evt.wait(self.interval)

    def stop(self):
        self._stop_evt.set()
        self.join(timeout=5)

    def peak_between(self, t0: float, t1: float) -> float | None:
        vals = [m for t, m in self.samples if t0 <= t <= t1]
        return max(vals) if vals else None


class Worker:

    def __init__(self, mode: str, worker: str, port: int, out_dir: Path, extra_env: dict[str, str]):
        self.mode, self.port, self.url = mode, port, f"http://127.0.0.1:{port}"
        env = {**os.environ, **extra_env, "QUANT_MODE": mode, "GPU_WORKER_MODE": worker,
               "GPU_WORKER_PORT": str(port)}
        self.log_path = out_dir / "worker.log"
        self.log = open(self.log_path, "w")
        self.proc = subprocess.Popen([sys.executable, str(REPO / "gpu_worker.py")], cwd=REPO, env=env,
                                     stdout=self.log, stderr=subprocess.STDOUT, start_new_session=True)

    def wait_ready(self, timeout: float) -> float:
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.proc.poll() is not None:
                raise RuntimeError(f"worker [{self.mode}] exited with {self.proc.returncode}; see {self.log_path}")
            try:
                if _http_json(f"{self.url}/health", timeout=5).get("ready"):
                    return time.time() - t0
            except (urllib.error.URLError, ConnectionError, TimeoutError, json.JSONDecodeError):
                pass
            time.sleep(5)
        raise TimeoutError(f"worker [{self.mode}] not ready after {timeout:.0f}s; see {self.log_path}")

    def generate(self, payload: dict, timeout: float) -> dict:
        return _http_json(f"{self.url}/generate", payload, timeout=timeout)

    def stop(self):
        if self.proc.poll() is None:
            os.killpg(self.proc.pid, signal.SIGTERM)
            try:
                self.proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)
                self.proc.wait()
        self.log.close()


# --------------------------------------------------------------------------- quality metrics


def _decode_luma(path: Path) -> tuple[np.ndarray, int, int]:
    probe = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                     "stream=width,height", "-of", "csv=p=0", str(path)], text=True)
    w, h = (int(v) for v in probe.strip().split(",")[:2])
    raw = subprocess.check_output(["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "gray",
                                   "-"])
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w).astype(np.float64), w, h


def _decode_audio(path: Path) -> np.ndarray | None:
    try:
        raw = subprocess.check_output(["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", "16000",
                                       "-f", "f32le", "-"])
    except subprocess.CalledProcessError:
        return None
    audio = np.frombuffer(raw, np.float32)
    return audio if audio.size else None


def _box(img: np.ndarray, r: int) -> np.ndarray:
    """Mean over a (2r+1)^2 window, valid region only, via integral image."""
    c = np.pad(img, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    k = 2 * r + 1
    return (c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]) / (k * k)


def _ssim(a: np.ndarray, b: np.ndarray, r: int = 3) -> float:
    c1, c2 = (0.01 * 255)**2, (0.03 * 255)**2
    mu_a, mu_b = _box(a, r), _box(b, r)
    var_a = _box(a * a, r) - mu_a**2
    var_b = _box(b * b, r) - mu_b**2
    cov = _box(a * b, r) - mu_a * mu_b
    s = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2))
    return float(s.mean())


def _log_spec(audio: np.ndarray, n_fft: int = 512, hop: int = 160) -> np.ndarray:
    frames = np.lib.stride_tricks.sliding_window_view(audio, n_fft)[::hop] * np.hanning(n_fft)
    return np.log10(np.abs(np.fft.rfft(frames, axis=-1)) + 1e-5)


def compare_videos(ref: Path, test: Path) -> dict:
    a, _, _ = _decode_luma(ref)
    b, _, _ = _decode_luma(test)
    n = min(len(a), len(b))
    mse = ((a[:n] - b[:n])**2).reshape(n, -1).mean(1)
    psnr = 10 * np.log10(255.0**2 / np.maximum(mse, 1e-10))
    step = max(1, n // 25)  # SSIM on ~25 evenly spaced frames keeps this fast
    ssim = [_ssim(a[i], b[i]) for i in range(0, n, step)]
    out = {"frames": int(n), "psnr_db": float(np.mean(psnr)), "psnr_min_db": float(np.min(psnr)),
           "ssim": float(np.mean(ssim))}
    aa, ab = _decode_audio(ref), _decode_audio(test)
    if aa is not None and ab is not None:
        m = min(len(aa), len(ab))
        if m > 1024:
            out["audio_logspec_l1"] = float(np.abs(_log_spec(aa[:m]) - _log_spec(ab[:m])).mean())
    return out


def side_by_side(ref: Path, test: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(ref), "-i", str(test), "-filter_complex",
                    "[0:v][1:v]hstack=inputs=2[v]", "-map", "[v]", "-map", "1:a?", "-c:v", "libx264", "-crf", "18",
                    "-pix_fmt", "yuv420p", str(dst)], check=False)


# --------------------------------------------------------------------------- main


def run_mode(mode: str, idx: int, args, jobs: list[dict], out: Path, extra_env: dict[str, str]) -> dict:
    mode_dir = out / mode
    mode_dir.mkdir(parents=True, exist_ok=True)
    sampler = GpuMemSampler()
    baseline = sampler.read_mib()
    sampler.start()
    worker = Worker(mode, args.worker, args.port + idx, mode_dir, extra_env)
    result: dict = {"mode": mode, "baseline_mib": baseline, "runs": []}
    try:
        print(f"[{mode}] loading ...", flush=True)
        result["load_s"] = worker.wait_ready(args.load_timeout)
        time.sleep(2)
        result["loaded_mib"] = sampler.read_mib() - baseline
        print(f"[{mode}] ready in {result['load_s']:.0f}s, {result['loaded_mib']:.0f} MiB", flush=True)

        warm = dict(jobs[0]["payload"], output_path=str(mode_dir / "warmup.mp4"))
        resp = worker.generate(warm, args.gen_timeout)
        result["warmup_s"] = resp.get("elapsed")
        if not resp.get("ok"):
            raise RuntimeError(f"warm-up failed: {str(resp.get('error'))[-2000:]}")

        for job in jobs:
            path = mode_dir / f"{job['name']}.mp4"
            t0 = time.time()
            resp = worker.generate(dict(job["payload"], output_path=str(path)), args.gen_timeout)
            t1 = time.time()
            peak = sampler.peak_between(t0, t1)
            run = {"name": job["name"], "ok": bool(resp.get("ok")), "elapsed_s": resp.get("elapsed"),
                   "wall_s": t1 - t0, "peak_mib": None if peak is None else peak - baseline, "video": str(path)}
            if not run["ok"]:
                run["error"] = str(resp.get("error"))[-2000:]
            result["runs"].append(run)
            print(f"[{mode}] {job['name']}: ok={run['ok']} {run['elapsed_s'] or 0:.1f}s", flush=True)
    except Exception as exc:  # noqa: BLE001 - keep benchmarking the other modes
        result["error"] = str(exc)
        print(f"[{mode}] FAILED: {exc}", flush=True)
    finally:
        worker.stop()
        sampler.stop()
    ok = [r["elapsed_s"] for r in result["runs"] if r["ok"] and r["elapsed_s"]]
    peaks = [r["peak_mib"] for r in result["runs"] if r["peak_mib"] is not None]
    result["mean_s"] = statistics.mean(ok) if ok else None
    result["peak_mib"] = max(peaks) if peaks else None
    return result


def write_report(out: Path, args, results: list[dict], quality: dict) -> None:
    base = next((r for r in results if r["mode"] == "bf16" and r.get("mean_s")), None)
    lines = [
        f"# Сравнение QUANT_MODE ({args.worker})", "",
        f"{len(args.prompts_list)} промптов × сиды {args.seeds}, {args.aspect}, {args.duration}s, steps={args.steps}. "
        "Время из ответа воркера (`elapsed`), память GPU-wide минус baseline.", "",
        "| режим | загрузка, с | VRAM после загрузки, GiB | пик VRAM, GiB | среднее время, с | ускорение | $ за видео |",
        "|---|---|---|---|---|---|---|",
    ]

    def fmt(v, spec):
        return "—" if v is None else format(v, spec)

    for r in results:
        speed = base["mean_s"] / r["mean_s"] if base and r.get("mean_s") else None
        cost = r["mean_s"] / 3600 * args.gpu_price_per_hour if r.get("mean_s") and args.gpu_price_per_hour else None
        gib = lambda m: None if m is None else m / 1024  # noqa: E731
        lines.append(f"| {r['mode']}{' ⚠ ' + r['error'][:60] if r.get('error') else ''} | {fmt(r.get('load_s'), '.0f')} | "
                     f"{fmt(gib(r.get('loaded_mib')), '.1f')} | {fmt(gib(r.get('peak_mib')), '.1f')} | "
                     f"{fmt(r.get('mean_s'), '.1f')} | {fmt(speed, '.2f')}× | {fmt(cost, '.4f')} |")
    if quality:
        lines += ["", "## Качество относительно bf16 (тот же промпт и сид)", "",
                  "PSNR/SSIM по яркости: выше — ближе к bf16. Аудио: L1 лог-спектрограмм, ниже — ближе. "
                  "Это мера отклонения от bf16, а не абсолютного качества: смотрите side_by_side/*.mp4.", "",
                  "| режим | ролик | PSNR, дБ | min PSNR, дБ | SSIM | аудио L1 |", "|---|---|---|---|---|---|"]
        for mode, rows in quality.items():
            for name, q in rows.items():
                if "error" in q:
                    lines.append(f"| {mode} | {name} | ошибка: {q['error'][:80]} | | | |")
                    continue
                lines.append(f"| {mode} | {name} | {q['psnr_db']:.2f} | {q['psnr_min_db']:.2f} | {q['ssim']:.4f} | "
                             f"{fmt(q.get('audio_logspec_l1'), '.3f')} |")
    (out / "report.md").write_text("\n".join(lines) + "\n")
    (out / "report.json").write_text(json.dumps({"args": {k: v for k, v in vars(args).items()},
                                                 "results": results, "quality": quality}, indent=2))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--worker", choices=["t2va", "ref2va"], default="t2va")
    p.add_argument("--modes", default="bf16,svdq", help="comma list; bf16 first (it is the quality reference)")
    p.add_argument("--prompts", help="text file, one prompt per line (default: 3 built-in)")
    p.add_argument("--reference-image", default="", help="required for --worker ref2va")
    p.add_argument("--seeds", default="0", help="comma list")
    p.add_argument("--aspect", choices=list(ASPECTS), default="9:16")
    p.add_argument("--duration", choices=["5", "10"], default="5")
    p.add_argument("--steps", type=int, default=9)
    p.add_argument("--port", type=int, default=18091)
    p.add_argument("--env", action="append", default=[], help="KEY=VALUE for every worker (e.g. T2VA_DIT_OFFLOAD=0)")
    p.add_argument("--gpu-price-per-hour", type=float, default=0.0)
    p.add_argument("--load-timeout", type=float, default=3600)
    p.add_argument("--gen-timeout", type=float, default=1800)
    p.add_argument("--out", default=f"/workspace/quant_bench/{time.strftime('%Y%m%d-%H%M%S')}")
    args = p.parse_args()

    if args.worker == "ref2va" and not args.reference_image:
        p.error("--worker ref2va needs --reference-image")
    args.prompts_list = (Path(args.prompts).read_text().strip().splitlines() if args.prompts else DEFAULT_PROMPTS)
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    seeds = [int(s) for s in args.seeds.split(",")]
    width, height = ASPECTS[args.aspect]
    num_frames = {"5": 125, "10": 250}[args.duration]
    extra_env = dict(kv.split("=", 1) for kv in args.env)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    jobs = [{"name": f"p{i}_s{seed}",
             "payload": {"prompt": prompt, "num_frames": num_frames, "steps": args.steps, "height": height,
                         "width": width, "seed": seed, "reference_image": args.reference_image}}
            for i, prompt in enumerate(args.prompts_list) for seed in seeds]

    results = [run_mode(mode, i, args, jobs, out, extra_env) for i, mode in enumerate(modes)]

    quality: dict = {}
    if "bf16" in modes:
        for mode in modes:
            if mode == "bf16":
                continue
            quality[mode] = {}
            for job in jobs:
                ref, test = out / "bf16" / f"{job['name']}.mp4", out / mode / f"{job['name']}.mp4"
                if not (ref.exists() and test.exists()):
                    continue
                try:
                    quality[mode][job["name"]] = compare_videos(ref, test)
                except Exception as exc:  # noqa: BLE001
                    quality[mode][job["name"]] = {"error": str(exc)}
                side_by_side(ref, test, out / "side_by_side" / f"{mode}_{job['name']}.mp4")

    write_report(out, args, results, quality)
    print((out / "report.md").read_text())
    return 0 if all(not r.get("error") for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
