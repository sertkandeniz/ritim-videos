"""Run a worker script inside one of the per-model virtualenvs.

Protocol: the orchestrator writes `jobs/<name>.json` = {"options": {...}, "items": [{"id", ...}]}.
The worker loads its model once, processes every item (skipping items whose output already
exists), writes each output atomically and finally `jobs/<name>.result.json` =
{"items": {id: {...}}, "load_seconds": float, "device": str}. A crash mid-job loses at most one
item; the next run picks up the rest because finished outputs are on disk.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .config import REPO_ROOT, Config
from .manifest import read_json, write_json

log = logging.getLogger(__name__)

WORKERS_DIR = REPO_ROOT / "workers"


class WorkerError(RuntimeError):
    pass


def worker_env(cfg: Config) -> dict[str, str]:
    env = dict(os.environ)
    hf_home = cfg.paths.models_root / "hf"
    env.update(
        HF_HOME=str(hf_home),
        HF_HUB_OFFLINE=env.get("HF_HUB_OFFLINE", "1"),  # weights are pre-fetched by bootstrap
        TRANSFORMERS_OFFLINE=env.get("TRANSFORMERS_OFFLINE", "1"),
        TORCH_HOME=str(cfg.paths.models_root / "torch"),
        RITIM_MODELS=str(cfg.paths.models_root),
        PYTHONUNBUFFERED="1",
        PYTHONNOUSERSITE="1",
        # 4 vCPU box: keep CPU-side thread pools from oversubscribing
        OMP_NUM_THREADS=env.get("OMP_NUM_THREADS", "4"),
    )
    return env


def run_worker(cfg: Config, env_name: str, script: str, job: dict[str, Any], job_path: Path,
               cwd: Path | None = None, workers_dir: Path | None = None,
               python: Path | None = None) -> dict[str, Any]:
    py = python or cfg.env_python(env_name)
    if not py.exists():
        raise WorkerError(f"env {env_name!r} is not installed ({py}); run infra/bootstrap.sh")
    script_path = (workers_dir or WORKERS_DIR) / script
    write_json(job_path, job)
    result_path = job_path.with_name(job_path.stem + ".result.json")
    result_path.unlink(missing_ok=True)
    cmd = [str(py), str(script_path), str(job_path)]
    log.info("worker %s in env %s: %d items", script, env_name, len(job.get("items", [])))
    proc = subprocess.run(cmd, cwd=cwd, env=worker_env(cfg), check=False)
    if proc.returncode != 0:
        raise WorkerError(f"worker {script} exited with {proc.returncode}")
    if not result_path.exists():
        raise WorkerError(f"worker {script} wrote no result file {result_path}")
    result: dict[str, Any] = read_json(result_path)
    return result


@contextmanager
def gpu_sampler(out: dict[str, float], interval_s: float = 5.0) -> Iterator[None]:
    """Sample GPU utilisation/memory with nvidia-smi while a stage runs (no-op without a GPU)."""
    smi = shutil.which("nvidia-smi")
    if smi is None:
        yield
        return
    util: list[float] = []
    mem: list[float] = []
    stop = threading.Event()

    def loop() -> None:
        while not stop.is_set():
            try:
                line = subprocess.run(
                    [smi, "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=10, check=False,
                ).stdout.strip().splitlines()[0]
                u, m = (float(x) for x in line.split(","))
                util.append(u)
                mem.append(m)
            except Exception:  # sampling must never break a stage
                pass
            stop.wait(interval_s)

    t = threading.Thread(target=loop, daemon=True)
    started = time.monotonic()
    t.start()
    try:
        yield
    finally:
        stop.set()
        t.join(timeout=15)
        if util:
            out["gpu_util_avg"] = round(sum(util) / len(util), 1)
            out["gpu_mem_peak_mb"] = max(mem)
            out["gpu_seconds"] = round(time.monotonic() - started, 1)
