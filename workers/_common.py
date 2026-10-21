"""Shared helpers for worker scripts. Standard library only: workers run in model envs without ritim."""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {Path(sys.argv[0]).stem}: {msg}", flush=True)


def load_job() -> tuple[dict[str, Any], Path]:
    if len(sys.argv) != 2:
        sys.exit(f"usage: {sys.argv[0]} JOB.json")
    path = Path(sys.argv[1])
    return json.loads(path.read_text(encoding="utf-8")), path


@contextmanager
def atomic(path: str | Path) -> Iterator[Path]:
    """Yield a temp path next to `path` (same extension); rename it into place on success."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.stem + ".part" + target.suffix)
    try:
        yield tmp
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            tmp.unlink()


def run_items(job: dict[str, Any], job_path: Path, load: Callable[[], Any],
              process: Callable[[Any, dict[str, Any]], dict[str, Any] | None], out_key: str = "out") -> None:
    """Load the model once, process every item whose output is missing, write the result file."""
    items = job["items"]
    todo = [it for it in items if not Path(it[out_key]).exists()]
    results: dict[str, Any] = {it["id"]: {"skipped": True} for it in items if it not in todo}
    load_s = 0.0
    device = "none"
    if todo:
        t0 = time.time()
        model = load()
        load_s = time.time() - t0
        device = _device()
        log(f"model loaded in {load_s:.1f}s on {device}; {len(todo)} items")
        for n, it in enumerate(todo, 1):
            t1 = time.time()
            try:
                info = process(model, it) or {}
            except Exception:
                traceback.print_exc()
                results[it["id"]] = {"error": traceback.format_exc(limit=3)}
                continue
            results[it["id"]] = {"seconds": round(time.time() - t1, 2), **info}
            log(f"{n}/{len(todo)} {it['id']} {time.time() - t1:.1f}s")
    out = job_path.with_name(job_path.stem + ".result.json")
    out.write_text(json.dumps({"items": results, "load_seconds": round(load_s, 2), "device": device}, indent=1))
    if any("error" in r for r in results.values()):
        sys.exit(3)


def _device() -> str:
    try:
        import torch

        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except Exception:
        return "unknown"


def save_wav(path: Path, wav: Any, sr: int) -> None:
    import numpy as np
    import soundfile as sf

    data = np.asarray(wav, dtype="float32").squeeze()
    sf.write(str(path), data, sr, subtype="PCM_16")
