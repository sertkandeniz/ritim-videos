"""Demucs htdemucs two-stem split (vocals / everything else). Env: audio (demucs==4.1.0)."""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from _common import atomic, load_job, run_items


def main() -> None:
    job, job_path = load_job()
    o = job["options"]

    def load() -> Any:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"

    def process(device: Any, it: dict[str, Any]) -> dict[str, Any]:
        # The CLI is demucs' stable interface; the model loads in ~2 s, small next to a 7-minute chapter.
        with tempfile.TemporaryDirectory(dir=Path(it["vocals"]).parent.parent) as tmp:
            cmd = [sys.executable, "-m", "demucs", "--two-stems", "vocals", "-n", o["model"],
                   "--segment", str(int(o["segment"])), "--shifts", str(o["shifts"]), "-d", device,
                   "-o", tmp, "--filename", "{stem}.{ext}", it["audio"]]
            subprocess.run(cmd, check=True)
            out = Path(tmp) / o["model"]
            with atomic(it["background"]) as bg:
                shutil.move(out / "no_vocals.wav", bg)
            with atomic(it["vocals"]) as vo:
                shutil.move(out / "vocals.wav", vo)
        return {}

    run_items(job, job_path, load, process, out_key="background")


if __name__ == "__main__":
    main()
