"""MuseTalk 1.5 (MIT), the fast lip-sync option. Env: musetalk. Runs with cwd = the MuseTalk checkout,
whose ./models is a symlink to <models_root>/musetalk (made by infra/bootstrap.sh).

MuseTalk's own CLI already processes many tasks per model load, so we hand it one config with all clips.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from _common import load_job, log


def main() -> None:
    job, job_path = load_job()
    o = job["options"]
    repo = Path(o["repo"])
    todo = [it for it in job["items"] if not Path(it["out"]).exists()]
    results: dict[str, dict[str, object]] = {it["id"]: {"skipped": True} for it in job["items"] if it not in todo}
    if todo:
        work = job_path.with_name(job_path.stem + "_musetalk")
        work.mkdir(parents=True, exist_ok=True)
        lines = []
        for n, it in enumerate(todo):
            lines += [f"task_{n}:", f'  video_path: "{it["video"]}"', f'  audio_path: "{it["audio"]}"',
                      f"  bbox_shift: {int(o.get('bbox_shift', 0))}", f'  result_name: "{it["id"]}.mp4"']
        cfg = work / "tasks.yaml"
        cfg.write_text("\n".join(lines) + "\n", encoding="utf-8")
        ffmpeg_dir = str(Path(shutil.which("ffmpeg") or "/usr/bin/ffmpeg").parent)
        cmd = [sys.executable, "-m", "scripts.inference", "--inference_config", str(cfg), "--result_dir",
               str(work), "--unet_model_path", "models/musetalkV15/unet.pth", "--unet_config",
               "models/musetalkV15/musetalk.json", "--version", o.get("version", "v15"), "--use_float16",
               "--ffmpeg_path", ffmpeg_dir]
        log(" ".join(cmd))
        subprocess.run(cmd, cwd=repo, check=True)
        for it in todo:
            produced = work / o.get("version", "v15") / f"{it['id']}.mp4"
            if produced.exists():
                Path(it["out"]).parent.mkdir(parents=True, exist_ok=True)
                shutil.move(produced, it["out"])
                results[it["id"]] = {}
            else:
                results[it["id"]] = {"error": f"MuseTalk wrote no {produced}"}
    import json

    job_path.with_name(job_path.stem + ".result.json").write_text(json.dumps({"items": results}, indent=1))
    if any("error" in r for r in results.values()):
        sys.exit(3)


if __name__ == "__main__":
    main()
