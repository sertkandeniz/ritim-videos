"""Lip sync backends: LatentSync 1.6 / MuseTalk 1.5 (GPU workers), or a mock that marks the clip."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .. import media
from ..config import Config
from ..runner import run_worker


@dataclass
class LipsyncItem:
    id: str
    video: Path  # face crop of the original, 25 fps
    audio: Path  # dubbed voice for exactly that span
    out: Path


class LipsyncBackend(Protocol):
    name: str
    gpu: bool

    def run(self, items: list[LipsyncItem], job_path: Path) -> None: ...


class LatentSync:
    name = "latentsync"
    gpu = True

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    def run(self, items: list[LipsyncItem], job_path: Path) -> None:
        ls = self.cfg.lipsync.latentsync
        repo = self.cfg.repo_dir(ls.repo)
        ckpt = self.cfg.paths.models_root / "latentsync"
        job = {
            "options": {
                "repo": str(repo),
                "unet_config": "configs/unet/stage2_512.yaml",
                "inference_ckpt": str(ckpt / "latentsync_unet.pt"),
                "inference_steps": ls.inference_steps,
                "guidance_scale": ls.guidance_scale,
                "deepcache": ls.deepcache,
                "seed": 1247,
            },
            "items": [{"id": i.id, "video": str(i.video), "audio": str(i.audio), "out": str(i.out)}
                      for i in items if not i.out.exists()],
        }
        if job["items"]:
            run_worker(self.cfg, ls.env, "latentsync_worker.py", job, job_path, cwd=repo)


class MuseTalk:
    name = "musetalk"
    gpu = True

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    def run(self, items: list[LipsyncItem], job_path: Path) -> None:
        mt = self.cfg.lipsync.musetalk
        repo = self.cfg.repo_dir(mt.repo)
        job = {
            "options": {"repo": str(repo), "bbox_shift": mt.bbox_shift, "version": "v15"},
            "items": [{"id": i.id, "video": str(i.video), "audio": str(i.audio), "out": str(i.out)}
                      for i in items if not i.out.exists()],
        }
        if job["items"]:
            run_worker(self.cfg, mt.env, "musetalk_worker.py", job, job_path, cwd=repo)


class MockLipsync:
    """Draws a frame around the clip and muxes the dubbed audio, so overlays are visible in tests."""

    name = "mock"
    gpu = False

    def run(self, items: list[LipsyncItem], job_path: Path) -> None:
        for i in items:
            if i.out.exists():
                continue
            i.out.parent.mkdir(parents=True, exist_ok=True)
            tmp = i.out.with_suffix(".part.mp4")
            media.ffmpeg("-i", i.video, "-i", i.audio, "-map", "0:v", "-map", "1:a",
                         "-vf", "drawbox=x=0:y=0:w=iw:h=ih:color=lime@1.0:t=8",
                         "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", "-shortest", tmp)
            tmp.replace(i.out)


def make_lipsync(cfg: Config) -> LipsyncBackend | None:
    b = cfg.lipsync.backend
    if b == "latentsync":
        return LatentSync(cfg)
    if b == "musetalk":
        return MuseTalk(cfg)
    if b == "mock":
        return MockLipsync()
    return None
