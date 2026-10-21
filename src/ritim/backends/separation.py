"""Voice/background separation: Demucs htdemucs two-stem split, or passthrough for tests."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from .. import media
from ..config import Config
from ..runner import run_worker


@dataclass
class SepItem:
    id: str
    audio: Path
    vocals: Path
    background: Path


class Separator(Protocol):
    name: str
    gpu: bool

    def separate(self, items: list[SepItem], job_path: Path) -> None: ...


class DemucsSeparator:
    name = "demucs"
    gpu = True

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    def separate(self, items: list[SepItem], job_path: Path) -> None:
        s = self.cfg.separate
        job = {
            "options": {"model": s.model, "segment": s.segment, "shifts": s.shifts},
            "items": [{"id": i.id, "audio": str(i.audio), "vocals": str(i.vocals),
                       "background": str(i.background)} for i in items],
        }
        run_worker(self.cfg, s.env, "demucs_worker.py", job, job_path)


class PassthroughSeparator:
    """No model: vocals = input, background = silence. Used in tests and as an emergency fallback."""

    name = "passthrough"
    gpu = False

    def separate(self, items: list[SepItem], job_path: Path) -> None:
        for i in items:
            i.vocals.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(i.audio, i.vocals)
            data, sr = media.read_audio(i.audio, mono=False)
            media.write_audio(i.background, np.zeros_like(data), sr)


def make_separator(cfg: Config) -> Separator:
    return DemucsSeparator(cfg) if cfg.separate.backend == "demucs" else PassthroughSeparator()
