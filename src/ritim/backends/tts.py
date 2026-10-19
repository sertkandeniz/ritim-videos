"""Text-to-speech backends: ElevenLabs (API), Qwen3-TTS / Chatterbox / CosyVoice (GPU workers), mock."""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import httpx
import numpy as np

from .. import media
from ..config import Config, require_env
from ..runner import run_worker
from ..timing import text_length

log = logging.getLogger(__name__)

ELEVENLABS_API = "https://api.elevenlabs.io"


@dataclass
class TTSItem:
    id: str
    text: str
    lang: str
    out: Path  # wav
    prev_text: str = ""
    next_text: str = ""


class TTSBackend(Protocol):
    name: str
    gpu: bool
    usage: dict[str, float]

    def synthesize(self, items: list[TTSItem], job_path: Path) -> None:
        """Write one wav per item; items whose output exists are skipped."""


class ElevenLabsTTS:
    name = "elevenlabs"
    gpu = False

    def __init__(self, cfg: Config, client: httpx.Client | None = None, voice_id: str | None = None) -> None:
        self.cfg = cfg
        self.el = cfg.voice.elevenlabs
        self.voice_id = voice_id or self.el.voice_id
        self.client = client or httpx.Client(base_url=ELEVENLABS_API, timeout=120)
        self.usage: dict[str, float] = {"characters": 0.0, "requests": 0.0}

    def _one(self, item: TTSItem, key: str) -> None:
        body: dict[str, Any] = {
            "text": item.text,
            "model_id": self.el.model_id,
            "voice_settings": {
                "stability": self.el.stability,
                "similarity_boost": self.el.similarity_boost,
                "style": self.el.style,
                "use_speaker_boost": self.el.use_speaker_boost,
            },
            "previous_text": item.prev_text or None,
            "next_text": item.next_text or None,
        }
        if self.el.send_language_code:
            body["language_code"] = item.lang
        delay = 2.0
        for attempt in range(6):
            resp = self.client.post(
                f"/v1/text-to-speech/{self.voice_id}",
                params={"output_format": self.el.output_format},
                headers={"xi-api-key": key, "accept": "audio/mpeg"},
                json=body,
            )
            if resp.status_code in (429, 500, 502, 503, 504) and attempt < 5:
                time.sleep(delay)
                delay *= 2
                continue
            resp.raise_for_status()
            break
        mp3 = item.out.with_suffix(".mp3")
        mp3.parent.mkdir(parents=True, exist_ok=True)
        mp3.write_bytes(resp.content)
        media.to_wav(mp3, item.out, sr=self.cfg.voice.sample_rate, channels=1)
        mp3.unlink(missing_ok=True)
        self.usage["characters"] += len(item.text)
        self.usage["requests"] += 1

    def synthesize(self, items: list[TTSItem], job_path: Path) -> None:
        if not self.voice_id:
            raise RuntimeError("voice.elevenlabs.voice_id is empty (set ELEVENLABS_VOICE_ID)")
        key = require_env("ELEVENLABS_API_KEY")
        todo = [i for i in items if not i.out.exists()]
        with ThreadPoolExecutor(max_workers=max(1, self.el.concurrency)) as pool:
            for f in [pool.submit(self._one, i, key) for i in todo]:
                f.result()


class WorkerTTS:
    """Self-hosted model in its own env; one model load per call, all items in one job."""

    gpu = True

    def __init__(self, cfg: Config, name: str) -> None:
        self.cfg = cfg
        self.name = name
        self.usage: dict[str, float] = {"characters": 0.0}

    def _options(self) -> tuple[str, str, dict[str, Any], Path | None]:
        v = self.cfg.voice
        ref = v.reference
        common: dict[str, Any] = {"ref_audio": str(ref.audio), "ref_text": ref.text(), "ref_language": ref.language,
                                  "sample_rate": v.sample_rate}
        if self.name == "qwen3tts":
            q = v.qwen3tts
            return q.env, "qwen3tts_worker.py", {**common, "model": q.model,
                                                 "attn_implementation": q.attn_implementation}, None
        if self.name == "chatterbox":
            c = v.chatterbox
            return c.env, "chatterbox_worker.py", {**common, "model": c.model, "t3_model": c.t3_model,
                                                   "exaggeration": c.exaggeration,
                                                   "cfg_weight": c.cfg_weight}, None
        if self.name == "cosyvoice":
            c2 = v.cosyvoice
            model_dir = self.cfg.paths.models_root / "cosyvoice" / c2.model.split("/")[-1]
            repo = self.cfg.repo_dir(c2.repo)
            return c2.env, "cosyvoice_worker.py", {**common, "model_dir": str(model_dir),
                                                   "repo": str(repo)}, repo
        raise ValueError(self.name)

    def synthesize(self, items: list[TTSItem], job_path: Path) -> None:
        todo = [i for i in items if not i.out.exists()]
        if not todo:
            return
        env, script, options, cwd = self._options()
        if not Path(options["ref_audio"]).exists():
            raise RuntimeError(f"voice reference {options['ref_audio']} missing; run `ritim fetch-voice`")
        job = {"options": options,
               "items": [{"id": i.id, "text": i.text, "lang": i.lang, "out": str(i.out)} for i in todo]}
        run_worker(self.cfg, env, script, job, job_path, cwd=cwd)
        self.usage["characters"] += sum(len(i.text) for i in todo)


class MockTTS:
    """Tone whose length follows the text length; `slowness` > 1 simulates a verbose voice."""

    gpu = False

    def __init__(self, cfg: Config, slowness: float = 1.0, name: str = "mock") -> None:
        self.cfg = cfg
        self.slowness = slowness
        self.name = name
        self.usage: dict[str, float] = {"characters": 0.0}
        self.calls = 0

    def synthesize(self, items: list[TTSItem], job_path: Path) -> None:
        self.calls += 1
        sr = self.cfg.voice.sample_rate
        for i in items:
            if i.out.exists():
                continue
            cps = self.cfg.language(i.lang).chars_per_second
            dur = max(0.2, text_length(i.text, i.lang) / cps * self.slowness)
            t = np.arange(int(sr * dur)) / sr
            freq = 180 + (sum(map(ord, self.name)) % 5) * 20
            tone = 0.3 * np.sin(2 * np.pi * freq * t).astype(np.float32)
            pad = np.zeros(int(sr * 0.15), np.float32)  # engines pad output; the stage trims it
            media.write_audio(i.out, np.concatenate([pad, tone, pad]), sr)
            self.usage["characters"] += len(i.text)


def make_tts(cfg: Config, backend: str) -> TTSBackend:
    if backend == "elevenlabs":
        return ElevenLabsTTS(cfg)
    if backend in ("qwen3tts", "chatterbox", "cosyvoice"):
        return WorkerTTS(cfg, backend)
    if backend == "mock":
        return MockTTS(cfg)
    raise ValueError(f"unknown TTS backend {backend!r}")
