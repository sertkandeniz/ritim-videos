"""Speech recognition backends. All return raw word lists; the transcribe stage builds units."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import httpx

from .. import media
from ..config import Config, require_env
from ..manifest import read_json, write_json
from ..runner import run_worker
from ..segmentation import Word, fill_missing_times

log = logging.getLogger(__name__)

ELEVENLABS_API = "https://api.elevenlabs.io"


@dataclass
class AsrItem:
    id: str
    audio: Path
    raw_out: Path  # backend-specific raw JSON kept for debugging


class AsrBackend(Protocol):
    name: str
    gpu: bool

    def transcribe(self, items: list[AsrItem], job_path: Path) -> dict[str, list[Word]]: ...


class WhisperXAsr:
    name = "whisperx"
    gpu = True

    def __init__(self, cfg: Config, keyterms: list[str]) -> None:
        self.cfg = cfg
        self.keyterms = keyterms

    def transcribe(self, items: list[AsrItem], job_path: Path) -> dict[str, list[Word]]:
        t = self.cfg.transcribe
        job = {
            "options": {
                "model": t.model,
                "compute_type": t.compute_type,
                "batch_size": t.batch_size,
                "language": self.cfg.source_language,
                "vad_method": t.vad_method,
                "align": t.align,
                "initial_prompt": t.initial_prompt,
                "download_root": str(self.cfg.paths.models_root / "whisper"),
            },
            "items": [{"id": i.id, "audio": str(i.audio), "out": str(i.raw_out)} for i in items],
        }
        run_worker(self.cfg, t.env, "whisperx_worker.py", job, job_path)
        out: dict[str, list[Word]] = {}
        for i in items:
            raw = read_json(i.raw_out)
            words: list[Word] = []
            for seg in raw.get("segments", []):
                words += fill_missing_times(seg.get("words", []), float(seg["start"]), float(seg["end"]))
            out[i.id] = words
        return out


class ScribeAsr:
    """ElevenLabs Scribe over HTTP (POST /v1/speech-to-text), no GPU needed."""

    name = "scribe"
    gpu = False

    def __init__(self, cfg: Config, keyterms: list[str], client: httpx.Client | None = None) -> None:
        self.cfg = cfg
        self.keyterms = keyterms[:50]
        self.client = client or httpx.Client(base_url=ELEVENLABS_API, timeout=600)

    def transcribe(self, items: list[AsrItem], job_path: Path) -> dict[str, list[Word]]:
        key = require_env("ELEVENLABS_API_KEY")
        out: dict[str, list[Word]] = {}
        for item in items:
            if item.raw_out.exists():
                raw = read_json(item.raw_out)
            else:
                upload = item.raw_out.with_suffix(".16k.wav")
                media.to_wav(item.audio, upload, sr=16000, channels=1)
                data: dict[str, Any] = {
                    "model_id": self.cfg.transcribe.scribe_model,
                    "language_code": self.cfg.source_language,
                    "timestamps_granularity": "word",
                    "tag_audio_events": "false",
                    "diarize": "false",
                }
                if self.keyterms:
                    data["keyterms"] = self.keyterms
                with upload.open("rb") as fh:
                    resp = self.client.post(
                        "/v1/speech-to-text", headers={"xi-api-key": key}, data=data,
                        files={"file": (upload.name, fh, "audio/wav")},
                    )
                upload.unlink(missing_ok=True)
                resp.raise_for_status()
                raw = resp.json()
                write_json(item.raw_out, raw)
            out[item.id] = [
                Word(word=w["text"].strip(), start=float(w["start"]), end=float(w["end"]))
                for w in raw.get("words", [])
                if w.get("type", "word") == "word" and w.get("start") is not None and w["text"].strip()
            ]
        return out


class MockAsr:
    """Deterministic fake Turkish: eight words per sentence, a protected term every sentence."""

    name = "mock"
    gpu = False
    VOCAB = ("bu", "eğitimde", "süreç", "denetimi", "için", "kök", "neden", "analizi", "yapıyoruz")
    TERMS = ("FMEA", "8D", "VDA 6.3", "IATF 16949")

    def transcribe(self, items: list[AsrItem], job_path: Path) -> dict[str, list[Word]]:
        out: dict[str, list[Word]] = {}
        for item in items:
            dur = media.audio_duration(item.audio)
            words: list[Word] = []
            t, n = 0.3, 0
            while t + 0.4 < dur - 0.3:
                k = n % 8
                text = self.TERMS[(n // 8) % len(self.TERMS)] if k == 3 else self.VOCAB[n % len(self.VOCAB)]
                if k == 7:
                    text += "."
                words.append(Word(word=text, start=round(t, 3), end=round(t + 0.32, 3)))
                t += 0.4 + (0.9 if k == 7 else 0.0)  # a pause after each sentence
                n += 1
            out[item.id] = words
            write_json(item.raw_out, {"words": list(words)})
        return out


def make_asr(cfg: Config, keyterms: list[str]) -> AsrBackend:
    backend = cfg.transcribe.backend
    if backend == "whisperx":
        return WhisperXAsr(cfg, keyterms)
    if backend == "scribe":
        return ScribeAsr(cfg, keyterms)
    return MockAsr()
