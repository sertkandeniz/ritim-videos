"""Pipeline stages in run order. Each reads/writes the course work dir and is resumable per item."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .audio import run_separate, run_transcribe
from .base import Ctx
from .prepare import run_chapters, run_inventory
from .qa import run_qa
from .text import run_subtitles, run_translate
from .video import run_assemble, run_lipsync
from .voice import run_voice

STAGES: dict[str, Callable[[Ctx], Any]] = {
    "inventory": run_inventory,
    "chapters": run_chapters,
    "transcribe": run_transcribe,
    "separate": run_separate,
    "translate": run_translate,
    "subtitles": run_subtitles,
    "voice": run_voice,
    "lipsync": run_lipsync,
    "assemble": run_assemble,
    "qa": run_qa,
}

# `voice` may shorten lines, which changes their text and timing, so subtitles run again after it
# (they are fingerprinted, so the second pass only redoes lessons whose translations changed).
RUN_ORDER = ["inventory", "chapters", "transcribe", "separate", "translate", "subtitles", "voice",
             "subtitles", "lipsync", "assemble", "qa"]

__all__ = ["RUN_ORDER", "STAGES", "Ctx"]
