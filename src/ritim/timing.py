"""Length budgets for translations and the tempo/shorten decision for synthesized lines."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

CJK_LANGS = ("ja", "zh", "ko")


def text_length(text: str, lang: str) -> int:
    """Characters that cost speaking time: CJK counts non-space characters, others everything."""
    if lang in CJK_LANGS:
        return sum(1 for ch in text if not ch.isspace())
    return len(text.strip())


def assign_slots(units: list[dict[str, Any]], chapter_duration: float, max_extend_s: float,
                 min_gap_s: float) -> None:
    """Give every unit a `slot_end`: how long its dubbed line may run (into the following pause)."""
    for i, u in enumerate(units):
        limit = units[i + 1]["start"] - min_gap_s if i + 1 < len(units) else chapter_duration
        u["slot_end"] = round(max(u["end"], min(u["end"] + max_extend_s, limit)), 3)


def char_budget(unit: dict[str, Any], cps: float, factor: float) -> int:
    return max(4, math.floor(cps * (unit["slot_end"] - unit["start"]) * factor))


@dataclass
class FitDecision:
    action: str  # "ok" | "tempo" | "shorten"
    tempo: float


def decide_fit(duration: float, slot: float, max_tempo: float) -> FitDecision:
    if slot <= 0:
        return FitDecision("shorten", float("inf"))
    ratio = duration / slot
    if ratio <= 1.0:
        return FitDecision("ok", 1.0)
    if ratio <= max_tempo:
        return FitDecision("tempo", ratio)
    return FitDecision("shorten", ratio)


def shorter_budget(text: str, lang: str, duration: float, slot: float, max_tempo: float) -> int:
    """Characters a re-translation may use so that, sped up by max_tempo, it fits the slot."""
    target = slot * max_tempo * 0.97
    return max(3, math.floor(text_length(text, lang) * target / max(duration, 1e-3)))
