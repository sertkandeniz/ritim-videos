"""Turn word-timed ASR output into sentence units: the atoms we translate, subtitle and dub."""

from __future__ import annotations

from typing import Any, TypedDict

SENTENCE_END = (".", "?", "!", "…", "。", "？", "！")
SOFT_BREAK = (",", ";", ":", "，", "、")


class Word(TypedDict):
    word: str
    start: float
    end: float


class Unit(TypedDict):
    id: str
    start: float
    end: float
    text: str


def fill_missing_times(words: list[dict[str, Any]], seg_start: float, seg_end: float) -> list[Word]:
    """WhisperX leaves start/end off tokens it cannot align (numbers, symbols); interpolate them."""
    out: list[Word] = []
    n = len(words)
    for i, w in enumerate(words):
        text = str(w.get("word", w.get("text", ""))).strip()
        if not text:
            continue
        start = w.get("start")
        end = w.get("end")
        if start is None or end is None:
            prev_end = out[-1]["end"] if out else seg_start
            nxt = next((x for x in words[i + 1 :] if x.get("start") is not None), None)
            next_start = float(nxt["start"]) if nxt else seg_end
            remaining = sum(1 for x in words[i:] if x.get("start") is None) or 1
            step = max(0.0, (next_start - prev_end) / max(1, min(remaining, n - i)))
            start = prev_end
            end = prev_end + step
        out.append(Word(word=text, start=round(float(start), 3), end=round(float(end), 3)))
    return out


def _split_long(words: list[Word], max_s: float) -> list[list[Word]]:
    if len(words) < 2 or words[-1]["end"] - words[0]["start"] <= max_s:
        return [words]
    mid = (words[0]["start"] + words[-1]["end"]) / 2
    best, best_score = 1, float("-inf")
    for i in range(1, len(words)):
        gap = words[i]["start"] - words[i - 1]["end"]
        soft = 0.6 if words[i - 1]["word"].endswith(SOFT_BREAK) else 0.0
        centre = 1.0 - abs(words[i]["start"] - mid) / max_s
        score = gap * 2 + soft + centre
        if score > best_score:
            best, best_score = i, score
    return _split_long(words[:best], max_s) + _split_long(words[best:], max_s)


def build_units(words: list[Word], max_unit_s: float = 12.0, pause_s: float = 0.8) -> list[Unit]:
    groups: list[list[Word]] = []
    cur: list[Word] = []
    for i, w in enumerate(words):
        cur.append(w)
        nxt = words[i + 1] if i + 1 < len(words) else None
        gap = (nxt["start"] - w["end"]) if nxt else 0.0
        if nxt is None or w["word"].endswith(SENTENCE_END) or gap >= pause_s:
            groups.append(cur)
            cur = []
    units: list[Unit] = []
    for group in groups:
        for part in _split_long(group, max_unit_s):
            units.append(
                Unit(
                    id=f"u{len(units) + 1:04d}",
                    start=part[0]["start"],
                    end=part[-1]["end"],
                    text=" ".join(w["word"] for w in part),
                )
            )
    return units
