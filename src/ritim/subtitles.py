"""Cue building and WebVTT/SRT writers."""

from __future__ import annotations

from dataclasses import dataclass

from .timing import CJK_LANGS

CJK_BREAK_AFTER = "、。，！？；：）」』"


@dataclass
class Cue:
    start: float
    end: float
    text: str


def _wrap_words(text: str, width: int) -> list[str]:
    lines: list[str] = []
    cur = ""
    for word in text.split():
        cand = f"{cur} {word}".strip()
        if len(cand) <= width or not cur:
            cur = cand
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def _wrap_cjk(text: str, width: int) -> list[str]:
    text = text.strip()
    lines: list[str] = []
    while len(text) > width:
        cut = width
        for i in range(width, max(width // 2, 1), -1):
            if text[i - 1] in CJK_BREAK_AFTER:
                cut = i
                break
        lines.append(text[:cut].strip())
        text = text[cut:].strip()
    if text:
        lines.append(text)
    return lines


def wrap(text: str, width: int, lang: str) -> list[str]:
    return _wrap_cjk(text, width) if lang in CJK_LANGS else _wrap_words(text, width)


def _balance(lines: list[str], width: int, lang: str) -> list[str]:
    """Two lines of similar length read better than one full line and a short tail."""
    if len(lines) != 2 or lang in CJK_LANGS:
        return lines
    words = " ".join(lines).split()
    best = lines
    best_diff = abs(len(lines[0]) - len(lines[1]))
    for i in range(1, len(words)):
        a, b = " ".join(words[:i]), " ".join(words[i:])
        if len(a) <= width and len(b) <= width and abs(len(a) - len(b)) < best_diff:
            best, best_diff = [a, b], abs(len(a) - len(b))
    return best


def make_cues(units: list[dict[str, object]], lang: str, line_chars: int, max_lines: int = 2,
              min_dur: float = 0.8, max_dur: float = 7.0, gap: float = 0.04) -> list[Cue]:
    """One cue per unit, split into several cues if the text needs more than `max_lines` lines."""
    cues: list[Cue] = []
    for u in units:
        text = str(u.get("text", "")).strip()
        if not text:
            continue
        start = float(u["start"])  # type: ignore[arg-type]
        end = float(u.get("subtitle_end", u["end"]))  # type: ignore[arg-type]
        lines = wrap(text, line_chars, lang)
        chunks = [lines[i : i + max_lines] for i in range(0, len(lines), max_lines)]
        total = sum(len("".join(c)) for c in chunks) or 1
        t = start
        for chunk in chunks:
            share = (end - start) * len("".join(chunk)) / total
            chunk_end = min(t + max(share, min_dur), t + max_dur)
            cues.append(Cue(t, chunk_end, "\n".join(_balance(chunk, line_chars, lang))))
            t = chunk_end
    cues.sort(key=lambda c: c.start)
    for a, b in zip(cues, cues[1:], strict=False):
        if a.end > b.start - gap:
            a.end = max(a.start + 0.2, b.start - gap)
    return cues


def _ts(seconds: float, sep: str) -> str:
    ms = int(round(max(0.0, seconds) * 1000))
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def to_vtt(cues: list[Cue]) -> str:
    out = ["WEBVTT", ""]
    for i, c in enumerate(cues, 1):
        out += [str(i), f"{_ts(c.start, '.')} --> {_ts(c.end, '.')}", c.text, ""]
    return "\n".join(out)


def to_srt(cues: list[Cue]) -> str:
    out: list[str] = []
    for i, c in enumerate(cues, 1):
        out += [str(i), f"{_ts(c.start, ',')} --> {_ts(c.end, ',')}", c.text, ""]
    return "\n".join(out)


def parse_vtt(text: str) -> list[Cue]:
    cues: list[Cue] = []
    blocks = text.replace("\r\n", "\n").split("\n\n")
    for block in blocks:
        lines = [ln for ln in block.split("\n") if ln.strip()]
        for i, ln in enumerate(lines):
            if "-->" in ln:
                a, b = (x.strip().split(" ")[0] for x in ln.split("-->"))
                cues.append(Cue(_parse_ts(a), _parse_ts(b), "\n".join(lines[i + 1 :])))
                break
    return cues


def _parse_ts(ts: str) -> float:
    ts = ts.replace(",", ".")
    parts = ts.split(":")
    secs = float(parts[-1])
    mins = int(parts[-2]) if len(parts) >= 2 else 0
    hours = int(parts[-3]) if len(parts) >= 3 else 0
    return hours * 3600 + mins * 60 + secs
