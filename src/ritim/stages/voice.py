"""`voice`: synthesize every line, fit it to its slot (small tempo change, else a shorter line)."""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import numpy as np

from .. import media
from ..backends.translation import CourseContext, LineIn, Translator, make_translator
from ..backends.tts import TTSBackend, TTSItem, make_tts
from ..config import Config
from ..manifest import Chapter, Timer, read_json, write_json
from ..timing import decide_fit, shorter_budget
from .base import Ctx, log


def _take_duration(path: Path, sr: int, cache: dict[Path, tuple[np.ndarray, float]]) -> float:
    if path not in cache:
        audio, _ = media.read_audio(path, sr=sr)
        audio = media.trim_silence(audio, sr)
        cache[path] = (audio, audio.size / sr)
    return cache[path][1]


def _items(ctx: Ctx, lang: str, ch: Chapter, units: list[dict[str, Any]]) -> list[TTSItem]:
    take_dir = ctx.cd.voice_take_dir(lang, ch.id)
    out = []
    for i, u in enumerate(units):
        out.append(TTSItem(
            id=f"{ch.id}:{u['id']}", text=u["text"], lang=lang, out=take_dir / f"{u['id']}.wav",
            prev_text=units[i - 1]["text"] if i else "",
            next_text=units[i + 1]["text"] if i + 1 < len(units) else "",
        ))
    return out


def run_voice(ctx: Ctx) -> None:
    v = ctx.cfg.voice
    sr = v.sample_rate
    m = ctx.store.data
    course = CourseContext(m.title or ctx.cd.course_id, m.description)
    for lang in ctx.langs:
        pending = [ch for ch in m.chapters
                   if ctx.cd.translation(lang, ch.id).exists() and not ctx.done("voice", f"{ch.id}:{lang}")]
        with ctx.timed(f"voice:{lang}") as stats:
            if not pending:
                continue
            tts: TTSBackend = ctx.backend(f"tts:{lang}",
                                          functools.partial(make_tts, ctx.cfg, ctx.cfg.tts_backend_for(lang)))
            stats.backend, stats.gpu = tts.name, tts.gpu
            translator: Translator | None = None
            timer = Timer()
            docs = {ch.id: read_json(ctx.cd.translation(lang, ch.id)) for ch in pending}
            units_of = {ch.id: [u for u in docs[ch.id]["units"] if u.get("text", "").strip()] for ch in pending}
            items = {it.id: it for ch in pending for it in _items(ctx, lang, ch, units_of[ch.id])}
            tts.synthesize(list(items.values()), ctx.cd.jobs_dir / f"voice_{lang}.json")
            cache: dict[Path, tuple[np.ndarray, float]] = {}

            for round_no in range(v.shorten_rounds + 1):
                too_long: list[tuple[Chapter, dict[str, Any], float, float]] = []
                for ch in pending:
                    for u in units_of[ch.id]:
                        it = items[f"{ch.id}:{u['id']}"]
                        dur = _take_duration(it.out, sr, cache)
                        slot = u["slot_end"] - u["start"]
                        if decide_fit(dur, slot, v.max_tempo).action == "shorten":
                            too_long.append((ch, u, dur, slot))
                if not too_long or round_no == v.shorten_rounds:
                    break
                if translator is None:
                    translator = ctx.backend("translator", lambda: make_translator(ctx.cfg, ctx.glossary))
                lines = [LineIn(u["id"], u["source"], shorter_budget(u["text"], lang, dur, slot, v.max_tempo),
                                current=u["text"],
                                problem=f"spoken it takes {dur:.1f}s but must fit {slot * v.max_tempo:.1f}s")
                         for _ch, u, dur, slot in too_long]
                revised = translator.revise(lang, lines, course)
                redo: list[TTSItem] = []
                for ch, u, _dur, _slot in too_long:
                    new = revised.get(u["id"], "").strip()
                    if not new or new == u["text"]:
                        continue
                    u.setdefault("text_before_shortening", u["text"])
                    u["text"] = new
                    u["shorten_rounds"] = u.get("shorten_rounds", 0) + 1
                    it = items[f"{ch.id}:{u['id']}"]
                    it.text = new
                    it.out.unlink(missing_ok=True)
                    cache.pop(it.out, None)
                    redo.append(it)
                log.info("voice %s round %d: %d lines too long, %d re-synthesized", lang, round_no + 1,
                         len(too_long), len(redo))
                if not redo:
                    break
                tts.synthesize(redo, ctx.cd.jobs_dir / f"voice_{lang}_r{round_no + 1}.json")

            per_chapter = timer.elapsed / len(pending)
            for ch in pending:
                report = _render_chapter(ctx, lang, ch, units_of[ch.id], items, cache)
                docs[ch.id]["voice_backend"] = tts.name
                write_json(ctx.cd.translation(lang, ch.id), docs[ch.id])
                track = ctx.cd.voice_track(lang, ch.id)
                rep_path = ctx.cd.voice_report(lang, ch.id)
                write_json(rep_path, report)
                ctx.store.mark_done("voice", f"{ch.id}:{lang}", [track, rep_path], per_chapter,
                                    info={k: report[k] for k in ("lines", "tempo_changed", "overflow",
                                                                 "max_drift_s")})
                stats.items += 1
                stats.media_seconds += ch.duration
            stats.usage = dict(tts.usage)
            if translator is not None:
                stats.usage.update({f"llm_{k}": val for k, val in translator.usage.as_dict().items()})


def _render_chapter(ctx: Ctx, lang: str, ch: Chapter, units: list[dict[str, Any]],
                    items: dict[str, TTSItem], cache: dict[Path, tuple[np.ndarray, float]]) -> dict[str, Any]:
    takes = {u["id"]: items[f"{ch.id}:{u['id']}"].out for u in units}
    report = render_track(ctx.cfg, units, takes, ch.duration, ctx.cd.voice_track(lang, ch.id),
                          ctx.cd.voice_take_dir(lang, ch.id), cache)
    return {"chapter": ch.id, "lang": lang, **report}


def render_track(cfg: Config, units: list[dict[str, Any]], takes: dict[str, Path], duration: float,
                 out: Path, tmp_dir: Path, cache: dict[Path, tuple[np.ndarray, float]] | None = None,
                 ) -> dict[str, Any]:
    """Place every take on a silent track at its unit start, sped up where needed; never overlapping."""
    v = cfg.voice
    sr = v.sample_rate
    cache = {} if cache is None else cache
    total = round(duration * sr)
    track = np.zeros(total, np.float32)
    prev_end = 0.0
    rows = []
    tempo_changed = overflow = 0
    max_drift = 0.0
    tmp_dir.mkdir(parents=True, exist_ok=True)
    for u in units:
        take = takes[u["id"]]
        dur = _take_duration(take, sr, cache)
        audio = cache[take][0]
        slot = u["slot_end"] - u["start"]
        fit = decide_fit(dur, slot, v.max_tempo)
        tempo = 1.0 if fit.action == "ok" else min(fit.tempo, v.hard_max_tempo)
        if tempo > 1.0 + 1e-3:
            trimmed = tmp_dir / f"{u['id']}.trim.wav"
            fitted = tmp_dir / f"{u['id']}.fit.wav"
            media.write_audio(trimmed, audio, sr)
            media.change_tempo(trimmed, fitted, tempo, v.tempo_filter)
            audio, _ = media.read_audio(fitted, sr=sr)
            trimmed.unlink(missing_ok=True)
            fitted.unlink(missing_ok=True)
            tempo_changed += 1
        start = max(u["start"], prev_end + v.min_gap_s)
        a = round(start * sr)
        b = min(total, a + audio.size)
        if a < total:
            track[a:b] += audio[: b - a]
        end = start + audio.size / sr
        if end > u["slot_end"] + 0.05:
            overflow += 1
        drift = start - u["start"]
        max_drift = max(max_drift, drift)
        u["placed_start"], u["placed_end"], u["tempo"] = round(start, 3), round(end, 3), round(tempo, 3)
        rows.append({"id": u["id"], "start": u["start"], "slot_end": u["slot_end"], "placed_start": round(start, 3),
                     "placed_end": round(end, 3), "raw_s": round(dur, 3), "tempo": round(tempo, 3),
                     "action": fit.action, "chars": len(u["text"]), "shorten_rounds": u.get("shorten_rounds", 0)})
        prev_end = end
    media.write_audio(out, track, sr)
    return {"lines": len(rows), "tempo_changed": tempo_changed, "overflow": overflow,
            "max_drift_s": round(max_drift, 3), "units": rows}
