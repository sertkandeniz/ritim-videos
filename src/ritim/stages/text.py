"""`translate` (glossary + length budgets + validation rounds) and `subtitles` (WebVTT/SRT per lesson)."""

from __future__ import annotations

from typing import Any

from ..backends.translation import CourseContext, LineIn, Translator, make_translator
from ..glossary import Glossary
from ..manifest import Timer, fingerprint_files, read_json, write_json
from ..subtitles import make_cues, to_srt, to_vtt
from ..timing import assign_slots, char_budget, text_length
from .base import Ctx, log

BATCH_UNITS = 80  # lines per request; a 10-minute chapter has ~60-110
OVER_BUDGET_TOLERANCE = 1.05


def _problems(unit: dict[str, Any], text: str | None, lang: str, glossary: Glossary) -> str:
    if text is None or not text.strip():
        return "missing"
    issues = glossary.check(unit["source"], text, lang)
    n = text_length(text, lang)
    if n > unit["max_chars"] * OVER_BUDGET_TOLERANCE:
        issues.insert(0, f"too long: {n} characters, max {unit['max_chars']}")
    return "; ".join(issues)


def translate_units(translator: Translator, glossary: Glossary, lang: str, units: list[dict[str, Any]],
                    context: list[tuple[str, str]], course: CourseContext, rounds: int) -> None:
    """Fill unit["text"] (and unit["issues"] for anything still wrong after `rounds` revisions)."""
    for start in range(0, len(units), BATCH_UNITS):
        batch = units[start : start + BATCH_UNITS]
        got = translator.translate(lang, [LineIn(u["id"], u["source"], u["max_chars"]) for u in batch],
                                   context, course)
        for u in batch:
            u["text"] = got.get(u["id"], "")
        for _ in range(rounds):
            bad = [(u, p) for u in batch if (p := _problems(u, u["text"], lang, glossary))]
            if not bad:
                break
            missing = [u for u, p in bad if p == "missing"]
            if missing:
                again = translator.translate(lang, [LineIn(u["id"], u["source"], u["max_chars"])
                                                    for u in missing], context, course)
                for u in missing:
                    u["text"] = again.get(u["id"], "")
            fix = [LineIn(u["id"], u["source"], u["max_chars"], current=u["text"], problem=p)
                   for u, p in bad if p != "missing"]
            if fix:
                revised = translator.revise(lang, fix, course)
                for u, _p in bad:
                    if revised.get(u["id"]):
                        u["text"] = revised[u["id"]]
        for u in batch:
            problem = _problems(u, u["text"], lang, glossary)
            u["issues"] = [problem] if problem else []
        context = [(u["source"], u["text"]) for u in batch[-4:]]


def run_translate(ctx: Ctx) -> None:
    t = ctx.cfg.translate
    m = ctx.store.data
    course = CourseContext(m.title or ctx.cd.course_id, m.description)
    translator: Translator | None = None
    for lang in ctx.langs:
        lcfg = ctx.cfg.language(lang)
        with ctx.timed(f"translate:{lang}") as stats:
            for lesson in m.lessons:
                context: list[tuple[str, str]] = []
                for ch in m.chapters_of(lesson.id):
                    out = ctx.cd.translation(lang, ch.id)
                    item = f"{ch.id}:{lang}"
                    if ctx.done("translate", item):
                        prev = read_json(out)["units"]
                        context = [(u["source"], u["text"]) for u in prev[-t.context_units:]]
                        continue
                    if translator is None:
                        translator = ctx.backend("translator", lambda: make_translator(ctx.cfg, ctx.glossary))
                    stats.backend = translator.name
                    timer = Timer()
                    tr = read_json(ctx.cd.transcript(ch.id))
                    units = [{"id": u["id"], "start": u["start"], "end": u["end"], "source": u["text"]}
                             for u in tr["units"] if u["text"].strip()]
                    assign_slots(units, ch.duration, t.max_extend_s, ctx.cfg.voice.min_gap_s)
                    for u in units:
                        u["max_chars"] = char_budget(u, lcfg.chars_per_second, t.budget_factor)
                    translate_units(translator, ctx.glossary, lang, units, context, course, t.validation_rounds)
                    write_json(out, {"chapter": ch.id, "lang": lang, "backend": translator.name,
                                     "model": t.model if translator.name == "anthropic" else translator.name,
                                     "units": units})
                    flagged = sum(1 for u in units if u["issues"])
                    ctx.store.mark_done("translate", item, [out], timer.elapsed,
                                        info={"units": len(units), "flagged": flagged})
                    if flagged:
                        log.warning("translate %s: %d of %d lines still flagged", item, flagged, len(units))
                    context = [(u["source"], u["text"]) for u in units[-t.context_units:]]
                    stats.items += 1
                    stats.media_seconds += ch.duration
            if translator is not None:
                stats.usage = translator.usage.as_dict()


def lesson_units(ctx: Ctx, lesson_id: str, lang: str) -> list[dict[str, Any]]:
    """All units of a lesson in lesson time; `lang` == source language reads the transcripts."""
    units: list[dict[str, Any]] = []
    for ch in ctx.store.data.chapters_of(lesson_id):
        if lang == ctx.cfg.source_language:
            src = read_json(ctx.cd.transcript(ch.id))["units"]
            rows = [{"start": u["start"], "end": u["end"], "text": u["text"]} for u in src]
        else:
            rows = [dict(u) for u in read_json(ctx.cd.translation(lang, ch.id))["units"]]
        for u in rows:
            u["start"] = round(u["start"] + ch.start, 3)
            u["end"] = round(u["end"] + ch.start, 3)
            if "slot_end" in u:
                u["slot_end"] = round(u["slot_end"] + ch.start, 3)
            if "placed_start" in u:  # subtitles follow the dubbed voice when it was shifted
                u["start"] = round(u["placed_start"] + ch.start, 3)
                u["end"] = round(u["placed_end"] + ch.start, 3)
            units.append(u)
    return units


def run_subtitles(ctx: Ctx) -> None:
    m = ctx.store.data
    for lang in [ctx.cfg.source_language, *ctx.langs]:
        line_chars = 42 if lang == ctx.cfg.source_language else ctx.cfg.language(lang).subtitle_line_chars
        with ctx.timed(f"subtitles:{lang}") as stats:
            stats.backend = "local"
            for lesson in m.lessons:
                chapters = m.chapters_of(lesson.id)
                if lang == ctx.cfg.source_language:
                    inputs = [ctx.cd.transcript(c.id) for c in chapters]
                else:
                    inputs = [ctx.cd.translation(lang, c.id) for c in chapters]
                if not all(p.exists() for p in inputs):
                    continue
                item = f"{lesson.id}:{lang}"
                fp = fingerprint_files(*inputs, extra=str(line_chars))
                if ctx.done("subtitles", item, fp):
                    continue
                cues = make_cues(lesson_units(ctx, lesson.id, lang), lang, line_chars)
                vtt = ctx.cd.subtitle(lang, lesson.id, "vtt")
                srt = ctx.cd.subtitle(lang, lesson.id, "srt")
                vtt.parent.mkdir(parents=True, exist_ok=True)
                vtt.write_text(to_vtt(cues), encoding="utf-8")
                srt.write_text(to_srt(cues), encoding="utf-8")
                ctx.store.mark_done("subtitles", item, [vtt, srt], fingerprint=fp, info={"cues": len(cues)})
                stats.items += 1
