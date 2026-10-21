"""`transcribe` (ASR -> word timings -> sentence units) and `separate` (Demucs voice/background)."""

from __future__ import annotations

from ..backends.asr import AsrItem, make_asr
from ..backends.separation import SepItem, make_separator
from ..manifest import Timer, write_json
from ..segmentation import build_units
from .base import Ctx


def run_transcribe(ctx: Ctx) -> None:
    t = ctx.cfg.transcribe
    pending = [ch for ch in ctx.store.data.chapters if not ctx.done("transcribe", ch.id)]
    with ctx.timed("transcribe") as stats:
        if not pending:
            return
        asr = ctx.backend("asr", lambda: make_asr(ctx.cfg, ctx.glossary.never_translate))
        stats.backend, stats.gpu = asr.name, asr.gpu
        items = [AsrItem(ch.id, ctx.cd.chapter_audio(ch.id), ctx.cd.root / "transcripts" / f"{ch.id}.raw.json")
                 for ch in pending]
        timer = Timer()
        words_by_id = asr.transcribe(items, ctx.cd.jobs_dir / "transcribe.json")
        per_item = timer.elapsed / max(1, len(pending))
        for ch in pending:
            words = words_by_id.get(ch.id)
            if words is None:
                continue
            units = build_units(words, t.max_unit_s, t.unit_pause_s)
            out = ctx.cd.transcript(ch.id)
            write_json(out, {"chapter": ch.id, "backend": asr.name, "language": ctx.cfg.source_language,
                             "duration": round(ch.duration, 3), "words": words, "units": units})
            ctx.store.mark_done("transcribe", ch.id, [out], per_item,
                                info={"words": len(words), "units": len(units)})
            stats.items += 1
            stats.media_seconds += ch.duration


def run_separate(ctx: Ctx) -> None:
    pending = [ch for ch in ctx.store.data.chapters if not ctx.done("separate", ch.id)]
    with ctx.timed("separate") as stats:
        if not pending:
            return
        sep = ctx.backend("separator", lambda: make_separator(ctx.cfg))
        stats.backend, stats.gpu = sep.name, sep.gpu
        items = [SepItem(ch.id, ctx.cd.chapter_audio(ch.id), ctx.cd.vocals(ch.id), ctx.cd.background(ch.id))
                 for ch in pending]
        timer = Timer()
        sep.separate(items, ctx.cd.jobs_dir / "separate.json")
        per_item = timer.elapsed / len(pending)
        for item, ch in zip(items, pending, strict=True):
            if item.vocals.exists() and item.background.exists():
                ctx.store.mark_done("separate", ch.id, [item.vocals, item.background], per_item)
                stats.items += 1
                stats.media_seconds += ch.duration
