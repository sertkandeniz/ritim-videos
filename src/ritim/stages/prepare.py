"""`inventory` and `chapters`: probe the source videos, find pauses/scenes/faces, cut chapters."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from .. import media
from ..backends.faces import make_face_detector
from ..manifest import Lesson, Timer, read_json, write_json
from ..planning import FaceSegment, large_face_segments, plan_chapters
from .base import Ctx, log

VIDEO_EXT = (".mp4", ".mov", ".mkv", ".m4v", ".webm")


def _natural_key(name: str) -> list[Any]:
    return [int(p) if p.isdigit() else p.lower() for p in re.split(r"(\d+)", name)]


def _register_lessons(ctx: Ctx) -> None:
    """Assign stable lesson ids (L01, L02, ...) to source videos in natural filename order."""
    m = ctx.store.data
    meta = ctx.cd.source / "course.yaml"
    if meta.exists():
        info = yaml.safe_load(meta.read_text(encoding="utf-8")) or {}
        m.title = str(info.get("title", m.title))
        m.description = str(info.get("description", m.description))
    known = {lesson.source for lesson in m.lessons}
    files = sorted((p for p in ctx.cd.source.iterdir() if p.suffix.lower() in VIDEO_EXT),
                   key=lambda p: _natural_key(p.name)) if ctx.cd.source.exists() else []
    if not files:
        raise RuntimeError(f"no videos in {ctx.cd.source}; run `ritim pull` or copy them there")
    for f in files:
        rel = str(f.relative_to(ctx.cd.root))
        if rel in known:
            continue
        p = media.probe(f)
        m.lessons.append(Lesson(id=f"L{len(m.lessons) + 1:02d}", source=rel, duration=round(p.duration, 3),
                                width=p.width, height=p.height, fps=round(p.fps, 3), has_audio=p.has_audio))
    if not m.title:
        m.title = ctx.cd.course_id
    ctx.store.save()


def run_inventory(ctx: Ctx) -> None:
    _register_lessons(ctx)
    inv = ctx.cfg.inventory
    with ctx.timed("inventory") as stats:
        stats.backend = inv.face_backend
        detector = None
        for lesson in ctx.store.data.lessons:
            if ctx.done("inventory", lesson.id):
                continue
            timer = Timer()
            src = ctx.cd.source_video(lesson.source)
            silences = media.detect_silences(src, inv.silence_db, inv.silence_min_s) if lesson.has_audio else []
            scenes = media.detect_scenes(src, inv.scene_threshold, inv.scene_fps)
            samples: list[dict[str, Any]] = []
            if inv.face_backend != "none":
                detector = detector or ctx.backend(
                    "faces", lambda: make_face_detector(inv.face_backend, ctx.cfg.paths.models_root))
                for t, frame in media.iter_frames(src, inv.face_sample_fps):
                    face = detector.largest_face(frame)
                    samples.append({"t": round(t, 3), "h": face[0] if face else 0.0,
                                    "box": face[1] if face else None})
            segments = large_face_segments(samples, inv.large_face_min_height, inv.min_face_segment_s,
                                           1.0 / inv.face_sample_fps)
            out = ctx.cd.inventory(lesson.id)
            write_json(out, {
                "lesson": lesson.model_dump(),
                "silences": silences,
                "scenes": scenes,
                "face_samples": samples,
                "large_face_segments": segments,
            })
            ctx.store.mark_done("inventory", lesson.id, [out], timer.elapsed,
                                info={"silences": len(silences), "scenes": len(scenes),
                                      "large_face_s": round(sum(s["end"] - s["start"] for s in segments), 1)})
            stats.items += 1
            stats.media_seconds += lesson.duration
            log.info("inventory %s: %d pauses, %d scene cuts, %d large-face segments",
                     lesson.id, len(silences), len(scenes), len(segments))
        if detector is not None:
            detector.close()


def face_segments(ctx: Ctx, lesson_id: str) -> list[FaceSegment]:
    data = read_json(ctx.cd.inventory(lesson_id))
    segs: list[FaceSegment] = data.get("large_face_segments", [])
    return segs


def run_chapters(ctx: Ctx) -> None:
    c = ctx.cfg.chapters
    m = ctx.store.data
    with ctx.timed("chapters") as stats:
        stats.backend = "ffmpeg"
        for lesson in m.lessons:
            plan_key = f"plan:{lesson.id}"
            if not ctx.store.is_done("chapters", plan_key):
                if any(ch.lesson_id == lesson.id for ch in m.chapters):
                    raise RuntimeError(f"{lesson.id} already has chapters; refusing to re-plan "
                                       "(delete the work dir to start over)")
                inv = read_json(ctx.cd.inventory(lesson.id))
                bounds = plan_chapters(lesson.duration, [tuple(s) for s in inv["silences"]], inv["scenes"],
                                       c.min_s, c.target_s, c.max_s, inv.get("large_face_segments", []))
                from ..manifest import Chapter

                for i, (a, b) in enumerate(bounds, 1):
                    m.chapters.append(Chapter(id=f"{lesson.id}-C{i:03d}", lesson_id=lesson.id, index=i,
                                              start=a, end=b))
                ctx.store.mark_done("chapters", plan_key, info={"chapters": len(bounds)})
            src = ctx.cd.source_video(lesson.source)
            for ch in m.chapters_of(lesson.id):
                if ctx.done("chapters", ch.id):
                    continue
                timer = Timer()
                out = ctx.cd.chapter_audio(ch.id)
                if lesson.has_audio:
                    media.extract_audio(src, out, ch.start, ch.end, sr=ctx.cfg.voice.sample_rate, channels=2)
                else:
                    media.ffmpeg("-f", "lavfi", "-i", f"anullsrc=r={ctx.cfg.voice.sample_rate}:cl=stereo",
                                 "-t", f"{ch.duration:.3f}", "-c:a", "pcm_s16le", out)
                ctx.store.mark_done("chapters", ch.id, [out], timer.elapsed)
                stats.items += 1
                stats.media_seconds += ch.duration


def chapter_paths(ctx: Ctx) -> list[tuple[str, Path]]:
    return [(ch.id, ctx.cd.chapter_audio(ch.id)) for ch in ctx.store.data.chapters]
