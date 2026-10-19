"""`lipsync` (face-crop clips on large-face segments) and `assemble` (mix, loudness, HLS).

HLS layout per lesson (output/<lesson>/hls/):
  video/index.m3u8 + seg_*.ts          the original picture, encoded once, fixed GOP = segment length
  video_<lang>/index.m3u8 + seg_*.ts   only for lip-synced languages: re-encoded segments where a
                                       face was lip-synced; every other entry points at ../video/
  audio_<lang>/index.m3u8              voice over separated background, EBU R128 normalised
  subs_<lang>/index.m3u8 + seg_*.vtt   segmented WebVTT
  master.m3u8                          original picture + every audio and subtitle track
  master_<lang>.m3u8                   lip-synced picture for <lang> (only where lip sync ran)
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .. import media
from ..backends.lipsync import LipsyncItem, make_lipsync
from ..manifest import Lesson, Timer, fingerprint_files, read_json, write_json
from ..subtitles import parse_vtt
from .base import Ctx, log
from .prepare import face_segments

LIPSYNC_FPS = 25


# ---------------------------------------------------------------- lesson tracks
def lesson_track(ctx: Ctx, kind: str, lesson_id: str, lang: str = "") -> Path:
    """Concatenate chapter wavs (voice/background/source) into one lesson-long wav."""
    chapters = ctx.store.data.chapters_of(lesson_id)
    if kind == "voice":
        parts = [ctx.cd.voice_track(lang, c.id) for c in chapters]
    elif kind == "background":
        parts = [ctx.cd.background(c.id) for c in chapters]
    else:
        parts = [ctx.cd.chapter_audio(c.id) for c in chapters]
    out = ctx.cd.tmp / f"{lesson_id}_{kind}{'_' + lang if lang else ''}.wav"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.stat().st_mtime >= max(p.stat().st_mtime for p in parts):
        return out
    listing = out.with_suffix(".txt")
    listing.write_text("".join(f"file '{p.resolve()}'\n" for p in parts), encoding="utf-8")
    media.ffmpeg("-f", "concat", "-safe", "0", "-i", listing, "-c:a", "pcm_s16le", out)
    listing.unlink()
    return out


# ---------------------------------------------------------------- lip sync
def _crop_geometry(box: list[float], lesson: Lesson, scale: float) -> tuple[int, int, int, int]:
    cx, cy = (box[0] + box[2]) / 2 * lesson.width, (box[1] + box[3]) / 2 * lesson.height
    side = max((box[2] - box[0]) * lesson.width, (box[3] - box[1]) * lesson.height) * scale
    w = min(lesson.width, int(side) // 2 * 2)
    h = min(lesson.height, int(side * 1.15) // 2 * 2)
    x = int(min(max(0, cx - w / 2), lesson.width - w)) // 2 * 2
    y = int(min(max(0, cy - h / 2), lesson.height - h)) // 2 * 2
    return x, y, w, h


def run_lipsync(ctx: Ctx) -> None:
    lcfg = ctx.cfg.lipsync
    m = ctx.store.data
    langs = [lang for lang in ctx.langs if ctx.cfg.language(lang).lipsync]
    if lcfg.backend == "none" or not langs:
        log.info("lipsync: disabled")
        return
    for lang in langs:
        with ctx.timed(f"lipsync:{lang}") as stats:
            backend = ctx.backend("lipsync", lambda: make_lipsync(ctx.cfg))
            if backend is None:
                return
            stats.backend, stats.gpu = backend.name, backend.gpu
            batch: list[tuple[Lesson, list[dict[str, Any]], list[LipsyncItem]]] = []
            for lesson in m.lessons:
                item = f"{lesson.id}:{lang}"
                if ctx.done("lipsync", item):
                    continue
                if not all(ctx.store.is_done("voice", f"{c.id}:{lang}") for c in m.chapters_of(lesson.id)):
                    log.warning("lipsync %s: voice not finished, skipping", item)
                    continue
                segs = face_segments(ctx, lesson.id)
                out_dir = ctx.cd.lipsync_dir(lang, lesson.id)
                out_dir.mkdir(parents=True, exist_ok=True)
                clips: list[dict[str, Any]] = []
                items: list[LipsyncItem] = []
                if segs:
                    voice = lesson_track(ctx, "voice", lesson.id, lang)
                    src = ctx.cd.source_video(lesson.source)
                    for si, seg in enumerate(segs):
                        a0 = max(0.0, seg["start"] - lcfg.pad_s)
                        b0 = min(lesson.duration, seg["end"] + lcfg.pad_s)
                        n = max(1, math.ceil((b0 - a0) / lcfg.max_chunk_s))
                        x, y, w, h = _crop_geometry(seg["box"], lesson, lcfg.crop_scale)
                        for k in range(n):
                            a = a0 + (b0 - a0) * k / n
                            b = a0 + (b0 - a0) * (k + 1) / n
                            cid = f"s{si + 1:03d}_{k + 1:02d}"
                            vin, ain, out = (out_dir / f"{cid}.in.mp4", out_dir / f"{cid}.in.wav",
                                             out_dir / f"{cid}.mp4")
                            if not out.exists():
                                media.ffmpeg("-ss", f"{a:.3f}", "-t", f"{b - a:.3f}", "-i", src, "-an",
                                             "-vf", f"crop={w}:{h}:{x}:{y},fps={LIPSYNC_FPS}",
                                             "-c:v", "libx264", "-preset", "fast", "-crf", "12",
                                             "-pix_fmt", "yuv420p", vin)
                                media.ffmpeg("-ss", f"{a:.3f}", "-t", f"{b - a:.3f}", "-i", voice,
                                             "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", ain)
                            clips.append({"id": cid, "start": round(a, 3), "end": round(b, 3),
                                          "x": x, "y": y, "w": w, "h": h, "file": out.name})
                            items.append(LipsyncItem(cid, vin, ain, out))
                batch.append((lesson, clips, items))
            all_items = [it for _, _, items in batch for it in items]
            timer = Timer()
            if all_items:
                backend.run(all_items, ctx.cd.jobs_dir / f"lipsync_{lang}.json")
            for lesson, clips, items in batch:
                missing = [it.id for it in items if not it.out.exists()]
                if missing:
                    log.error("lipsync %s:%s: no output for %s", lesson.id, lang, missing)
                    continue
                for it in items:
                    it.video.unlink(missing_ok=True)
                    it.audio.unlink(missing_ok=True)
                manifest_path = ctx.cd.lipsync_dir(lang, lesson.id) / "clips.json"
                write_json(manifest_path, {"lesson": lesson.id, "lang": lang, "backend": backend.name,
                                           "clips": clips})
                ctx.store.mark_done("lipsync", f"{lesson.id}:{lang}",
                                    [manifest_path] + [it.out for it in items], timer.elapsed,
                                    info={"clips": len(clips),
                                          "seconds": round(sum(c["end"] - c["start"] for c in clips), 1)})
                stats.items += 1
                stats.media_seconds += sum(c["end"] - c["start"] for c in clips)


# ---------------------------------------------------------------- HLS helpers
def _video_args(ctx: Ctx, fps: float) -> list[str]:
    a = ctx.cfg.assemble
    seg = a.hls_segment_s
    gop = max(1, round((fps or 25) * seg))
    encoder = a.video_encoder
    if encoder == "auto":
        encoder = "h264_nvenc" if media.has_encoder("h264_nvenc") else "libx264"
    common = ["-pix_fmt", "yuv420p", "-profile:v", "high", "-bf", "0", "-g", str(gop),
              "-force_key_frames", f"expr:gte(t,n_forced*{seg})"]
    if encoder == "h264_nvenc":
        return ["-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", str(a.video_quality), "-b:v", "0",
                "-forced-idr", "1", *common]
    return ["-c:v", "libx264", "-preset", "veryfast", "-crf", str(a.video_quality), "-keyint_min", str(gop),
            "-sc_threshold", "0", "-x264-params", "open-gop=0", *common]


TS_FLAGS = ["-muxdelay", "0", "-muxpreload", "0"]


def _hls_out(ctx: Ctx, out_dir: Path) -> list[str | Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    return [*TS_FLAGS, "-f", "hls", "-hls_time", str(ctx.cfg.assemble.hls_segment_s), "-hls_playlist_type",
            "vod", "-hls_flags", "independent_segments", "-hls_segment_type", "mpegts",
            "-hls_segment_filename", out_dir / "seg_%05d.ts", out_dir / "index.m3u8"]


def parse_playlist(path: Path) -> list[tuple[float, str]]:
    segs: list[tuple[float, str]] = []
    dur = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#EXTINF:"):
            dur = float(line[8:].split(",")[0])
        elif line and not line.startswith("#") and dur is not None:
            segs.append((dur, line))
            dur = None
    return segs


def first_pts(ts: Path, stream: str = "v:0") -> float:
    out = media.run([media.FFPROBE, "-v", "error", "-select_streams", stream, "-show_entries",
                     "packet=pts_time", "-read_intervals", "%+#1", "-of", "csv=p=0", ts]).stdout.split()
    return float(out[0]) if out else 0.0


def write_playlist(path: Path, segs: list[tuple[float, str]]) -> None:
    target = math.ceil(max(d for d, _ in segs)) if segs else 1
    lines = ["#EXTM3U", "#EXT-X-VERSION:3", f"#EXT-X-TARGETDURATION:{target}", "#EXT-X-MEDIA-SEQUENCE:0",
             "#EXT-X-PLAYLIST-TYPE:VOD", "#EXT-X-INDEPENDENT-SEGMENTS"]
    for d, uri in segs:
        lines += [f"#EXTINF:{d:.6f},", uri]
    lines.append("#EXT-X-ENDLIST")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _mix_audio(ctx: Ctx, voice: Path | None, background: Path | None, source: Path | None, out: Path) -> Path:
    a = ctx.cfg.assemble
    raw = out.with_name(out.stem + ".mix.wav")
    if source is not None:
        media.ffmpeg("-i", source, "-ac", "2", "-ar", "48000", "-c:a", "pcm_s16le", raw)
    else:
        assert voice is not None and background is not None
        fc = (f"[0:a]aresample=48000,pan=stereo|c0=c0|c1=c0[v];"
              f"[1:a]aresample=48000,aformat=channel_layouts=stereo,volume={a.background_gain_db}dB[b];"
              "[v][b]amix=inputs=2:normalize=0:duration=first[m]")
        media.ffmpeg("-i", voice, "-i", background, "-filter_complex", fc, "-map", "[m]",
                     "-c:a", "pcm_s16le", raw)
    norm = media.loudnorm_filter(raw, a.loudness_i, a.loudness_tp, a.loudness_lra)
    media.ffmpeg("-i", raw, "-af", norm, "-ar", "48000", "-c:a", "pcm_s16le", out)
    raw.unlink(missing_ok=True)
    return out


def _segment_vtt(vtt: Path, out_dir: Path, video_segs: list[tuple[float, str]], pts0: float) -> None:
    cues = parse_vtt(vtt.read_text(encoding="utf-8"))
    out_dir.mkdir(parents=True, exist_ok=True)
    segs: list[tuple[float, str]] = []
    t = 0.0
    header = f"WEBVTT\nX-TIMESTAMP-MAP=MPEGTS:{round(pts0 * 90000)},LOCAL:00:00:00.000\n\n"
    from ..subtitles import to_vtt

    for i, (dur, _) in enumerate(video_segs):
        inside = [c for c in cues if c.end > t and c.start < t + dur]
        body = to_vtt(inside).split("\n", 2)[2] if inside else ""
        name = f"seg_{i:05d}.vtt"
        (out_dir / name).write_text(header + body, encoding="utf-8")
        segs.append((dur, name))
        t += dur
    write_playlist(out_dir / "index.m3u8", segs)


def _patch_video(ctx: Ctx, lesson: Lesson, lang: str, hls: Path) -> int:
    """Re-encode only the video segments that overlap lip-synced clips; reuse all others."""
    clips_doc = read_json(ctx.cd.lipsync_dir(lang, lesson.id) / "clips.json")
    clips = clips_doc["clips"]
    base = parse_playlist(hls / "video" / "index.m3u8")
    out_dir = hls / f"video_{lang}"
    out_dir.mkdir(parents=True, exist_ok=True)
    src = ctx.cd.source_video(lesson.source)
    args = _video_args(ctx, lesson.fps)
    new: list[tuple[float, str]] = []
    t, patched = 0.0, 0
    for dur, uri in base:
        hits = [c for c in clips if c["end"] > t and c["start"] < t + dur]
        if not hits:
            new.append((dur, f"../video/{uri}"))
            t += dur
            continue
        inputs: list[str | Path] = ["-ss", f"{t:.6f}", "-t", f"{dur:.6f}", "-i", src]
        chain, last = [], "[0:v]"
        for k, c in enumerate(hits, 1):
            clip = ctx.cd.lipsync_dir(lang, lesson.id) / c["file"]
            offset = c["start"] - t
            if offset >= 0:
                inputs += ["-itsoffset", f"{offset:.6f}", "-i", clip]
            else:
                inputs += ["-ss", f"{-offset:.6f}", "-i", clip]
            s, e = max(0.0, offset), min(dur, c["end"] - t)
            chain.append(f"[{k}:v]scale={c['w']}:{c['h']},setsar=1[c{k}]")
            chain.append(f"{last}[c{k}]overlay={c['x']}:{c['y']}:eof_action=pass:"
                         f"enable='between(t,{s:.3f},{e:.3f})'[o{k}]")
            last = f"[o{k}]"
        pts0 = first_pts(hls / "video" / uri)
        tmp = out_dir / (uri + ".part")
        media.ffmpeg(*inputs, "-filter_complex", ";".join(chain), "-map", last, "-an", *args,
                     "-output_ts_offset", f"{pts0:.6f}", *TS_FLAGS, "-f", "mpegts", tmp)
        tmp.replace(out_dir / uri)
        new.append((dur, uri))
        patched += 1
        t += dur
    write_playlist(out_dir / "index.m3u8", new)
    return patched


def _lang_name(ctx: Ctx, lang: str) -> str:
    if lang == ctx.cfg.source_language:
        return "Türkçe"
    return ctx.cfg.language(lang).name


def _write_masters(ctx: Ctx, lesson: Lesson, hls: Path, audio_langs: list[str], sub_langs: list[str],
                   lip_langs: list[str]) -> list[Path]:
    seg_s = ctx.cfg.assemble.hls_segment_s

    def peak_bps(folder: Path) -> int:
        segs = parse_playlist(folder / "index.m3u8")
        rates = [(folder / uri).stat().st_size * 8 / max(d, 0.1) for d, uri in segs if (folder / uri).exists()]
        return int(max(rates, default=1_000_000))

    audio_bps = max((peak_bps(hls / f"audio_{lang}") for lang in audio_langs), default=0)

    def master(default_audio: str, video_dir: str) -> str:
        lines = ["#EXTM3U", "#EXT-X-VERSION:4", "#EXT-X-INDEPENDENT-SEGMENTS"]
        for lang in audio_langs:
            default = "YES" if lang == default_audio else "NO"
            lines.append(f'#EXT-X-MEDIA:TYPE=AUDIO,GROUP-ID="aud",LANGUAGE="{lang}",NAME="{_lang_name(ctx, lang)}",'
                         f'DEFAULT={default},AUTOSELECT=YES,URI="audio_{lang}/index.m3u8"')
        for lang in sub_langs:
            lines.append(f'#EXT-X-MEDIA:TYPE=SUBTITLES,GROUP-ID="subs",LANGUAGE="{lang}",'
                         f'NAME="{_lang_name(ctx, lang)}",DEFAULT=NO,AUTOSELECT=YES,FORCED=NO,'
                         f'URI="subs_{lang}/index.m3u8"')
        attrs = [f"BANDWIDTH={peak_bps(hls / video_dir.split('/')[0]) + audio_bps}",
                 f"RESOLUTION={lesson.width}x{lesson.height}", f"FRAME-RATE={lesson.fps:.3f}"]
        if audio_langs:
            attrs.append('AUDIO="aud"')
        if sub_langs:
            attrs.append('SUBTITLES="subs"')
        lines += ["#EXT-X-STREAM-INF:" + ",".join(attrs), f"{video_dir}/index.m3u8"]
        return "\n".join(lines) + "\n"

    first_audio = audio_langs[0] if audio_langs else ""
    written = [hls / "master.m3u8"]
    written[0].write_text(master(first_audio, "video"), encoding="utf-8")
    for lang in lip_langs:
        p = hls / f"master_{lang}.m3u8"
        p.write_text(master(lang, f"video_{lang}"), encoding="utf-8")
        written.append(p)
    log.info("assemble %s: %d audio, %d subtitle tracks, lip-synced %s (segment %ss)", lesson.id,
             len(audio_langs), len(sub_langs), lip_langs or "none", seg_s)
    return written


# ---------------------------------------------------------------- assemble
def run_assemble(ctx: Ctx) -> None:
    a = ctx.cfg.assemble
    m = ctx.store.data
    src_lang = ctx.cfg.source_language
    with ctx.timed("assemble") as stats:
        stats.backend = "ffmpeg"
        for lesson in m.lessons:
            chapters = m.chapters_of(lesson.id)
            hls = ctx.cd.hls_dir(lesson.id)
            src = ctx.cd.source_video(lesson.source)
            timer = Timer()
            work = 0

            if not ctx.done("assemble", f"video:{lesson.id}"):
                media.ffmpeg("-i", src, "-map", "0:v:0", "-an", *_video_args(ctx, lesson.fps),
                             *_hls_out(ctx, hls / "video"))
                ctx.store.mark_done("assemble", f"video:{lesson.id}", [hls / "video" / "index.m3u8"])
                work += 1
            video_segs = parse_playlist(hls / "video" / "index.m3u8")
            pts0 = first_pts(hls / "video" / video_segs[0][1]) if video_segs else 0.0

            audio_langs: list[str] = []
            if a.include_source_audio and lesson.has_audio:
                item = f"audio:{lesson.id}:{src_lang}"
                if not ctx.done("assemble", item):
                    wav = _mix_audio(ctx, None, None, lesson_track(ctx, "source", lesson.id),
                                     ctx.cd.tmp / f"{lesson.id}_{src_lang}_final.wav")
                    media.ffmpeg("-i", wav, "-c:a", "aac", "-b:a", a.audio_bitrate, "-ac", "2",
                                 *_hls_out(ctx, hls / f"audio_{src_lang}"))
                    wav.unlink(missing_ok=True)
                    ctx.store.mark_done("assemble", item, [hls / f"audio_{src_lang}" / "index.m3u8"])
                    work += 1
                audio_langs.append(src_lang)
            for lang in ctx.langs:
                voices = [ctx.cd.voice_track(lang, c.id) for c in chapters]
                if not all(ctx.store.is_done("voice", f"{c.id}:{lang}") for c in chapters):
                    continue
                item = f"audio:{lesson.id}:{lang}"
                fp = fingerprint_files(*voices, extra=f"{a.loudness_i}:{a.background_gain_db}")
                if not ctx.done("assemble", item, fp):
                    wav = _mix_audio(ctx, lesson_track(ctx, "voice", lesson.id, lang),
                                     lesson_track(ctx, "background", lesson.id), None,
                                     ctx.cd.tmp / f"{lesson.id}_{lang}_final.wav")
                    media.ffmpeg("-i", wav, "-c:a", "aac", "-b:a", a.audio_bitrate, "-ac", "2",
                                 *_hls_out(ctx, hls / f"audio_{lang}"))
                    wav.unlink(missing_ok=True)
                    ctx.store.mark_done("assemble", item, [hls / f"audio_{lang}" / "index.m3u8"], fingerprint=fp)
                    work += 1
                audio_langs.append(lang)

            sub_langs: list[str] = []
            for lang in [src_lang, *ctx.langs]:
                vtt = ctx.cd.subtitle(lang, lesson.id, "vtt")
                if not vtt.exists():
                    continue
                item = f"subs:{lesson.id}:{lang}"
                fp = fingerprint_files(vtt)
                if not ctx.done("assemble", item, fp):
                    _segment_vtt(vtt, hls / f"subs_{lang}", video_segs, pts0)
                    ctx.store.mark_done("assemble", item, [hls / f"subs_{lang}" / "index.m3u8"], fingerprint=fp)
                    work += 1
                sub_langs.append(lang)

            lip_langs: list[str] = []
            for lang in ctx.langs:
                clips_json = ctx.cd.lipsync_dir(lang, lesson.id) / "clips.json"
                if not (ctx.store.is_done("lipsync", f"{lesson.id}:{lang}") and clips_json.exists()):
                    continue
                if not read_json(clips_json)["clips"]:
                    continue
                item = f"lipvideo:{lesson.id}:{lang}"
                fp = fingerprint_files(clips_json)
                if not ctx.done("assemble", item, fp):
                    n = _patch_video(ctx, lesson, lang, hls)
                    ctx.store.mark_done("assemble", item, [hls / f"video_{lang}" / "index.m3u8"], fingerprint=fp,
                                        info={"patched_segments": n})
                    work += 1
                lip_langs.append(lang)

            masters = _write_masters(ctx, lesson, hls, audio_langs, sub_langs, lip_langs)
            ctx.store.mark_done("assemble", f"master:{lesson.id}", masters, timer.elapsed,
                                info={"audio": audio_langs, "subtitles": sub_langs, "lipsync": lip_langs})
            if work:
                stats.items += 1
                stats.media_seconds += lesson.duration
