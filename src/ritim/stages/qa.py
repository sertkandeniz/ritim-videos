"""`qa`: timing (overlap/gap/drift/tempo), glossary presence, subtitle reading speed, cost/time report."""

from __future__ import annotations

import html
import itertools
from collections import defaultdict
from typing import Any

from ..manifest import read_json, write_json
from ..subtitles import parse_vtt
from ..timing import CJK_LANGS, text_length
from .base import Ctx, log

MAX_READING_CPS = {"ja": 9.0, "zh": 9.0}  # others: 21 (Netflix-style adult limits are 17-20)


def _speech_stats(report: dict[str, Any]) -> tuple[int, float]:
    chars = sum(u["chars"] for u in report["units"])
    secs = sum(u["raw_s"] for u in report["units"])
    return chars, secs


def run_qa(ctx: Ctx) -> dict[str, Any]:
    q = ctx.cfg.qa
    m = ctx.store.data
    issues: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}

    def flag(severity: str, lang: str, where: str, kind: str, detail: str) -> None:
        issues.append({"severity": severity, "lang": lang, "where": where, "kind": kind, "detail": detail})

    for lang in ctx.langs:
        lang_sum: dict[str, Any] = defaultdict(float)
        chars_total, speech_total = 0, 0.0
        for ch in m.chapters:
            tpath = ctx.cd.translation(lang, ch.id)
            if tpath.exists():
                for u in read_json(tpath)["units"]:
                    where = f"{ch.id}/{u['id']}"
                    lang_sum["lines"] += 1
                    for problem in u.get("issues", []):
                        flag("error" if "protected" in problem else "warn", lang, where, "translation", problem)
                    for problem in ctx.glossary.check(u["source"], u.get("text", ""), lang):
                        if problem not in u.get("issues", []):
                            flag("warn", lang, where, "glossary", problem)
                    if u.get("text", "").strip() == u["source"].strip() and len(u["source"]) > 12:
                        flag("warn", lang, where, "untranslated", u["source"][:60])
            rpath = ctx.cd.voice_report(lang, ch.id)
            if not rpath.exists():
                continue
            rep = read_json(rpath)
            c, s = _speech_stats(rep)
            chars_total += c
            speech_total += s
            prev_end = None
            for u in rep["units"]:
                where = f"{ch.id}/{u['id']}"
                if prev_end is not None and u["placed_start"] < prev_end - q.max_overlap_s:
                    flag("error", lang, where, "overlap", f"starts {prev_end - u['placed_start']:.2f}s early")
                drift = u["placed_start"] - u["start"]
                if drift > q.max_drift_s:
                    flag("warn", lang, where, "drift", f"{drift:.2f}s late against the original")
                if u["placed_end"] > u["slot_end"] + 0.05:
                    flag("warn", lang, where, "overflow", f"runs {u['placed_end'] - u['slot_end']:.2f}s past slot")
                if u["tempo"] > q.max_tempo_warn:
                    flag("info", lang, where, "tempo", f"sped up x{u['tempo']:.2f}")
                if u["placed_end"] > ch.duration + 0.01:
                    flag("error", lang, where, "cut", "line runs past the chapter end and was cut")
                prev_end = u["placed_end"]
            lang_sum["tempo_changed"] += rep["tempo_changed"]
            lang_sum["max_drift_s"] = max(lang_sum["max_drift_s"], rep["max_drift_s"])
        if speech_total:
            measured = chars_total / speech_total
            lang_sum["measured_chars_per_second"] = round(measured, 2)
            lang_sum["configured_chars_per_second"] = ctx.cfg.language(lang).chars_per_second
        for lesson in m.lessons:
            vtt = ctx.cd.subtitle(lang, lesson.id, "vtt")
            if not vtt.exists():
                continue
            cues = parse_vtt(vtt.read_text(encoding="utf-8"))
            limit = MAX_READING_CPS.get(lang, 21.0)
            for a, b in itertools.pairwise(cues):
                if b.start < a.end - 0.001:
                    flag("error", lang, lesson.id, "subtitle-overlap", f"{a.end:.2f}>{b.start:.2f}")
            for cue in cues:
                dur = max(0.01, cue.end - cue.start)
                cps = text_length(cue.text.replace("\n", "" if lang in CJK_LANGS else " "), lang) / dur
                if cps > limit:
                    flag("info", lang, lesson.id, "reading-speed", f"{cue.start:.1f}s: {cps:.0f} chars/s")
        summary[lang] = dict(lang_sum)

    video_h = m.total_duration / 3600 if m.total_duration else 0.0
    by_stage: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for r in m.runs:
        st = by_stage[r.stage]
        st["seconds"] += r.seconds
        st["items"] += r.items
        if r.gpu:
            st["gpu_seconds"] += r.seconds
        for k, v in r.usage.items():
            if not k.startswith("gpu_util") and not k.startswith("gpu_mem"):
                st[k] += v
    gpu_h = sum(st.get("gpu_seconds", 0.0) for st in by_stage.values()) / 3600
    cost = {
        "video_hours": round(video_h, 3),
        "gpu_hours": round(gpu_h, 3),
        "gpu_hours_per_video_hour": round(gpu_h / video_h, 3) if video_h else None,
        "wall_hours": round(sum(st["seconds"] for st in by_stage.values()) / 3600, 3),
        "stages": {k: {kk: round(vv, 2) for kk, vv in v.items()} for k, v in by_stage.items()},
    }
    counts: dict[str, Any] = {sev: sum(1 for i in issues if i["severity"] == sev) for sev in ("error", "warn", "info")}
    report: dict[str, Any] = {
        "course": m.course_id,
        "title": m.title,
        "languages": summary,
        "counts": counts,
        "cost": cost,
        "issues": issues,
    }
    ctx.cd.qa_dir.mkdir(parents=True, exist_ok=True)
    write_json(ctx.cd.qa_dir / "report.json", report)
    (ctx.cd.qa_dir / "report.html").write_text(_html(report), encoding="utf-8")
    ctx.store.mark_done("qa", "report", [ctx.cd.qa_dir / "report.json"], info=counts)
    log.info("qa: %s -> %s", report["counts"], ctx.cd.qa_dir / "report.html")
    return report


def _html(report: dict[str, Any]) -> str:
    e = html.escape
    rows = "".join(
        f"<tr class='{e(i['severity'])}'><td>{e(i['severity'])}</td><td>{e(i['lang'])}</td><td>{e(i['where'])}</td>"
        f"<td>{e(i['kind'])}</td><td>{e(i['detail'])}</td></tr>" for i in report["issues"])
    langs = "".join(f"<tr><td>{e(k)}</td><td><code>{e(str(v))}</code></td></tr>"
                    for k, v in report["languages"].items())
    stages = "".join(f"<tr><td>{e(k)}</td><td><code>{e(str(v))}</code></td></tr>"
                     for k, v in report["cost"]["stages"].items())
    c = report["cost"]
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>QA {e(report['course'])}</title>
<style>body{{font:14px system-ui,sans-serif;margin:16px;max-width:1100px}}table{{border-collapse:collapse;width:100%}}
td,th{{border-bottom:1px solid #ddd;padding:4px 6px;text-align:left;vertical-align:top}}
tr.error td{{background:#fde8e8}}tr.warn td{{background:#fff6dd}}code{{font-size:12px}}</style></head><body>
<h1>QA report: {e(report['title'])}</h1>
<p>Errors {report['counts']['error']} · warnings {report['counts']['warn']} · info {report['counts']['info']}</p>
<h2>Cost and time</h2><p>Video {c['video_hours']} h · GPU {c['gpu_hours']} h ·
GPU-h per video-h {c['gpu_hours_per_video_hour']} · wall {c['wall_hours']} h</p>
<table><tr><th>stage</th><th>totals</th></tr>{stages}</table>
<h2>Languages</h2><table>{langs}</table>
<h2>Issues</h2><table><tr><th>severity</th><th>lang</th><th>where</th><th>kind</th><th>detail</th></tr>{rows}</table>
</body></html>"""
