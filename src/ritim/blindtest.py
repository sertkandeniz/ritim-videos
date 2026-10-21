"""Blind voice test: same lines, every TTS backend, shuffled anonymous labels, a rating sheet.

Output folder (zip it and send it to the native-speaker rater):
  index.html          rating sheet; ratings are exported as CSV from the page (no server needed)
  ratings.csv         the same sheet as an empty CSV, for raters who prefer a spreadsheet
  reference.wav       the original Turkish excerpt (for "sounds like Hakan Bey")
  full_<X>.wav        the whole sample dubbed by voice X, mixed over the background
  lines/<n>_<X>.wav   short single-line clips, the same lines for every voice
key.json is written next to the folder, not inside it: it maps letters to backends.
"""

from __future__ import annotations

import csv
import functools
import html
import io
import json
import random
import shutil
import string
from pathlib import Path
from typing import Any

from . import media
from .backends.tts import TTSBackend, TTSItem, make_tts
from .config import Config
from .manifest import open_manifest, read_json
from .stages import STAGES, Ctx
from .stages.video import _mix_audio, lesson_track
from .stages.voice import render_track
from .workdir import CourseDir

CRITERIA = [
    ("natural", "Natural, human-sounding speech"),
    ("similarity", "Sounds like the original speaker"),
    ("pronunciation", "Correct pronunciation, incl. technical terms"),
    ("pace", "Comfortable pace and rhythm"),
]


def prepare_sample(cfg: Config, sample: Path, name: str, start: float | None, duration: float | None) -> Path:
    course_dir = cfg.paths.work_root / "_blindtest" / name
    src = course_dir / "source" / "sample.mp4"
    if not src.exists():
        src.parent.mkdir(parents=True, exist_ok=True)
        args: list[str | Path] = []
        if start is not None:
            args += ["-ss", f"{start:.3f}"]
        if duration is not None:
            args += ["-t", f"{duration:.3f}"]
        media.ffmpeg(*args, "-i", sample, "-c", "copy", src)
    return course_dir


def run_blindtest(cfg: Config, sample: Path, lang: str, backends: list[str], out_dir: Path,
                  start: float | None = None, duration: float | None = 300.0, lines: int = 8, seed: int = 7,
                  overrides: dict[str, Any] | None = None) -> Path:
    course_dir = prepare_sample(cfg, sample, f"{sample.stem}_{lang}", start, duration)
    cd = CourseDir(course_dir)
    # The test sample is one chapter; no face scan needed.
    cfg = cfg.model_copy(deep=True)
    cfg.inventory.face_backend = "none"
    cfg.chapters.max_s = max(cfg.chapters.max_s, (duration or 600) + 1)
    with open_manifest(course_dir) as store:
        ctx = Ctx(cfg=cfg, cd=cd, store=store, langs=[lang], overrides=dict(overrides or {}))
        for stage in ("inventory", "chapters", "transcribe", "separate", "translate"):
            STAGES[stage](ctx)
        lesson = store.data.lessons[0]
        chapters = store.data.chapters_of(lesson.id)
        if len(chapters) != 1:
            raise RuntimeError("blind test sample must fit in one chapter; use --duration <= chapters.max_s")
        ch = chapters[0]
        units = [u for u in read_json(cd.translation(lang, ch.id))["units"] if u.get("text")]

        rng = random.Random(seed)
        letters = list(string.ascii_uppercase[: len(backends)])
        rng.shuffle(letters)
        key = dict(zip(letters, backends, strict=True))
        if out_dir.exists():
            shutil.rmtree(out_dir)
        (out_dir / "lines").mkdir(parents=True)
        media.to_wav(cd.chapter_audio(ch.id), out_dir / "reference.wav", sr=48000, channels=2)

        picked = sorted(rng.sample(range(len(units)), min(lines, len(units))))
        background = lesson_track(ctx, "background", lesson.id)
        for letter, backend in key.items():
            tts: TTSBackend = ctx.backend(f"tts:{backend}", functools.partial(make_tts, cfg, backend))
            take_dir = course_dir / "blind" / lang / backend
            items = [TTSItem(id=u["id"], text=u["text"], lang=lang, out=take_dir / f"{u['id']}.wav",
                             prev_text=units[i - 1]["text"] if i else "",
                             next_text=units[i + 1]["text"] if i + 1 < len(units) else "")
                     for i, u in enumerate(units)]
            tts.synthesize(items, cd.jobs_dir / f"blind_{lang}_{backend}.json")
            takes = {it.id: it.out for it in items}
            voice = course_dir / "blind" / lang / f"{backend}_voice.wav"
            render_track(cfg, [dict(u) for u in units], takes, ch.duration, voice, take_dir / "fit")
            _mix_audio(ctx, voice, background, None, out_dir / f"full_{letter}.wav")
            for n, idx in enumerate(picked, 1):
                media.to_wav(takes[units[idx]["id"]], out_dir / "lines" / f"{n:02d}_{letter}.wav", sr=48000)
        sheet_lines = [{"n": n, "source": units[idx]["source"], "text": units[idx]["text"]}
                       for n, idx in enumerate(picked, 1)]
        order = {n: rng.sample(letters, len(letters)) for n in range(1, len(picked) + 1)}
        full_order = rng.sample(letters, len(letters))
        (out_dir / "index.html").write_text(_sheet_html(lang, cfg.language(lang).name, full_order, sheet_lines,
                                                        order), encoding="utf-8")
        (out_dir / "ratings.csv").write_text(_sheet_csv(full_order, sheet_lines, order), encoding="utf-8")
    key_path = out_dir.with_name(out_dir.name + ".key.json")
    key_path.write_text(json.dumps({"lang": lang, "key": key, "seed": seed}, indent=2), encoding="utf-8")
    return out_dir


def _sheet_csv(full_order: list[str], lines: list[dict[str, Any]], order: dict[int, list[str]]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["item", "voice", *[c for c, _ in CRITERIA], "comment"])
    for letter in full_order:
        w.writerow(["full", letter, *[""] * len(CRITERIA), ""])
    for line in lines:
        for letter in order[line["n"]]:
            w.writerow([f"line{line['n']:02d}", letter, *[""] * len(CRITERIA), ""])
    return buf.getvalue()


def _sheet_html(lang: str, lang_name: str, full_order: list[str], lines: list[dict[str, Any]],
                order: dict[int, list[str]]) -> str:
    e = html.escape

    def rating_row(item: str, letter: str, audio: str) -> str:
        cells = "".join(
            f"<td><select data-item='{item}' data-voice='{letter}' data-c='{c}'>"
            + "".join(f"<option>{v}</option>" for v in ["", "1", "2", "3", "4", "5"]) + "</select></td>"
            for c, _ in CRITERIA)
        return (f"<tr><td><b>{letter}</b></td><td><audio controls preload='none' src='{audio}'></audio></td>{cells}"
                f"<td><input data-item='{item}' data-voice='{letter}' data-c='comment'></td></tr>")

    head = "".join(f"<th>{e(label)}</th>" for _, label in CRITERIA)
    full = "".join(rating_row("full", x, f"full_{x}.wav") for x in full_order)
    blocks = []
    for line in lines:
        rows = "".join(rating_row(f"line{line['n']:02d}", x, f"lines/{line['n']:02d}_{x}.wav")
                       for x in order[line["n"]])
        blocks.append(f"<h3>Line {line['n']}</h3><p class='t'>{e(line['text'])}</p>"
                      f"<table><tr><th>voice</th><th>listen</th>{head}<th>comment</th></tr>{rows}</table>")
    return f"""<!doctype html><html lang="{lang}"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Voice test {e(lang_name)}</title>
<style>body{{font:15px system-ui,sans-serif;margin:16px;max-width:1100px}}table{{border-collapse:collapse;width:100%}}
td,th{{border-bottom:1px solid #ddd;padding:4px;text-align:left}}audio{{width:220px}}.t{{color:#444}}
.wrap{{overflow-x:auto}}button{{font-size:16px;padding:8px 16px}}</style></head><body>
<h1>Voice test – {e(lang_name)}</h1>
<p>Each voice reads the same {e(lang_name)} translation of a Ritim Online training excerpt. Voices are labelled
with letters in random order. Listen to the <a href="reference.wav">original Turkish excerpt</a> first, then rate
every voice from 1 (poor) to 5 (excellent). When finished, press <b>Download ratings</b> and send the CSV back.</p>
<h2>Full excerpt</h2><div class="wrap"><table>
<tr><th>voice</th><th>listen</th>{head}<th>comment</th></tr>{full}</table></div>
<h2>Single lines</h2><div class="wrap">{''.join(blocks)}</div>
<p><label>Your name <input id="rater"></label> <button onclick="dl()">Download ratings</button></p>
<script>
function dl(){{const rows={{}};document.querySelectorAll('[data-item]').forEach(el=>{{
const k=el.dataset.item+'|'+el.dataset.voice;rows[k]=rows[k]||{{}};rows[k][el.dataset.c]=el.value;}});
const cols={json.dumps([c for c, _ in CRITERIA] + ["comment"])};const q=s=>'"'+String(s||'').replace(/"/g,'""')+'"';
let out='rater,item,voice,'+cols.join(',')+'\\n';const r=document.getElementById('rater').value;
for(const k in rows){{const[i,v]=k.split('|');out+=[q(r),i,v,...cols.map(c=>q(rows[k][c]))].join(',')+'\\n';}}
const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([out],{{type:'text/csv'}}));
a.download='ratings_{lang}.csv';a.click();}}
</script></body></html>"""


def score(ratings_csv: list[Path], key_path: Path) -> dict[str, dict[str, float]]:
    """Average each criterion per backend over all raters and items."""
    key = json.loads(key_path.read_text(encoding="utf-8"))["key"]
    sums: dict[str, dict[str, list[float]]] = {}
    for path in ratings_csv:
        with path.open(encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                backend = key.get(row.get("voice", ""))
                if not backend:
                    continue
                for c, _ in CRITERIA:
                    if (row.get(c) or "").strip():
                        sums.setdefault(backend, {}).setdefault(c, []).append(float(row[c]))
    result: dict[str, dict[str, float]] = {}
    for backend, crit in sums.items():
        result[backend] = {c: round(sum(v) / len(v), 2) for c, v in crit.items()}
        result[backend]["overall"] = round(sum(result[backend].values()) / len(crit), 2)
    return dict(sorted(result.items(), key=lambda kv: -kv[1]["overall"]))
