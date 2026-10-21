"""Command line: `ritim <stage> COURSE`, `ritim run COURSE --lang en,de`, S3 sync, blind test, doctor."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

from . import s3sync
from .config import DEFAULT_CONFIG, REPO_ROOT, Config, load_config, load_dotenv
from .manifest import open_manifest
from .stages import RUN_ORDER, STAGES, Ctx
from .workdir import CourseDir

log = logging.getLogger("ritim")


def _langs(cfg: Config, value: str | None) -> list[str]:
    langs = [x.strip() for x in value.split(",")] if value else list(cfg.languages)
    for lang in langs:
        cfg.language(lang)  # validates
    return langs


def _course_dir(cfg: Config, course: str) -> CourseDir:
    return CourseDir(cfg.paths.work_root / course)


def cmd_stage(cfg: Config, args: argparse.Namespace) -> int:
    cd = _course_dir(cfg, args.course)
    if args.sync:
        s3sync.pull_course(cfg, args.course, cd.root)
    with open_manifest(cd.root, args.course) as store:
        ctx = Ctx(cfg=cfg, cd=cd, store=store, langs=_langs(cfg, args.lang), force=args.force)
        STAGES[args.command](ctx)
    if args.sync:
        s3sync.push_work(cfg, args.course, cd.root)
    return 0


def cmd_run(cfg: Config, args: argparse.Namespace) -> int:
    cd = _course_dir(cfg, args.course)
    order = list(RUN_ORDER)
    if args.from_stage:
        order = order[order.index(args.from_stage):]
    if args.to_stage:
        order = order[: len(order) - order[::-1].index(args.to_stage)]
    skip = set(args.skip.split(",")) if args.skip else set()
    order = [s for s in order if s not in skip]
    if args.sync:
        s3sync.pull_course(cfg, args.course, cd.root)
    with open_manifest(cd.root, args.course) as store:
        ctx = Ctx(cfg=cfg, cd=cd, store=store, langs=_langs(cfg, args.lang))
        for stage in order:
            log.info("== %s", stage)
            STAGES[stage](ctx)
            if args.sync:  # after every stage, so a Spot interruption loses at most one stage
                s3sync.push_work(cfg, args.course, cd.root)
    if args.sync:
        s3sync.push_outputs(cfg, args.course, cd.root)
    return 0


def cmd_status(cfg: Config, args: argparse.Namespace) -> int:
    cd = _course_dir(cfg, args.course)
    if not (cd.root / "manifest.json").exists():
        print(f"no work dir for {args.course} at {cd.root}")
        return 1
    from .manifest import ManifestStore

    m = ManifestStore(cd.root).data
    print(f"{m.course_id}: {m.title}  {len(m.lessons)} lessons, {len(m.chapters)} chapters, "
          f"{m.total_duration / 3600:.2f} h video")
    for stage, records in m.stages.items():
        print(f"  {stage:<11} {len(records):>4} items done")
    gpu_s = sum(r.seconds for r in m.runs if r.gpu)
    if m.total_duration:
        print(f"  GPU time {gpu_s / 3600:.2f} h = {gpu_s / m.total_duration:.2f} GPU-h per video-h")
    return 0


def cmd_doctor(cfg: Config, args: argparse.Namespace) -> int:
    import os

    from . import media

    ok = True

    def check(label: str, good: bool, hint: str = "") -> None:
        nonlocal ok
        ok &= good
        print(f"[{'ok' if good else '!!'}] {label}" + ("" if good else f"  -> {hint}"))

    check("ffmpeg", shutil.which("ffmpeg") is not None, "apt install ffmpeg")
    check("h264_nvenc", media.has_encoder("h264_nvenc"), "no NVENC: assemble falls back to libx264")
    check("nvidia-smi", shutil.which("nvidia-smi") is not None, "not a GPU host")
    envs = {cfg.transcribe.env, cfg.separate.env}
    for lang in cfg.languages:
        b = cfg.languages[lang].tts_backend
        if b in ("qwen3tts", "chatterbox", "cosyvoice"):
            envs.add(getattr(cfg.voice, b).env)
    if cfg.lipsync.backend in ("latentsync", "musetalk"):
        envs.add(getattr(cfg.lipsync, cfg.lipsync.backend).env)
    for env in sorted(envs):
        check(f"env {env}", cfg.env_python(env).exists(), "run infra/bootstrap.sh")
    check("mediapipe face model", (cfg.paths.models_root / "mediapipe" / "blaze_face_short_range.tflite").exists(),
          "run infra/bootstrap.sh")
    check("voice reference", cfg.voice.reference.audio.exists(), "ritim fetch-voice")
    for var in ("ANTHROPIC_API_KEY", "ELEVENLABS_API_KEY", "ELEVENLABS_VOICE_ID"):
        check(var, bool(os.environ.get(var)), "set it in .env")
    lock = REPO_ROOT / "infra" / "models.lock.json"
    check("models.lock.json", lock.exists(), "first bootstrap writes it; commit it")
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ritim", description=__doc__)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--work-root", type=Path, help="override paths.work_root")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    for name in STAGES:
        sp = sub.add_parser(name, help=f"run the {name} stage for one course")
        sp.add_argument("course")
        sp.add_argument("--lang", help="comma-separated target languages (default: all in config)")
        sp.add_argument("--force", action="store_true", help="redo items even if finished")
        sp.add_argument("--sync", action="store_true", help="pull from / push to S3 around the stage")

    sp = sub.add_parser("run", help="chain stages for a course and language list")
    sp.add_argument("course")
    sp.add_argument("--lang", help="comma-separated target languages, e.g. en,de")
    sp.add_argument("--from", dest="from_stage", choices=list(STAGES))
    sp.add_argument("--to", dest="to_stage", choices=list(STAGES))
    sp.add_argument("--skip", help="comma-separated stages to skip, e.g. lipsync")
    sp.add_argument("--sync", action="store_true", help="pull inputs, push work after each stage and outputs")

    sp = sub.add_parser("pull", help="download course inputs (and existing work) from S3")
    sp.add_argument("course")
    sp = sub.add_parser("push", help="upload work dir (and outputs with --outputs) to S3")
    sp.add_argument("course")
    sp.add_argument("--outputs", action="store_true")
    sub.add_parser("fetch-voice", help="download the voice reference from S3")
    sp = sub.add_parser("status", help="show stage progress and GPU time for a course")
    sp.add_argument("course")
    sub.add_parser("doctor", help="check envs, models, tools and secrets on this machine")

    sp = sub.add_parser("blindtest", help="render a sample with several TTS backends for blind rating")
    sp.add_argument("--sample", type=Path, required=True, help="a course video (or a 5-minute excerpt)")
    sp.add_argument("--lang", required=True)
    sp.add_argument("--backends", default="qwen3tts,chatterbox,cosyvoice")
    sp.add_argument("--start", type=float, help="excerpt start in seconds")
    sp.add_argument("--duration", type=float, default=300.0)
    sp.add_argument("--lines", type=int, default=8)
    sp.add_argument("--out", type=Path, required=True)
    sp = sub.add_parser("blindtest-score", help="average returned rating CSVs per backend")
    sp.add_argument("--key", type=Path, required=True)
    sp.add_argument("ratings", type=Path, nargs="+")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    load_dotenv(REPO_ROOT / ".env")
    overrides = {"paths.work_root": str(args.work_root)} if args.work_root else None
    cfg = load_config(args.config, overrides)
    c = args.command
    if c in STAGES:
        return cmd_stage(cfg, args)
    if c == "run":
        return cmd_run(cfg, args)
    if c == "pull":
        s3sync.pull_course(cfg, args.course, _course_dir(cfg, args.course).root)
        return 0
    if c == "push":
        s3sync.push_work(cfg, args.course, _course_dir(cfg, args.course).root)
        if args.outputs:
            s3sync.push_outputs(cfg, args.course, _course_dir(cfg, args.course).root)
        return 0
    if c == "fetch-voice":
        print(s3sync.fetch_voice(cfg))
        return 0
    if c == "status":
        return cmd_status(cfg, args)
    if c == "doctor":
        return cmd_doctor(cfg, args)
    if c == "blindtest":
        from .blindtest import run_blindtest

        out = run_blindtest(cfg, args.sample, args.lang, args.backends.split(","), args.out,
                            start=args.start, duration=args.duration, lines=args.lines)
        print(f"rating sheet: {out / 'index.html'}  (key: {out.with_name(out.name + '.key.json')})")
        return 0
    if c == "blindtest-score":
        from .blindtest import score

        print(json.dumps(score(args.ratings, args.key), indent=2))
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
