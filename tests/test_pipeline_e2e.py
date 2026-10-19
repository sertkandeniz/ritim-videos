"""Whole pipeline on a synthetic 36 s lesson with mocked models: outputs, HLS validity, resumability."""

from __future__ import annotations

import itertools
import json
import shutil
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

from ritim.backends.tts import MockTTS
from ritim.cli import main as cli_main
from ritim.config import DEFAULT_CONFIG, Config, load_config
from ritim.manifest import open_manifest, read_json
from ritim.stages import RUN_ORDER, STAGES, Ctx
from ritim.stages.video import first_pts, parse_playlist
from ritim.subtitles import parse_vtt
from ritim.workdir import CourseDir

from .conftest import mock_overrides


@dataclass
class Course:
    cfg: Config
    cd: CourseDir
    tts: dict[str, MockTTS]


def run_stages(course: Course, stages: list[str], langs: list[str] | None = None) -> None:
    with open_manifest(course.cd.root, course.cd.course_id) as store:
        ctx = Ctx(cfg=course.cfg, cd=course.cd, store=store, langs=langs or ["en", "de"],
                  overrides={f"tts:{lang}": (lambda t=t: t) for lang, t in course.tts.items()})
        for stage in stages:
            STAGES[stage](ctx)


@pytest.fixture(scope="module")
def course(tmp_path_factory: pytest.TempPathFactory, sample_video: Path) -> Iterator[Course]:
    tmp = tmp_path_factory.mktemp("e2e")
    cfg = load_config(DEFAULT_CONFIG, mock_overrides(tmp))
    cd = CourseDir(cfg.paths.work_root / "vda-8d")
    (cd.source).mkdir(parents=True)
    shutil.copy(sample_video, cd.source / "01 Giriş.mp4")
    (cd.source / "course.yaml").write_text("title: VDA 8D Problem Çözme Teknikleri\n", encoding="utf-8")
    # German voice is "verbose": forces the tempo / shorter-line path.
    c = Course(cfg, cd, {"en": MockTTS(cfg, 1.0, "en"), "de": MockTTS(cfg, 1.35, "de")})
    run_stages(c, RUN_ORDER)
    yield c


def test_inventory_and_chapters(course: Course) -> None:
    inv = read_json(course.cd.inventory("L01"))
    assert len(inv["silences"]) == 3
    assert any(9.5 <= t <= 10.5 for t in inv["scenes"]) and any(21.5 <= t <= 22.5 for t in inv["scenes"])
    segs = inv["large_face_segments"]
    assert len(segs) == 1 and segs[0]["start"] == 0 and 9 <= segs[0]["end"] <= 11
    m = json.loads((course.cd.root / "manifest.json").read_text())
    assert m["title"] == "VDA 8D Problem Çözme Teknikleri"
    chapters = m["chapters"]
    assert next(c["id"] for c in chapters) == "L01-C001"
    assert chapters[0]["end"] >= segs[0]["end"] - 0.01  # no cut inside the face segment
    for a, b in itertools.pairwise(chapters):
        assert a["end"] == b["start"]
    assert chapters[-1]["end"] == pytest.approx(36, abs=0.1)


def test_translations_keep_terms_and_budgets(course: Course) -> None:
    doc = read_json(course.cd.translation("en", "L01-C002"))
    assert doc["units"], doc
    for u in doc["units"]:
        assert u["issues"] == [], u
        assert len(u["text"]) <= u["max_chars"] * 1.05
        for term in ("FMEA", "8D", "VDA 6.3", "IATF 16949"):
            if term in u["source"]:
                assert term in u["text"]


def test_voice_fit_and_shortening(course: Course) -> None:
    en = read_json(course.cd.voice_report("en", "L01-C002"))
    assert en["overflow"] == 0
    de_reports = [read_json(course.cd.voice_report("de", ch)) for ch in ("L01-C001", "L01-C002", "L01-C003")]
    assert sum(r["tempo_changed"] for r in de_reports) > 0
    shortened = [u for r in de_reports for u in r["units"] if u["shorten_rounds"]]
    assert shortened, "the verbose German voice should have triggered shorter re-translations"
    assert course.tts["de"].calls >= 2  # first pass + at least one re-synthesis batch
    for r in de_reports:
        prev = 0.0
        for u in r["units"]:
            assert u["placed_start"] >= prev  # never overlapping
            prev = u["placed_end"]
            assert u["tempo"] <= course.cfg.voice.hard_max_tempo + 1e-6


def test_subtitles_follow_shortened_text(course: Course) -> None:
    vtt = course.cd.subtitle("de", "L01", "vtt").read_text(encoding="utf-8")
    cues = parse_vtt(vtt)
    assert cues and all(a.end <= b.start for a, b in itertools.pairwise(cues))
    doc = read_json(course.cd.translation("de", "L01-C002"))
    short = [u for u in doc["units"] if "text_before_shortening" in u]
    for u in short:
        assert u["text"].split("\n")[0][:15] in vtt.replace("\n", " ")
    assert course.cd.subtitle("tr", "L01", "srt").exists()


def test_hls_outputs_are_valid(course: Course) -> None:
    hls = course.cd.hls_dir("L01")
    master = (hls / "master.m3u8").read_text()
    for lang in ("tr", "en", "de"):
        assert f'LANGUAGE="{lang}"' in master
        assert (hls / f"audio_{lang}" / "index.m3u8").exists()
        assert (hls / f"subs_{lang}" / "index.m3u8").exists()
    assert 'TYPE=SUBTITLES' in master and 'AUDIO="aud"' in master and "RESOLUTION=640x360" in master
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type",
                            "-of", "json", str(hls / "master.m3u8")], capture_output=True, text=True, check=True)
    kinds = [s["codec_type"] for s in json.loads(probe.stdout)["streams"]]
    assert kinds.count("video") >= 1 and kinds.count("audio") >= 3
    segs = parse_playlist(hls / "video" / "index.m3u8")
    assert sum(d for d, _ in segs) == pytest.approx(36, abs=0.2)
    assert all(d <= course.cfg.assemble.hls_segment_s + 0.05 for d, _ in segs)
    first_vtt = (hls / "subs_en" / "seg_00000.vtt").read_text()
    assert first_vtt.startswith("WEBVTT\nX-TIMESTAMP-MAP=MPEGTS:")


def test_audio_is_loudness_normalised(course: Course) -> None:
    hls = course.cd.hls_dir("L01")
    out = subprocess.run(["ffmpeg", "-hide_banner", "-i", str(hls / "audio_en" / "index.m3u8"),
                          "-af", "ebur128", "-f", "null", "-"], capture_output=True, text=True).stderr
    integrated = float(out.rsplit("I:", 1)[1].split("LUFS")[0])
    assert integrated == pytest.approx(course.cfg.assemble.loudness_i, abs=1.5)


def test_lipsync_patches_only_face_segments(course: Course) -> None:
    hls = course.cd.hls_dir("L01")
    for lang in ("en", "de"):
        assert (hls / f"master_{lang}.m3u8").exists()
        segs = parse_playlist(hls / f"video_{lang}" / "index.m3u8")
        base = parse_playlist(hls / "video" / "index.m3u8")
        assert [d for d, _ in segs] == [d for d, _ in base]
        patched = [uri for _, uri in segs if not uri.startswith("../video/")]
        assert patched == ["seg_00000.ts", "seg_00001.ts"]  # face is on screen 0-10 s only
        for uri in patched:
            assert first_pts(hls / f"video_{lang}" / uri) == pytest.approx(first_pts(hls / "video" / uri), abs=0.002)
            frames = [subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
                                      "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(p)],
                                     capture_output=True, text=True, check=True).stdout.strip()
                      for p in (hls / f"video_{lang}" / uri, hls / "video" / uri)]
            assert frames[0] == frames[1]
    clips = read_json(course.cd.lipsync_dir("en", "L01") / "clips.json")["clips"]
    assert clips and clips[0]["w"] % 2 == 0 and clips[0]["x"] + clips[0]["w"] <= 640


def test_qa_report(course: Course) -> None:
    report = read_json(course.cd.qa_dir / "report.json")
    assert report["counts"]["error"] == 0, [i for i in report["issues"] if i["severity"] == "error"]
    assert report["languages"]["de"]["measured_chars_per_second"] > 0
    assert report["cost"]["video_hours"] == pytest.approx(0.01, abs=0.001)
    assert "voice:de" in report["cost"]["stages"]
    assert (course.cd.qa_dir / "report.html").read_text().startswith("<!doctype html>")


def test_rerun_is_a_no_op_and_deleting_an_output_redoes_one_item(course: Course) -> None:
    manifest = course.cd.root / "manifest.json"
    runs_before = len(json.loads(manifest.read_text())["runs"])
    run_stages(course, [s for s in RUN_ORDER if s != "qa"])
    after = json.loads(manifest.read_text())["runs"]
    assert len(after) == runs_before, [r["stage"] for r in after[runs_before:]]

    course.cd.voice_track("en", "L01-C002").unlink()
    run_stages(course, ["voice"], langs=["en"])
    new = json.loads(manifest.read_text())["runs"][runs_before:]
    assert [(r["stage"], r["items"]) for r in new] == [("voice:en", 1)]
    assert course.cd.voice_track("en", "L01-C002").exists()


def test_cli_status_and_stage_commands(course: Course, capsys: pytest.CaptureFixture[str],
                                       tmp_path: Path) -> None:
    cfg_file = tmp_path / "cfg.yaml"
    import yaml

    raw = yaml.safe_load(DEFAULT_CONFIG.read_text())
    raw["paths"]["work_root"] = str(course.cfg.paths.work_root)
    cfg_file.write_text(yaml.safe_dump(raw))
    assert cli_main(["--config", str(cfg_file), "status", "vda-8d"]) == 0
    out = capsys.readouterr().out
    assert "1 lessons" in out and "transcribe" in out
