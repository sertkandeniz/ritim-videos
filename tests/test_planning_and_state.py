"""Chapter planning, face segments, manifest resumability and config loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from ritim.config import Config, expand_env, load_config
from ritim.manifest import ManifestStore, fingerprint_files, open_manifest
from ritim.planning import large_face_segments, plan_chapters


def test_chapters_respect_bounds_and_prefer_long_pauses_at_scene_changes() -> None:
    silences = [(170.0, 170.5), (400.0, 402.0), (430.0, 430.4), (800.0, 801.0), (1250.0, 1251.0)]
    scenes = [401.0, 900.0]
    bounds = plan_chapters(1500, silences, scenes, min_s=180, target_s=420, max_s=600)
    assert bounds[0] == (0.0, 401.0)  # long pause + scene change beats the closer short pause at 430
    assert bounds[-1][1] == 1500
    for a, b in bounds:
        assert 180 <= b - a <= 600 * 1.25


def test_chapters_never_cut_inside_a_face_segment_and_fall_back_to_hard_cut() -> None:
    face = [{"start": 380.0, "end": 420.0, "box": [0, 0, 1, 1]}]
    bounds = plan_chapters(1000, [(401.0, 402.0)], [], 180, 420, 600, avoid=face)
    assert all(not (380 < a < 420) for a, _ in bounds[1:])
    assert bounds[0] == (0.0, 600.0)  # no usable pause -> hard cut at max


def test_short_tail_is_merged_and_short_lesson_is_one_chapter() -> None:
    assert plan_chapters(300, [], [], 180, 420, 600) == [(0.0, 300)]
    assert plan_chapters(650, [(420.0, 421.0)], [], 180, 420, 600) == [(0.0, 420.5), (420.5, 650)]
    # the only pause leaves a 140 s tail (< min): merged, since 650 s <= 1.25 x max
    assert plan_chapters(650, [(509.0, 511.0)], [], 180, 420, 600) == [(0.0, 650)]


def test_large_face_segments_merge_and_filter() -> None:
    def s(t: float, h: float) -> dict[str, object]:
        return {"t": t, "h": h, "box": [0.4, 0.2, 0.6, 0.6] if h else None}

    samples = [s(0, 0.5), s(1, 0.5), s(2, 0.0), s(3, 0.5), s(4, 0.5), s(5, 0.0), s(6, 0.0), s(7, 0.05), s(8, 0.5)]
    segs = large_face_segments(samples, min_height=0.22, min_len_s=3, period_s=1.0)
    assert [(x["start"], x["end"]) for x in segs] == [(0, 5)]  # one-sample hole bridged; 8 s alone too short
    assert segs[0]["box"] == [0.4, 0.2, 0.6, 0.6]


def test_manifest_done_requires_outputs_and_matching_fingerprint(tmp_path: Path) -> None:
    out = tmp_path / "x.json"
    out.write_text("{}")
    with open_manifest(tmp_path, "c1") as store:
        store.mark_done("stage", "L01-C001", [out], seconds=1.0, fingerprint="abc")
        assert store.is_done("stage", "L01-C001")
        assert store.is_done("stage", "L01-C001", "abc")
        assert not store.is_done("stage", "L01-C001", "other")
        out.unlink()
        assert not store.is_done("stage", "L01-C001")  # deleted output -> redo
    reloaded = ManifestStore(tmp_path)
    assert reloaded.data.stages["stage"]["L01-C001"].outputs == ["x.json"]
    assert reloaded.invalidate("stage", "L01") == 1


def test_manifest_lock_is_exclusive(tmp_path: Path) -> None:
    with open_manifest(tmp_path, "c1"), pytest.raises(RuntimeError, match="another ritim process"):
        ManifestStore(tmp_path).lock()


def test_fingerprint_changes_with_json_content(tmp_path: Path) -> None:
    p = tmp_path / "a.json"
    p.write_text('{"a": 1}')
    one = fingerprint_files(p)
    p.write_text('{"a": 2}')
    assert fingerprint_files(p) != one


def test_config_env_expansion_and_backend_choice(monkeypatch: pytest.MonkeyPatch, cfg: Config) -> None:
    monkeypatch.setenv("RITIM_TEST_X", "/data")
    assert expand_env("${RITIM_TEST_X:-/fallback}/w") == "/data/w"
    assert expand_env("${RITIM_TEST_UNSET:-/fallback}/w") == "/fallback/w"
    monkeypatch.setenv("RITIM_TEST_EMPTY", "")
    assert expand_env("${RITIM_TEST_EMPTY:-d}") == "d"
    real = load_config()
    assert real.tts_backend_for("en") == "elevenlabs"
    assert real.translate.model == "claude-sonnet-5-5"
    with pytest.raises(ValueError, match="blind test"):
        real.tts_backend_for("fr")
    assert cfg.tts_backend_for("de") == "mock"
    assert {w["name"] for w in real.models["weights"]} >= {"faster-whisper-large-v3", "latentsync-1.6"}
    assert not any("insightface" in str(w).lower() for w in real.models["weights"])
