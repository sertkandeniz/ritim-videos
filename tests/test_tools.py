"""Blind voice test, worker protocol, S3 sync, worker scripts and the LatentSync face-detector patch."""

from __future__ import annotations

import csv
import importlib.util
import json
import py_compile
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ritim.backends.tts import MockTTS
from ritim.blindtest import run_blindtest, score
from ritim.config import REPO_ROOT, Config
from ritim.runner import WorkerError, run_worker
from ritim.s3sync import download_prefix, upload_dir

WORKERS = REPO_ROOT / "workers"


def test_blindtest_sheet_is_anonymous_and_complete(cfg: Config, sample_video: Path, tmp_path: Path) -> None:
    backends = ["qwen3tts", "chatterbox", "cosyvoice"]
    overrides = {f"tts:{b}": (lambda b=b, i=i: MockTTS(cfg, 1.0 + 0.1 * i, b)) for i, b in enumerate(backends)}
    out = run_blindtest(cfg, sample_video, "fr", backends, tmp_path / "fr_test", duration=15, lines=3,
                        overrides=overrides)
    html = (out / "index.html").read_text()
    for b in backends:
        assert b not in html  # no backend names leak into the sheet
    letters = sorted(p.stem.split("_")[1] for p in out.glob("full_*.wav"))
    assert letters == ["A", "B", "C"]
    assert len(list((out / "lines").glob("*.wav"))) == 3 * 3
    assert (out / "reference.wav").exists()
    key_path = out.with_name(out.name + ".key.json")
    key = json.loads(key_path.read_text())["key"]
    assert sorted(key.values()) == sorted(backends)
    assert not (out / "key.json").exists()

    rows = list(csv.DictReader((out / "ratings.csv").open()))
    assert {r["voice"] for r in rows} == {"A", "B", "C"} and len(rows) == 3 + 3 * 3
    best = next(letter for letter, b in key.items() if b == "chatterbox")
    filled = tmp_path / "ratings_fr.csv"
    with filled.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            v = "5" if r["voice"] == best else "3"
            w.writerow({**r, "natural": v, "similarity": v, "pronunciation": v, "pace": v})
    result = score([filled], key_path)
    assert next(iter(result)) == "chatterbox" and result["chatterbox"]["overall"] == 5.0


def test_run_worker_protocol(cfg: Config, tmp_path: Path) -> None:
    wdir = tmp_path / "workers"
    wdir.mkdir()
    (wdir / "_common.py").write_text((WORKERS / "_common.py").read_text())
    (wdir / "echo_worker.py").write_text(
        "from pathlib import Path\nfrom _common import atomic, load_job, run_items\n"
        "job, path = load_job()\n"
        "def process(model, it):\n"
        "    if it['id'] == 'bad':\n        raise ValueError('boom')\n"
        "    with atomic(it['out']) as tmp:\n        Path(tmp).write_text(model + it['id'])\n"
        "    return {'n': 1}\n"
        "run_items(job, path, lambda: 'loaded:', process)\n")
    done = tmp_path / "done.txt"
    done.write_text("old")
    job = {"options": {}, "items": [{"id": "a", "out": str(tmp_path / "a.txt")},
                                    {"id": "done", "out": str(done)}]}
    result = run_worker(cfg, "x", "echo_worker.py", job, tmp_path / "jobs" / "j.json", workers_dir=wdir,
                        python=Path(sys.executable))
    assert (tmp_path / "a.txt").read_text() == "loaded:a"
    assert done.read_text() == "old" and result["items"]["done"] == {"skipped": True}
    assert result["items"]["a"]["n"] == 1
    job["items"].append({"id": "bad", "out": str(tmp_path / "bad.txt")})
    with pytest.raises(WorkerError):
        run_worker(cfg, "x", "echo_worker.py", job, tmp_path / "jobs" / "j.json", workers_dir=wdir,
                   python=Path(sys.executable))
    assert not (tmp_path / "bad.txt").exists() and not list(tmp_path.glob("bad.part*"))
    with pytest.raises(WorkerError, match="not installed"):
        run_worker(cfg, "missing-env", "echo_worker.py", job, tmp_path / "j2.json")


class FakeS3:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.extra: dict[str, Any] = {}

    def upload_file(self, filename: str, bucket: str, key: str, ExtraArgs: dict[str, Any]) -> None:
        self.objects[key] = Path(filename).read_bytes()
        self.extra[key] = ExtraArgs

    def download_file(self, bucket: str, key: str, filename: str) -> None:
        Path(filename).write_bytes(self.objects[key])

    def get_paginator(self, name: str) -> Any:
        s3 = self

        class P:
            def paginate(self, Bucket: str, Prefix: str) -> list[dict[str, Any]]:
                return [{"Contents": [{"Key": k, "Size": len(v)} for k, v in s3.objects.items()
                                      if k.startswith(Prefix)]}]

        return P()


def test_s3_upload_is_incremental_and_download_restores(tmp_path: Path) -> None:
    src = tmp_path / "work" / "c1"
    (src / "hls").mkdir(parents=True)
    (src / "tmp").mkdir()
    (src / "hls" / "index.m3u8").write_text("#EXTM3U")
    (src / "manifest.json").write_text("{}")
    (src / "tmp" / "scratch.wav").write_text("x")
    s3 = FakeS3()
    state = src / ".s3state.json"
    assert upload_dir(s3, "b", src, "work/c1/", state) == 2
    assert "work/c1/tmp/scratch.wav" not in s3.objects
    assert s3.extra["work/c1/hls/index.m3u8"]["ContentType"] == "application/vnd.apple.mpegurl"
    assert upload_dir(s3, "b", src, "work/c1/", state) == 0  # unchanged -> nothing uploaded
    dest = tmp_path / "restore"
    assert download_prefix(s3, "b", "work/c1/", dest) == 2
    assert (dest / "hls" / "index.m3u8").read_text() == "#EXTM3U"
    assert download_prefix(s3, "b", "work/c1/", dest) == 0


def test_worker_scripts_compile_and_do_not_import_models_at_top_level() -> None:
    for script in WORKERS.glob("*.py"):
        py_compile.compile(str(script), doraise=True)
        top = [ln for ln in script.read_text().splitlines() if ln.startswith(("import ", "from "))]
        heavy = ("torch", "whisperx", "demucs", "qwen_tts", "chatterbox", "cosyvoice", "latentsync", "diffusers")
        assert not [ln for ln in top if any(h in ln for h in heavy)], (script.name, top)


def test_latentsync_patch_maps_landmarks_without_insightface(monkeypatch: pytest.MonkeyPatch) -> None:
    patch = REPO_ROOT / "infra" / "patches" / "latentsync_face_detector.py"
    assert "import insightface" not in patch.read_text() and "from insightface" not in patch.read_text()

    # Fake MediaPipe Face Mesh: a frontal face centred at (320, 260) in a 640x480 frame.
    def landmark(i: int) -> tuple[float, float]:
        cx, cy = 320, 260
        table = {152: (cx, cy + 110), 454: (cx + 90, cy - 10), 234: (cx - 90, cy - 10),
                 6: (cx, cy - 45), 197: (cx, cy - 25), 195: (cx, cy - 5), 5: (cx, cy + 10), 4: (cx, cy + 20)}
        for k, idx in enumerate([70, 63, 105, 66, 107]):
            table[idx] = (cx - 70 + k * 10, cy - 60)
        for k, idx in enumerate([300, 293, 334, 296, 336]):
            table[idx] = (cx + 70 - k * 10, cy - 60)
        for k, idx in enumerate([323, 361, 288, 397, 365, 379, 378, 400, 377]):
            table[idx] = (cx + 85 - k * 9, cy + 10 + k * 11)
        for k, idx in enumerate([93, 132, 58, 172, 136, 150, 149, 176, 148]):
            table[idx] = (cx - 85 + k * 9, cy + 10 + k * 11)
        x, y = table.get(i, (cx, cy))
        return x / 640, y / 480

    class Mesh:
        def __init__(self, **kw: Any) -> None:
            pass

        def process(self, frame: np.ndarray) -> Any:
            lm = [types.SimpleNamespace(x=x, y=y) for x, y in (landmark(i) for i in range(468))]
            return types.SimpleNamespace(multi_face_landmarks=[types.SimpleNamespace(landmark=lm)])

    fake = types.ModuleType("mediapipe")
    fake.solutions = types.SimpleNamespace(face_mesh=types.SimpleNamespace(FaceMesh=Mesh))  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mediapipe", fake)
    spec = importlib.util.spec_from_file_location("lsfd", patch)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    bbox, lmk = mod.FaceDetector("cuda")(np.zeros((480, 640, 3), np.uint8))
    assert lmk.shape == (106, 2)
    x1, y1, x2, y2 = bbox
    assert 0 <= x1 < 320 < x2 <= 640 and 0 <= y1 < 215 and y2 > 370  # covers brows .. chin
    left = lmk[[43, 48, 49, 51, 50]].mean(axis=0)
    right = lmk[101:106].mean(axis=0)
    nose = lmk[[74, 77, 83, 86]].mean(axis=0)
    assert left[0] < nose[0] < right[0] and nose[1] > left[1]  # template order: left brow, right brow, nose


@pytest.mark.vision
def test_mediapipe_face_detector_runs(tmp_path: Path) -> None:
    pytest.importorskip("mediapipe")
    model = Path(__file__).parent / "data" / "blaze_face_short_range.tflite"
    if not model.exists():
        pytest.skip("face model not downloaded (infra/fetch_models.py fetches it on the instance)")
    from ritim.backends.faces import MediaPipeFaces

    det = MediaPipeFaces(model)
    try:
        assert det.largest_face(np.zeros((360, 640, 3), np.uint8)) is None
    finally:
        det.close()
