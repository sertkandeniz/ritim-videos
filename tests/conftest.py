"""Shared fixtures: a config pointing at tmp dirs with mock backends, and a synthetic course video."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

from ritim.config import DEFAULT_CONFIG, Config, load_config

# 36 s lesson: red (= "large face" for MockFaces) 0-10 s, blue 10-22 s, green 22-36 s;
# a 440 Hz tone with pauses at 8.0-9.2, 17.0-18.4 and 26.0-27.5 s.
DURATION = 36
PAUSES = [(8.0, 9.2), (17.0, 18.4), (26.0, 27.5)]


def make_video(path: Path, duration: int = DURATION, size: str = "640x360") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    gate = "+".join(f"between(t,{a},{b})" for a, b in PAUSES)
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", f"color=c=red:s={size}:r=25:d=10",
        "-f", "lavfi", "-i", f"color=c=blue:s={size}:r=25:d=12",
        "-f", "lavfi", "-i", f"color=c=green:s={size}:r=25:d={duration - 22}",
        "-f", "lavfi", "-i", f"aevalsrc='0.3*sin(2*PI*440*t)*not({gate})':s=48000:d={duration}",
        "-filter_complex", "[0:v][1:v][2:v]concat=n=3:v=1:a=0[v]",
        "-map", "[v]", "-map", "3:a", "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-shortest", str(path),
    ]
    subprocess.run(cmd, check=True)
    return path


def make_tone(path: Path, seconds: float, freq: int = 300, sr: int = 48000) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                    f"sine=frequency={freq}:sample_rate={sr}:duration={seconds}", str(path)], check=True)
    return path


def mock_overrides(tmp: Path) -> dict[str, Any]:
    return {
        "paths.work_root": str(tmp / "work"),
        "paths.models_root": str(tmp / "models"),
        "paths.envs_root": str(tmp / "envs"),
        "paths.repos_root": str(tmp / "repos"),
        "voice.reference.audio": str(tmp / "models" / "voice" / "hakan" / "reference.wav"),
        "voice.reference.text_file": str(tmp / "models" / "voice" / "hakan" / "reference.txt"),
        "s3.bucket": "test-bucket",
        "inventory.face_backend": "mock",
        "inventory.min_face_segment_s": 2.0,
        "chapters.min_s": 6,
        "chapters.target_s": 12,
        "chapters.max_s": 16,
        "transcribe.backend": "mock",
        "separate.backend": "passthrough",
        "translate.backend": "mock",
        "lipsync.backend": "mock",
        "assemble.video_encoder": "libx264",
        "languages.en.tts_backend": "mock",
        "languages.de.tts_backend": "mock",
    }


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    return load_config(DEFAULT_CONFIG, mock_overrides(tmp_path))


@pytest.fixture(scope="session")
def sample_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return make_video(tmp_path_factory.mktemp("media") / "lesson.mp4")
