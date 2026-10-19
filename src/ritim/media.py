"""Thin ffmpeg/ffprobe wrappers. Media is always streamed from disk; nothing loads a whole video."""

from __future__ import annotations

import functools
import json
import logging
import re
import shutil
import subprocess
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

log = logging.getLogger(__name__)

FFMPEG = "ffmpeg"
FFPROBE = "ffprobe"


class MediaError(RuntimeError):
    pass


def run(cmd: Sequence[str | Path], capture: bool = True) -> subprocess.CompletedProcess[str]:
    args = [str(c) for c in cmd]
    log.debug("run: %s", " ".join(args))
    proc = subprocess.run(args, capture_output=capture, text=True, check=False)
    if proc.returncode != 0:
        tail = (proc.stderr or "")[-2000:]
        raise MediaError(f"command failed ({proc.returncode}): {' '.join(args[:8])} ...\n{tail}")
    return proc


def ffmpeg(*args: str | Path) -> subprocess.CompletedProcess[str]:
    return run([FFMPEG, "-hide_banner", "-nostdin", "-y", "-loglevel", "error", *args])


@dataclass
class Probe:
    duration: float
    width: int
    height: int
    fps: float
    has_video: bool
    has_audio: bool
    audio_rate: int


def _fps(rate: str) -> float:
    if "/" in rate:
        num, den = rate.split("/")
        return float(num) / float(den) if float(den) else 0.0
    return float(rate or 0)


def probe(path: Path) -> Probe:
    out = run([FFPROBE, "-v", "error", "-show_format", "-show_streams", "-of", "json", path]).stdout
    data = json.loads(out)
    video = next((s for s in data["streams"] if s.get("codec_type") == "video"), None)
    audio = next((s for s in data["streams"] if s.get("codec_type") == "audio"), None)
    duration = float(data.get("format", {}).get("duration") or 0)
    return Probe(
        duration=duration,
        width=int(video["width"]) if video else 0,
        height=int(video["height"]) if video else 0,
        fps=_fps(video.get("avg_frame_rate", "0")) if video else 0.0,
        has_video=video is not None,
        has_audio=audio is not None,
        audio_rate=int(audio.get("sample_rate", 0)) if audio else 0,
    )


_SIL_START = re.compile(r"silence_start: (-?[\d.]+)")
_SIL_END = re.compile(r"silence_end: ([\d.]+)")


def detect_silences(path: Path, noise_db: float, min_s: float) -> list[tuple[float, float]]:
    proc = run(
        [FFMPEG, "-hide_banner", "-nostdin", "-i", path, "-vn",
         "-af", f"silencedetect=noise={noise_db}dB:d={min_s}", "-f", "null", "-"]
    )
    silences: list[tuple[float, float]] = []
    start: float | None = None
    for line in proc.stderr.splitlines():
        if m := _SIL_START.search(line):
            start = max(0.0, float(m.group(1)))
        elif (m := _SIL_END.search(line)) and start is not None:
            silences.append((round(start, 3), round(float(m.group(1)), 3)))
            start = None
    if start is not None:  # silence runs to the end of the file
        silences.append((round(start, 3), round(probe(path).duration, 3)))
    return silences


_PTS = re.compile(r"pts_time:([\d.]+)")


def detect_scenes(path: Path, threshold: float, fps: float) -> list[float]:
    """Scene changes via ffmpeg's scene score on a small, low-fps copy of the stream."""
    vf = f"fps={fps},scale=320:-2,select='gt(scene\\,{threshold})',showinfo"
    proc = run([FFMPEG, "-hide_banner", "-nostdin", "-i", path, "-an", "-vf", vf, "-f", "null", "-"])
    return [round(float(m.group(1)), 3) for line in proc.stderr.splitlines() if (m := _PTS.search(line))]


def iter_frames(path: Path, fps: float, width: int = 640) -> Iterator[tuple[float, np.ndarray]]:
    """Yield (timestamp, RGB frame) at `fps`, scaled to `width`, streamed through a pipe."""
    p = probe(path)
    height = round(p.height * width / p.width / 2) * 2 if p.width else width
    cmd = [FFMPEG, "-hide_banner", "-nostdin", "-loglevel", "error", "-i", str(path), "-an",
           "-vf", f"fps={fps},scale={width}:{height}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    assert proc.stdout is not None
    frame_bytes = width * height * 3
    i = 0
    try:
        while True:
            buf = proc.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            yield i / fps, np.frombuffer(buf, np.uint8).reshape(height, width, 3)
            i += 1
    finally:
        proc.stdout.close()
        proc.wait()


def extract_audio(src: Path, dst: Path, start: float = 0.0, end: float | None = None,
                  sr: int = 48000, channels: int = 2) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    args: list[str | Path] = ["-ss", f"{start:.3f}"]
    if end is not None:
        args += ["-to", f"{end:.3f}"]
    tmp = dst.with_suffix(".part.wav")
    ffmpeg(*args, "-i", src, "-vn", "-ac", str(channels), "-ar", str(sr), "-c:a", "pcm_s16le", tmp)
    tmp.replace(dst)
    return dst


def to_wav(src: Path, dst: Path, sr: int = 48000, channels: int = 1) -> Path:
    return extract_audio(src, dst, sr=sr, channels=channels)


def read_audio(path: Path, sr: int | None = None, mono: bool = True) -> tuple[np.ndarray, int]:
    """Read audio as float32; resample through ffmpeg if `sr` differs from the file rate."""
    info = sf.info(str(path))
    if sr is not None and info.samplerate != sr:
        tmp = path.with_name(path.stem + f".{sr}.tmp.wav")
        to_wav(path, tmp, sr=sr, channels=1 if mono else info.channels)
        data, rate = sf.read(str(tmp), dtype="float32", always_2d=True)
        tmp.unlink(missing_ok=True)
    else:
        data, rate = sf.read(str(path), dtype="float32", always_2d=True)
    if mono:
        return data.mean(axis=1), int(rate)
    return data, int(rate)


def write_audio(path: Path, data: np.ndarray, sr: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".part.wav")
    sf.write(str(tmp), np.clip(data, -1.0, 1.0), sr, subtype="PCM_16")
    tmp.replace(path)
    return path


def audio_duration(path: Path) -> float:
    info = sf.info(str(path))
    return float(info.frames) / float(info.samplerate)


def trim_silence(audio: np.ndarray, sr: int, threshold_db: float = -45.0, keep_s: float = 0.03) -> np.ndarray:
    """Strip leading/trailing silence that TTS engines pad their output with."""
    if audio.size == 0:
        return audio
    win = max(1, int(sr * 0.01))
    n = audio.size // win
    if n == 0:
        return audio
    frames = audio[: n * win].reshape(n, win)
    rms = np.sqrt(np.mean(frames**2, axis=1) + 1e-12)
    loud = np.nonzero(20 * np.log10(rms) > threshold_db)[0]
    if loud.size == 0:
        return audio[:0]
    keep = int(sr * keep_s)
    start = max(0, loud[0] * win - keep)
    end = min(audio.size, (loud[-1] + 1) * win + keep)
    return audio[start:end]


def atempo_chain(tempo: float) -> str:
    """ffmpeg's atempo accepts 0.5..2.0 per instance; chain for anything outside."""
    parts = []
    t = tempo
    while t > 2.0:
        parts.append("atempo=2.0")
        t /= 2.0
    while t < 0.5:
        parts.append("atempo=0.5")
        t /= 0.5
    parts.append(f"atempo={t:.5f}")
    return ",".join(parts)


def change_tempo(src: Path, dst: Path, tempo: float, method: str = "atempo") -> Path:
    if abs(tempo - 1.0) < 1e-3:
        if src != dst:
            shutil.copyfile(src, dst)
        return dst
    af = f"rubberband=tempo={tempo:.5f}" if method == "rubberband" else atempo_chain(tempo)
    tmp = dst.with_name(dst.stem + ".tempo.part.wav")
    ffmpeg("-i", src, "-af", af, "-c:a", "pcm_s16le", tmp)
    tmp.replace(dst)
    return dst


def measure_loudness(path: Path, i: float, tp: float, lra: float) -> dict[str, str]:
    proc = run(
        [FFMPEG, "-hide_banner", "-nostdin", "-i", path, "-vn",
         "-af", f"loudnorm=I={i}:TP={tp}:LRA={lra}:print_format=json", "-f", "null", "-"]
    )
    text = proc.stderr
    start = text.rfind("{")
    end = text.rfind("}")
    if start < 0 or end < 0:
        raise MediaError("loudnorm produced no measurement")
    result: dict[str, str] = json.loads(text[start : end + 1])
    return result


def loudnorm_filter(path: Path, i: float, tp: float, lra: float) -> str:
    """Second-pass (linear) loudnorm filter string from a first-pass measurement."""
    m = measure_loudness(path, i, tp, lra)
    if m.get("input_i") in ("-inf", "inf"):
        return "anull"  # digital silence (e.g. test fixtures): nothing to normalise
    return (
        f"loudnorm=I={i}:TP={tp}:LRA={lra}:measured_I={m['input_i']}:measured_TP={m['input_tp']}"
        f":measured_LRA={m['input_lra']}:measured_thresh={m['input_thresh']}"
        f":offset={m['target_offset']}:linear=true"
    )


@functools.cache
def has_encoder(name: str) -> bool:
    try:
        out = run([FFMPEG, "-hide_banner", "-encoders"]).stdout
    except (MediaError, FileNotFoundError):
        return False
    if f" {name} " not in out:
        return False
    if name.endswith("nvenc"):  # listed in builds without a GPU too; try one frame
        try:
            ffmpeg("-f", "lavfi", "-i", "color=c=black:s=256x256:d=0.1", "-c:v", name, "-f", "null", "-")
        except MediaError:
            return False
    return True


@functools.cache
def has_filter(name: str) -> bool:
    try:
        out = run([FFMPEG, "-hide_banner", "-filters"]).stdout
    except (MediaError, FileNotFoundError):
        return False
    return f" {name} " in out
