"""Per-course manifest: what exists, which stage finished which item, and how long it took.

Every stage keys its work by item ("L01-C003", "L01-C003:en", "L01:de" ...). An item is done when
its record says so *and* its outputs still exist, so deleting an output file re-runs just that item.
The manifest is written atomically after every item, which makes all stages resumable.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class Lesson(BaseModel):
    id: str
    source: str  # path relative to the course dir
    duration: float
    width: int = 0
    height: int = 0
    fps: float = 0.0
    has_audio: bool = True


class Chapter(BaseModel):
    id: str
    lesson_id: str
    index: int
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


class TaskRecord(BaseModel):
    status: str = "done"
    finished_at: str = Field(default_factory=now_iso)
    seconds: float = 0.0
    fingerprint: str = ""
    outputs: list[str] = Field(default_factory=list)
    info: dict[str, Any] = Field(default_factory=dict)


class RunRecord(BaseModel):
    stage: str
    started_at: str
    seconds: float
    items: int
    gpu: bool = False
    backend: str = ""
    media_seconds: float = 0.0  # seconds of source video processed, for GPU-h per video-h
    usage: dict[str, float] = Field(default_factory=dict)  # API tokens / characters


class Manifest(BaseModel):
    course_id: str
    title: str = ""
    description: str = ""
    created_at: str = Field(default_factory=now_iso)
    lessons: list[Lesson] = Field(default_factory=list)
    chapters: list[Chapter] = Field(default_factory=list)
    stages: dict[str, dict[str, TaskRecord]] = Field(default_factory=dict)
    runs: list[RunRecord] = Field(default_factory=list)

    def lesson(self, lesson_id: str) -> Lesson:
        for lesson in self.lessons:
            if lesson.id == lesson_id:
                return lesson
        raise KeyError(lesson_id)

    def chapter(self, chapter_id: str) -> Chapter:
        for ch in self.chapters:
            if ch.id == chapter_id:
                return ch
        raise KeyError(chapter_id)

    def chapters_of(self, lesson_id: str) -> list[Chapter]:
        return sorted((c for c in self.chapters if c.lesson_id == lesson_id), key=lambda c: c.index)

    @property
    def total_duration(self) -> float:
        return sum(lesson.duration for lesson in self.lessons)


class ManifestStore:
    """Loads/saves manifest.json in a course work dir; holds an exclusive lock while open."""

    def __init__(self, course_dir: Path, course_id: str | None = None) -> None:
        self.course_dir = course_dir
        self.path = course_dir / "manifest.json"
        course_dir.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self.data = Manifest.model_validate_json(self.path.read_text(encoding="utf-8"))
        else:
            self.data = Manifest(course_id=course_id or course_dir.name)
        self._lock_fd: int | None = None

    # -- locking -------------------------------------------------------------
    def lock(self) -> None:
        fd = os.open(self.course_dir / ".lock", os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            raise RuntimeError(f"another ritim process is working on {self.course_dir}") from None
        self._lock_fd = fd

    def unlock(self) -> None:
        if self._lock_fd is not None:
            fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
            os.close(self._lock_fd)
            self._lock_fd = None

    # -- persistence ---------------------------------------------------------
    def save(self) -> None:
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(self.data.model_dump_json(indent=2), encoding="utf-8")
        os.replace(tmp, self.path)

    # -- task records --------------------------------------------------------
    def record(self, stage: str, item: str) -> TaskRecord | None:
        return self.data.stages.get(stage, {}).get(item)

    def is_done(self, stage: str, item: str, fingerprint: str = "") -> bool:
        rec = self.record(stage, item)
        if rec is None or rec.status != "done":
            return False
        if fingerprint and rec.fingerprint != fingerprint:
            return False
        return all((self.course_dir / o).exists() for o in rec.outputs)

    def mark_done(
        self,
        stage: str,
        item: str,
        outputs: list[Path] | None = None,
        seconds: float = 0.0,
        fingerprint: str = "",
        info: dict[str, Any] | None = None,
    ) -> None:
        rels = [str(p.relative_to(self.course_dir)) if p.is_absolute() else str(p) for p in outputs or []]
        self.data.stages.setdefault(stage, {})[item] = TaskRecord(
            seconds=round(seconds, 3), fingerprint=fingerprint, outputs=rels, info=info or {}
        )
        self.save()

    def invalidate(self, stage: str, prefix: str = "") -> int:
        records = self.data.stages.get(stage, {})
        doomed = [k for k in records if k.startswith(prefix)]
        for k in doomed:
            del records[k]
        if doomed:
            self.save()
        return len(doomed)

    def add_run(self, run: RunRecord) -> None:
        self.data.runs.append(run)
        self.save()


@contextmanager
def open_manifest(course_dir: Path, course_id: str | None = None) -> Iterator[ManifestStore]:
    store = ManifestStore(course_dir, course_id)
    store.lock()
    try:
        yield store
    finally:
        store.save()
        store.unlock()


def fingerprint_files(*paths: Path, extra: str = "") -> str:
    """Cheap content fingerprint (size + mtime + sha of small JSON inputs)."""
    h = hashlib.sha256(extra.encode())
    for p in paths:
        if not p.exists():
            h.update(b"missing:" + str(p).encode())
            continue
        if p.suffix == ".json" and p.stat().st_size < 4_000_000:
            h.update(p.read_bytes())
        else:
            st = p.stat()
            h.update(f"{p.name}:{st.st_size}:{st.st_mtime_ns}".encode())
    return h.hexdigest()[:16]


class Timer:
    def __init__(self) -> None:
        self.start = time.monotonic()

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.start


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)
