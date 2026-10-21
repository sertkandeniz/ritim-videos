"""Layout of a course work directory. Every stage reads and writes only through these paths.

work/<course>/
  manifest.json
  source/<lesson>.mp4                       input videos (synced from S3)
  inventory/<lesson>.json                   probe, scenes, silences, face samples + large-face segments
  chapters/<chapter>/audio.wav              48 kHz stereo cut of the source audio
  transcripts/<chapter>.json                words + sentence units (source language)
  separated/<chapter>/vocals.wav|background.wav
  translations/<lang>/<chapter>.json
  subtitles/<lang>/<lesson>.vtt|.srt
  voice/<lang>/<chapter>/<unit>.wav         raw TTS takes
  voice/<lang>/<chapter>.wav|.json          fitted chapter voice track + placement report
  lipsync/<lang>/<lesson>/<segment>.mp4
  output/<lesson>/hls/master.m3u8 ...
  qa/report.json|report.html
  jobs/                                     job files handed to GPU workers
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CourseDir:
    root: Path

    @property
    def course_id(self) -> str:
        return self.root.name

    @property
    def source(self) -> Path:
        return self.root / "source"

    def source_video(self, rel: str) -> Path:
        return self.root / rel

    def inventory(self, lesson: str) -> Path:
        return self.root / "inventory" / f"{lesson}.json"

    def chapter_dir(self, chapter: str) -> Path:
        return self.root / "chapters" / chapter

    def chapter_audio(self, chapter: str) -> Path:
        return self.chapter_dir(chapter) / "audio.wav"

    def transcript(self, chapter: str) -> Path:
        return self.root / "transcripts" / f"{chapter}.json"

    def vocals(self, chapter: str) -> Path:
        return self.root / "separated" / chapter / "vocals.wav"

    def background(self, chapter: str) -> Path:
        return self.root / "separated" / chapter / "background.wav"

    def translation(self, lang: str, chapter: str) -> Path:
        return self.root / "translations" / lang / f"{chapter}.json"

    def subtitle(self, lang: str, lesson: str, ext: str) -> Path:
        return self.root / "subtitles" / lang / f"{lesson}.{ext}"

    def voice_take_dir(self, lang: str, chapter: str) -> Path:
        return self.root / "voice" / lang / chapter

    def voice_track(self, lang: str, chapter: str) -> Path:
        return self.root / "voice" / lang / f"{chapter}.wav"

    def voice_report(self, lang: str, chapter: str) -> Path:
        return self.root / "voice" / lang / f"{chapter}.json"

    def lipsync_dir(self, lang: str, lesson: str) -> Path:
        return self.root / "lipsync" / lang / lesson

    def output_dir(self, lesson: str) -> Path:
        return self.root / "output" / lesson

    def hls_dir(self, lesson: str) -> Path:
        return self.output_dir(lesson) / "hls"

    @property
    def qa_dir(self) -> Path:
        return self.root / "qa"

    @property
    def jobs_dir(self) -> Path:
        return self.root / "jobs"

    @property
    def tmp(self) -> Path:
        return self.root / "tmp"
