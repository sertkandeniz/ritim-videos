# ritim-videos

Translation pipeline for Ritim Online's course videos: Turkish → EN, DE, FR, ES, JA, ZH.
Subtitles, cloned-voice dubbing and lip sync on full-face shots, run on one AWS g6e.xlarge (NVIDIA L40S, 48 GB).

Plan: https://claude.ai/artifact/KEYJjE75yWGeuh2dLHmT56 · Operations: [RUNBOOK.md](RUNBOOK.md) · For AI agents: [AGENTS.md](AGENTS.md) ·
Infrastructure, storage, Spot, licences: [infra/NOTES.md](infra/NOTES.md)

**Never commit media, transcripts or voice references.** This repository holds code only;
source videos, the voice reference and all outputs live in S3.

## How it runs

```bash
infra/bootstrap.sh                       # once per instance: envs, pinned weights, CUDA check
ritim run vda-8d --lang en,de --sync     # one command per course
```

| Stage | What it does | Model / service |
|---|---|---|
| `inventory` | ffprobe, pauses, scene changes, large-face segments | ffmpeg, MediaPipe BlazeFace |
| `chapters` | 3-10 min cuts at pauses/scene changes, never inside a face segment | |
| `transcribe` | word timings → sentence units | WhisperX large-v3 (or ElevenLabs Scribe) |
| `separate` | voice / background split | Demucs htdemucs |
| `translate` | glossary, course context, per-line length budget, JSON output, validation rounds | Claude (Sonnet 5.5) |
| `subtitles` | WebVTT + SRT per lesson and language | |
| `voice` | TTS per line, small tempo fit, else a shorter re-translation | ElevenLabs PVC (EN/DE), blind-test winner for the rest |
| `lipsync` | face crops of large-face segments only | LatentSync 1.6 (or MuseTalk 1.5) |
| `assemble` | voice over background, EBU R128 loudness, HLS multi-audio + captions | ffmpeg / NVENC |
| `qa` | overlaps, drift, glossary terms, reading speed, GPU-hours and cost | |

Each stage is its own command (`ritim transcribe vda-8d`), works on a per-course work dir, skips
finished chapters and records everything in `manifest.json`. GPU models run in separate virtualenvs
through `workers/*.py`, one model load per stage per course.

## Layout

```
config/pipeline.yaml   languages, backends per language, model pins, paths, S3 bucket, cost rates
glossary/              never-translate list and terminology table
src/ritim/             orchestrator package (CLI, stages, backends, S3 sync, blind test)
workers/               model-side scripts, one per GPU model family
infra/                 bootstrap.sh, fetch_models.py (+ models.lock.json), patches, Dockerfile, IAM policy
tests/                 CPU tests with generated media and mocked models/APIs
```

## Development

```bash
uv venv --python 3.12 && uv pip install -e ".[dev,vision]"
.venv/bin/ruff check src workers infra tests && .venv/bin/mypy && .venv/bin/pytest
```

Needs ffmpeg (and `libegl1` for MediaPipe). No GPU, keys or network access are needed for the tests.
