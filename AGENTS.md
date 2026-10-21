# Agent guide

Read this before changing anything. Human docs: [README.md](README.md) (overview),
[RUNBOOK.md](RUNBOOK.md) (operations), [infra/NOTES.md](infra/NOTES.md) (storage, Spot, IAM, licences).

## What this repo is

Pipeline that translates Ritim Online's Turkish course videos (trainer Hakan Borazan, slides + small
picture-in-picture face, occasional full-face shots) into EN, DE, FR, ES, JA, ZH: subtitles, dubbing in his
cloned voice, lip sync only where the face is large, output as HLS. It runs on one AWS GPU instance
(currently g6.2xlarge, NVIDIA L4 24 GB, Amazon Linux 2023, eu-north-1).

## Hard rules

- **The repo is public.** Never commit media, transcripts, translations of course content, voice
  references, model weights, `.env`, or anything from a work dir. `.gitignore` blocks the usual extensions;
  CI fails if media or weights are tracked. Course data lives in S3 and in the work dir only.
- **Secrets only via environment** (`.env`, loaded by the CLI): `ANTHROPIC_API_KEY`, `ELEVENLABS_API_KEY`,
  `ELEVENLABS_VOICE_ID`, AWS credentials (prefer the instance role). Never print or log them.
- **Commercial-safe licences only.** Do not add XTTS-v2, F5-TTS weights, Higgs Audio, Wav2Lip or InsightFace
  pretrained models. LatentSync's InsightFace detector is replaced by `infra/patches/latentsync_face_detector.py`.
  Check the licence of any new model or dependency and add it to the table in `infra/NOTES.md`.
- **Pin versions.** Code: exact pip versions / git commits in `config/pipeline.yaml` (`models.envs`).
  Weights: commit SHAs in `infra/models.lock.json` (written by the first bootstrap; commit it).
- Keep the CPU test suite green: `ruff`, strict `mypy`, `pytest` (commands below).

## Layout and flow

```
config/pipeline.yaml  the only config: languages + TTS backend per language, stage options, model pins, paths, S3, cost rates
glossary/             never_translate.txt + terms.yaml (per-language terminology)
src/ritim/
  cli.py              `ritim <stage> COURSE`, `ritim run`, pull/push, status, doctor, blindtest
  stages/             one module per stage group; STAGES and RUN_ORDER in stages/__init__.py
  backends/           asr, separation, translation (Claude), tts, lipsync, faces: real + mock implementations
  manifest.py         per-course manifest.json: per-item done records, fingerprints, run timings
  workdir.py          every path inside a course work dir
  media.py            ffmpeg/ffprobe helpers (streamed, never whole videos in RAM)
  runner.py           runs workers/*.py inside a model env with a JSON job file
workers/              model-side scripts, one per model family; stdlib + that model's libs only
infra/                bootstrap.sh (one-shot setup), fetch_models.py (weights + lock), cfg.py, patches/, docker/
tests/                CPU-only: generated media, mocked models and APIs
```

Stage order (`ritim run`): inventory → chapters → transcribe → separate → translate → subtitles → voice →
subtitles (again: voice may shorten lines) → lipsync → assemble → qa.

## Key mechanics

- **Resumable per item.** Each stage marks items done in `manifest.json` (`ctx.store.mark_done`); an item is
  redone if its outputs are missing or its input fingerprint changed. Deleting an output file re-runs only
  that item. Keep new stages idempotent the same way.
- **GPU models run in separate uv envs** under `$RITIM_ENVS` (`audio`, `qwen3tts`, `chatterbox`, `cosyvoice`,
  `latentsync`, `musetalk`) because their torch/CUDA pins conflict. The orchestrator (env `base`) never imports
  torch. `runner.run_worker` writes `jobs/<name>.json` = `{"options", "items"}`; the worker loads its model
  once, skips items whose output exists, writes outputs atomically and `jobs/<name>.result.json`.
- **One model per stage, whole course per stage** (4-8 vCPU, 32 GiB RAM, one GPU).
- **Timing fit.** Translation gets a per-line character budget from its time slot; voice speeds a line up
  to `voice.max_tempo`, else asks Claude for a shorter line (`shorten_rounds`), then caps at `hard_max_tempo`.
- **HLS.** One shared video, one audio rendition per language, segmented WebVTT; for lip-synced languages
  only the segments overlapping face clips are re-encoded (`video_<lang>/`), the rest reference `../video/`.
- **Mocks.** Every backend has a mock; tests inject backends through `Ctx.overrides`.

## Commands

```bash
# development (any machine with ffmpeg; Amazon Linux: bootstrap installs a static ffmpeg)
uv venv --python 3.12 && uv pip install -e ".[dev,vision]"
.venv/bin/ruff check src workers infra tests && .venv/bin/mypy && .venv/bin/pytest -q

# on the GPU instance (envs installed by infra/bootstrap.sh, `ritim` is on PATH)
ritim doctor                                   # what is missing: envs, models, keys, voice reference
ritim run <course> --lang en,de --sync --to subtitles   # stop for translation review
ritim run <course> --lang en,de --sync --from voice
ritim status <course>
```

Work dir: `$RITIM_WORK/<course>` (default `/opt/dlami/nvme/ritim/work`, instance-store NVMe, wiped on stop;
`--sync` saves it to S3 after every stage). Envs/repos/models: `/opt/ritim/{envs,repos,models}` on EBS.

## When something breaks on the instance

- Bootstrap step failed: re-run just that step (`infra/bootstrap.sh <step>`); finished env steps are skipped
  (`FORCE=1` redoes them). Steps: system storage base audio qwen3tts chatterbox cosyvoice latentsync musetalk
  models verify.
- A worker failed: read `$RITIM_WORK/<course>/jobs/<name>.result.json` (per-item errors) and the terminal
  output; reproduce with `$RITIM_ENVS/<env>/bin/python workers/<worker>.py <job.json>` (cwd = the model's repo
  for cosyvoice/latentsync/musetalk).
- Model not found offline: workers run with `HF_HUB_OFFLINE=1`; run `infra/bootstrap.sh models`.
- Fix the code in this repo (not in `/opt/ritim/repos/*` checkouts), add or adjust a test, keep CI green.

## Current status

Built and tested on CPU with mocks; not yet run end-to-end on the GPU. First real runs: bootstrap on the
g6.2xlarge, then the pilot course (VDA 8D Problem Çözme Teknikleri, EN + DE). RUNBOOK section 8 lists what
still needs verification on the GPU (model installs, LatentSync face-detector patch, NVENC playback,
GPU-hours per video-hour). FR/ES/JA/ZH voices wait for the blind test (`languages.<lang>.tts_backend: tbd`).
