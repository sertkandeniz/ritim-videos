# Infrastructure notes: single-GPU instance (g6.2xlarge or g6e.xlarge)

| | g6.2xlarge (in use, eu-north-1) | g6e.xlarge (original plan) |
|---|---|---|
| GPU | 1x NVIDIA L4, 24 GB, Ada (sm_89), NVENC | 1x NVIDIA L40S, 48 GB, Ada (sm_89), NVENC |
| CPU / RAM | 8 vCPU / 32 GiB | 4 vCPU / 32 GiB |
| Instance store | ~450 GB NVMe SSD (ephemeral) | 250 GB NVMe SSD (ephemeral) |
| AMI | Deep Learning AMI with the NVIDIA driver, Ubuntu 24.04 or Amazon Linux 2023 (the PyTorch variant is fine; its PyTorch is not used, every model family gets its own uv env). | same |

24 GB is enough because only one model is loaded at a time; the largest, LatentSync 1.6, needs 18 GB.
On the L4 lip sync is roughly 2x slower than on the L40S (estimate; the pilot measures it).

## Storage: instance store vs EBS

| What | Where | Why |
|---|---|---|
| Envs, pinned repos, model weights (`/opt/ritim/{envs,repos,models,python,uv-cache}`) | **EBS gp3, 250 GB**, kept across instances (snapshot it after the first bootstrap) | ~45 GB envs + ~35 GB weights. Re-downloading on every Spot start costs 20-40 min; a snapshot restores in minutes. |
| Course work dirs (`/opt/dlami/nvme/ritim/work`) | **Instance-store NVMe** | Fast scratch for multi-GB WAV/video intermediates; free; lost on stop/terminate, which is fine because every stage syncs the work dir to S3 (`ritim run --sync`). |
| Inputs, work backup, outputs, voice reference | **S3** (`courses/`, `work/`, `outputs/`, `voice/`) | Source of truth. Never in git. |

The DLAMI mounts the instance store at `/opt/dlami/nvme` on boot; `bootstrap.sh storage` formats and mounts
it if another AMI does not. Size the gp3 root at 250 GB (3000 IOPS / 125 MB/s default is enough; model
loading is a one-off per stage).

### Disk needed per course (estimate)

Per hour of source video (1080p, 48 kHz):

| Item | Size per video-hour |
|---|---|
| Source video (input copy) | 1-2 GB |
| `chapters/` audio (48 kHz stereo PCM) | 0.7 GB |
| `separated/` vocals + background (44.1 kHz stereo PCM) | 1.3 GB |
| `voice/` raw takes + chapter tracks, per language | 0.5 GB |
| `lipsync/` clips, per lip-synced language (only face segments, ~5-10 % of runtime) | 0.1-0.3 GB |
| `output/` HLS: video once (~0.6-1 GB at CQ 23, slides compress well) + 128 kb/s audio per language (0.06 GB) + patched segments | ~1-1.5 GB for 7 audio tracks |
| `tmp/` during assemble (lesson-long WAVs) | 1-2 GB, deleted after |

**Pilot (VDA 8D, 3 h, TR + EN + DE): ~15-20 GB.** Full course with all 6 languages: ~6-8 GB per video-hour,
so the largest course (~20 h) needs ~150 GB peak on NVMe; delete `chapters/`, `separated/` and `voice/*/` takes
after a course is accepted (they are in S3). All 42 courses (~368 h): ~2.5 TB in S3 (`work/` can move to
Glacier Instant Retrieval after acceptance; `outputs/` is ~1.5 GB per video-hour).

## Spot

* g6e Spot is typically 60-70 % below on-demand, but capacity for L40S is thin; use a Spot request with
  `capacity-optimized` over several AZs in the region (eu-central-1 a/b/c), fallback to on-demand.
* Interruption = 2-minute warning. The pipeline is resumable per item: `ritim run --sync` pushes the work dir
  to S3 after every stage, and every stage skips finished chapters, so a new instance resumes with
  `ritim run <course> --sync ...` (it pulls `work/<course>/` first). Worst case lost work = one stage of one
  course (e.g. ~1 h of LatentSync on a face-heavy course).
* Keep the EBS volume (or its snapshot) and attach it to the replacement instance; `bootstrap.sh` then only
  re-runs `storage` and `verify` (the other steps are marked done in `/opt/ritim/envs/.done-*`).
* Long API-bound stages (translate, ElevenLabs voice) do not need the GPU: run them from any small instance
  (or a laptop) with the same `ritim` CLI and `--sync`, and keep the GPU box for transcribe/separate/
  self-hosted TTS/lipsync.

## IAM

Attach an instance role with `infra/iam-policy.json` (replace `RITIM_BUCKET`). It can read `courses/` and
`voice/`, read/write `work/` and `outputs/`, and nothing else; no delete permission (lifecycle rules clean up).
Enable default bucket encryption (SSE-S3 or SSE-KMS; with KMS add `kms:Decrypt`/`kms:GenerateDataKey` on the key)
and Block Public Access. Voice and video are personal data (KVKK/GDPR): keep the bucket in eu-central-1, no
public outputs; publish HLS through CloudFront with signed URLs / OAC from `outputs/`.

## One model per stage

4 vCPU / 32 GiB RAM shapes the design:

* Each stage loads exactly one model once and processes the whole course (all chapters, one language) before
  exiting; the next stage loads the next model. Nothing else holds GPU memory.
* Media is streamed: face detection reads 1 fps frames from an ffmpeg pipe; chapters are 3-10 min WAV cuts;
  LatentSync gets face crops in <= 40 s chunks (it reads a clip fully into RAM: a 40 s 720x830 crop at 25 fps
  is ~1.8 GB).
* `OMP_NUM_THREADS=4` for workers; ffmpeg encodes use NVENC when present (CPU x264 on 4 vCPU runs about
  1x realtime at 1080p veryfast, NVENC 10x+).

## Licences of everything installed

| Component | Code | Weights | Notes |
|---|---|---|---|
| WhisperX 3.8.6 | BSD-2-Clause | faster-whisper large-v3: MIT | VAD model bundled in whisperx (MIT). Turkish alignment model `mpoyraz/wav2vec2-xls-r-300m-cv7-turkish`: check its model card licence before production; if unsuitable set `transcribe.align: false` (segment timestamps only) or use the Scribe backend. |
| Demucs 4.1.0 | MIT | MIT (adefossez/HTDemucs) | |
| Qwen3-TTS (qwen-tts 0.1.1) | Apache-2.0 | Apache-2.0 | |
| Chatterbox (commit 5de7a54) | MIT | MIT | Output carries Resemble's PerTh watermark (inaudible). Depends on pykakasi (GPL-3.0): runtime use on our server, not distributed. |
| Fun-CosyVoice 3 (commit 074ca6d) | Apache-2.0 | Apache-2.0 | Japanese text converted to katakana with pykakasi (GPL-3.0, runtime only). |
| LatentSync 1.6 (commit a229c39) | Apache-2.0 | Apache-2.0 | **InsightFace removed**: upstream inference loads InsightFace detection/landmark models (non-commercial). Replaced by `infra/patches/latentsync_face_detector.py` (MediaPipe Face Mesh, Apache-2.0). VAE stabilityai/sd-vae-ft-mse: MIT. |
| MuseTalk 1.5 (commit 0a89dec) | MIT | MIT | DWPose (Apache-2.0), face-parsing BiSeNet (MIT), S3FD via face-alignment (BSD-3), whisper-tiny (Apache-2.0), sd-vae (MIT). Its test data is non-commercial: not downloaded. |
| MediaPipe 1.1.1 / BlazeFace | Apache-2.0 | Apache-2.0 | |
| Not used | | | XTTS-v2 (CPML), F5-TTS weights (CC-BY-NC), Higgs Audio, Wav2Lip, InsightFace pretrained models. |
