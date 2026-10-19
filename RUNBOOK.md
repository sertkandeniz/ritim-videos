# Runbook

From an empty AWS account to the dubbed pilot course (VDA 8D Problem Çözme Teknikleri, EN + DE), then the
blind voice test that picks the TTS for FR/ES/JA/ZH. Background on storage, Spot, IAM and licences:
[infra/NOTES.md](infra/NOTES.md).

Conventions: `$BUCKET` is the S3 bucket (default `ritim-videos`, eu-central-1), the pilot course id is
`vda-8d`, instance commands run as `ubuntu` in the repo checkout, and `work/` means the work root
`/opt/dlami/nvme/ritim/work` (instance-store NVMe; `paths.work_root` in the config).

---

## 0. One-time AWS preparation

1. **Quota**: Service Quotas -> EC2 -> "Running On-Demand G and VT instances" and "All G and VT Spot Instance
   Requests" must each be >= 4 vCPU in the region.
2. **Bucket**: create `$BUCKET` in eu-central-1 with Block Public Access and default encryption.
3. **Instance role**: create a role for EC2 with [infra/iam-policy.json](infra/iam-policy.json)
   (replace `RITIM_BUCKET`) and an instance profile `ritim-gpu`.
4. **Secrets**: keep `ANTHROPIC_API_KEY`, `ELEVENLABS_API_KEY`, `ELEVENLABS_VOICE_ID` in your password manager;
   they go into `.env` on the instance only (never into git, never into S3).

## 1. Launch the instance

```bash
REGION=eu-central-1
AMI=$(aws ec2 describe-images --region $REGION --owners amazon \
  --filters "Name=name,Values=Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 24.04)*" \
  --query 'sort_by(Images,&CreationDate)[-1].ImageId' --output text)

aws ec2 run-instances --region $REGION --image-id "$AMI" --instance-type g6e.xlarge \
  --key-name <your-key> --security-group-ids <sg-with-ssh-from-your-ip> --subnet-id <subnet> \
  --iam-instance-profile Name=ritim-gpu \
  --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=250,VolumeType=gp3,DeleteOnTermination=false}' \
  --instance-market-options 'MarketType=spot,SpotOptions={SpotInstanceType=persistent,InstanceInterruptionBehavior=stop}' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=ritim-gpu}]'
```

* Drop `--instance-market-options` for on-demand. With `persistent` + `stop`, a Spot interruption stops the
  instance and keeps the EBS root (envs + weights); the NVMe work dir is lost but is in S3 (see 4.4).
* If Spot capacity is unavailable, try another subnet/AZ, then fall back to on-demand.

## 2. Bootstrap (one script, ~45-70 min the first time)

```bash
ssh ubuntu@<ip>
git clone https://github.com/sertkandeniz/ritim-videos.git && cd ritim-videos
cp .env.example .env && nano .env          # API keys, voice id, bucket
tmux new -s boot
infra/bootstrap.sh                          # system, storage, base, all envs, model weights, verify
```

What it does: apt packages (ffmpeg, sox, EGL for MediaPipe), uv, mounts the instance-store NVMe at
`/opt/dlami/nvme`, one virtualenv per model family under `/opt/ritim/envs` (`base`, `audio`, `qwen3tts`,
`chatterbox`, `cosyvoice`, `latentsync`, `musetalk`), pinned git checkouts under `/opt/ritim/repos`, the
LatentSync InsightFace replacement, then every weight from `config/pipeline.yaml` into `/opt/ritim/models`,
and finally a CUDA + import check per env and `ritim doctor`.

Steps can be run on their own (`infra/bootstrap.sh audio latentsync models verify`); finished env steps are
skipped (`FORCE=1` redoes them). Only need the pilot? `infra/bootstrap.sh system storage base audio latentsync
models verify` (~25 min); add the TTS envs before the blind test.

**After the first bootstrap**: `infra/models.lock.json` now holds the exact Hugging Face commit of every
weight. Commit it (`git add infra/models.lock.json && git commit -m "Pin model weights"` and push) so every
later instance installs the same files. Then snapshot the EBS volume (EC2 -> Volumes -> Create snapshot) to
skip the 45-minute bootstrap next time.

Expected `verify` output: each env prints `torch <ver> cuda <ver> gpu NVIDIA L40S` and `imports ok`, and
`ffmpeg h264_nvenc ok`. `ritim doctor` lists anything still missing (keys, voice reference).

## 3. Upload a course and the voice reference

From the machine that has the source files:

```bash
aws s3 cp --recursive ./VDA8D/ s3://$BUCKET/courses/vda-8d/            # the lesson videos, any names
cat > course.yaml <<'EOF'
title: VDA 8D Problem Çözme Teknikleri
description: 8D problem çözme metodu, VDA yaklaşımı; kök neden analizi, geçici ve kalıcı önlemler.
EOF
aws s3 cp course.yaml s3://$BUCKET/courses/vda-8d/course.yaml
```

Lessons are ordered by natural filename sort ("2 ..." before "10 ...") and get stable ids `L01, L02, ...`.

Voice reference for the self-hosted TTS models (ElevenLabs uses the professional clone instead): 10-20 s of
clean Hakan Bey speech, no music, ideally from a full-face segment, plus its exact Turkish transcript:

```bash
aws s3 cp reference.wav s3://$BUCKET/voice/hakan/reference.wav
aws s3 cp reference.txt s3://$BUCKET/voice/hakan/reference.txt
```

On the instance: `ritim fetch-voice`.

## 4. Pilot: VDA 8D, English + German

### 4.1 Run with a review stop after translation

```bash
tmux new -s pilot
ritim run vda-8d --lang en,de --sync --to subtitles     # inventory .. translate .. subtitles
```

Check before spending voice credits:

* `work/vda-8d/inventory/L*.json` -> `large_face_segments`: are intros/outros found, is the PiP corner window
  ignored? Tune `inventory.large_face_min_height` if not (QA lists the segments).
* `work/vda-8d/subtitles/{tr,en,de}/L*.srt`: open next to the video (VLC: Subtitle -> Add file).
* `work/vda-8d/translations/<lang>/<chapter>.json`: lines with a non-empty `issues` list need a look.
  You may edit `text` by hand; later stages use the edited text.

### 4.2 Voice, lip sync, HLS, QA

```bash
ritim run vda-8d --lang en,de --sync --from voice
```

`voice` synthesizes all lines of a language in one batch, speeds up lines by up to 12 % to fit their slot,
asks Claude for a shorter line when that is not enough (up to 2 rounds, then up to 25 % tempo), and
re-runs `subtitles` so captions match the spoken text. `lipsync` runs LatentSync 1.6 only on large-face
segments, as face crops. `assemble` writes `output/<lesson>/hls/`.

### 4.3 Review the result

* `work/vda-8d/qa/report.html`: errors (overlaps, lost protected terms), warnings (drift, overflow,
  terminology), reading speed, measured speech rate per language, GPU-hours and estimated cost.
* Play: `cd /opt/dlami/nvme/ritim/work/vda-8d/output/L01/hls && python3 -m http.server 8080`, open
  `http://localhost:8080/master.m3u8` in Safari or VLC (SSH tunnel `-L 8080:localhost:8080`).
  `master.m3u8` = original picture with TR/EN/DE audio + subtitles; `master_en.m3u8` / `master_de.m3u8` =
  lip-synced picture for that language.
* Outputs in S3: `s3://$BUCKET/outputs/vda-8d/<lesson>/hls/...`, plus `subtitles/` and `qa/`.

Re-running anything is safe: finished items are skipped. To redo one stage: `ritim voice vda-8d --lang de
--force`; to redo one chapter, delete its output file (e.g. `voice/de/L01-C003.wav`) and run the stage again.

### 4.4 After a Spot interruption

Start the instance again (persistent request restarts it when capacity returns), then:

```bash
cd ritim-videos && infra/bootstrap.sh storage verify
ritim run vda-8d --lang en,de --sync        # pulls work/vda-8d from S3 and continues where it stopped
```

## 5. Blind voice test (FR, ES, JA, ZH)

Pick a 5-minute excerpt with typical content and some technical terms (e.g. 10:00-15:00 of lesson 2):

```bash
ritim blindtest --sample /opt/dlami/nvme/ritim/work/vda-8d/source/<lesson2>.mp4 --start 600 --duration 300 --lang fr \
  --backends qwen3tts,chatterbox,cosyvoice,elevenlabs --out ~/blind/fr
```

* `elevenlabs` is optional; including it as a hidden anchor shows how far the open models are from it.
* Output: `~/blind/fr/` (index.html rating sheet, ratings.csv, `full_A.wav`..., `lines/`, `reference.wav`)
  and `~/blind/fr.key.json` (the letter -> backend key; keep it away from the rater).
* `cd ~/blind && zip -r fr.zip fr` and send `fr.zip` to a native French speaker. They open `index.html`,
  rate each voice 1-5 on naturalness, similarity to the Turkish original, pronunciation of terms and pace,
  press *Download ratings* and send back `ratings_fr.csv`.
* Score (several raters allowed):

```bash
ritim blindtest-score --key ~/blind/fr.key.json ratings_fr_*.csv
```

Set the winner in `config/pipeline.yaml` (`languages.fr.tts_backend: chatterbox`), commit, and repeat for
es, ja, zh. Chatterbox output carries Resemble's inaudible PerTh watermark (MIT licence, allowed); note it if
the ranking is close.

## 6. GPU-hours per video hour, time and cost tracking

Every stage run is recorded in `work/<course>/manifest.json` -> `runs` (wall seconds, items, backend,
whether it used the GPU, seconds of video processed, API tokens / TTS characters, GPU utilisation and peak
memory sampled with `nvidia-smi`).

```bash
ritim status vda-8d       # stage progress + GPU-h per video-h
ritim qa vda-8d --lang en,de && less /opt/dlami/nvme/ritim/work/vda-8d/qa/report.json   # cost section per stage
```

`qa/report.json -> cost`: `video_hours`, `gpu_hours`, `gpu_hours_per_video_hour`, per-stage totals and
`estimated_usd` (GPU, Claude, ElevenLabs) using the rates under `costs:` in the config; set
`costs.gpu_hour_usd` to the price you actually pay.

Planning numbers to replace with the pilot's measured values (L40S, per video-hour, before the pilot these
are estimates from upstream benchmarks, not measurements):

| Stage | Expected GPU time per video-hour | Notes |
|---|---|---|
| transcribe (WhisperX large-v3 + align) | 2-4 min | batch 16, fp16 |
| separate (htdemucs) | 2-4 min | |
| voice, self-hosted TTS | 15-40 min per language | ~70 % of runtime is speech |
| voice, ElevenLabs | none (API) | ~45-55k characters per video-hour per language |
| lipsync (LatentSync 1.6, 20 steps, DeepCache) | 10-30 min per language | only face segments, assume 5-10 % of runtime |
| assemble (NVENC) | 3-5 min | +1-2 min per extra language (audio only) |

Track per course in a sheet: course, video-h, GPU-h per stage, Claude tokens, TTS characters, QA errors,
review notes. All 42 courses x 6 languages is ~368 video-h; multiply by the measured GPU-h per video-h.

## 7. Disk per course

See [infra/NOTES.md](infra/NOTES.md#disk-needed-per-course-estimate): about 5-7 GB per video-hour for TR + EN
+ DE in the work dir (pilot: 15-20 GB) and ~1.5 GB per video-hour of HLS output for all 7 audio tracks. The
250 GB NVMe holds the largest course with all languages; `ritim push` and delete `work/<course>` after
acceptance.

## 8. What is verified where

| | Verified in CI (CPU, mocks) | Must be verified on the GPU instance |
|---|---|---|
| Pipeline logic: chapters, units, budgets, fit, resumability, manifest | yes | |
| HLS structure, patched lip-sync segments (timestamps, frame counts), loudness | yes (libx264) | NVENC output plays in Safari / hls.js / your LMS player |
| Claude request shape, validation rounds | yes (mock client) | translation quality (native review in the pilot) |
| ElevenLabs TTS + Scribe HTTP calls | yes (mock transport) | real voice id, output format allowed by the plan |
| WhisperX, Demucs, Qwen3-TTS, Chatterbox, CosyVoice, LatentSync, MuseTalk workers | syntax + job protocol only | install, model load, outputs; `bootstrap.sh verify` and the pilot |
| LatentSync MediaPipe face-detector patch | landmark mapping logic (fake mesh) | visual check of the first lip-synced clips (mouth aligned, no jitter); if the face is mirrored or offset, report it |
| Face tagging thresholds | logic (mock detector) | `large_face_min_height` on real videos |
