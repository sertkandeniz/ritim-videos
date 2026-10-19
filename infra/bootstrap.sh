#!/usr/bin/env bash
# One-shot setup of a g6e.xlarge (1x L40S 48 GB, 4 vCPU, 32 GiB) on the AWS Deep Learning Base
# OSS Nvidia Driver GPU AMI (Ubuntu 24.04). Idempotent: re-running skips finished steps.
#
#   infra/bootstrap.sh                 # everything the config needs (system, storage, all envs, models, verify)
#   infra/bootstrap.sh audio models    # just some steps
#   infra/bootstrap.sh verify
#
# Steps: system storage base audio qwen3tts chatterbox cosyvoice latentsync musetalk models verify
# Versions come from config/pipeline.yaml (models.envs / models.weights); weights are pinned by
# infra/models.lock.json, which the first run writes -- commit it.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "$REPO/.env" ] && set -a && . "$REPO/.env" && set +a

export RITIM_ENVS="${RITIM_ENVS:-/opt/ritim/envs}"
export RITIM_REPOS="${RITIM_REPOS:-/opt/ritim/repos}"
export RITIM_MODELS="${RITIM_MODELS:-/opt/ritim/models}"
export RITIM_WORK="${RITIM_WORK:-/opt/dlami/nvme/ritim/work}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/opt/ritim/uv-cache}"   # same filesystem as envs -> hardlinks
export UV_LINK_MODE=hardlink
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-/opt/ritim/python}"
export HF_HOME="$RITIM_MODELS/hf"
export TORCH_HOME="$RITIM_MODELS/torch"
UV_VERSION="0.11.32"  # the version this repo was developed with
TORCH_INDEX_CU128="https://download.pytorch.org/whl/cu128"
TORCH_INDEX_CU118="https://download.pytorch.org/whl/cu118"

SUDO=""; [ "$(id -u)" -ne 0 ] && SUDO="sudo"
log() { printf '\n\033[1;34m== %s\033[0m\n' "$*"; }
done_marker() { echo "$RITIM_ENVS/.done-$1"; }
is_done() { [ -f "$(done_marker "$1")" ] && [ "${FORCE:-0}" != 1 ]; }
mark_done() { date -u +%FT%TZ > "$(done_marker "$1")"; }
cfg() { "$RITIM_ENVS/base/bin/python" "$REPO/infra/cfg.py" "$@"; }   # read config/pipeline.yaml

git_checkout() {  # name url commit
  local dir="$RITIM_REPOS/$1"
  if [ ! -d "$dir/.git" ]; then git clone --filter=blob:none "$2" "$dir"; fi
  git -C "$dir" fetch --quiet origin "$3" || true
  git -C "$dir" checkout --quiet --detach "$3"
  git -C "$dir" submodule update --init --recursive --quiet
  [ "$(git -C "$dir" rev-parse HEAD)" = "$3" ] || { echo "checkout of $1 is not at $3"; exit 1; }
}

new_env() {  # name python [--seed]
  [ -x "$RITIM_ENVS/$1/bin/python" ] || uv venv --python "$2" ${3:-} "$RITIM_ENVS/$1"
}

step_system() {
  log "system packages"
  $SUDO apt-get update -q
  $SUDO DEBIAN_FRONTEND=noninteractive apt-get install -y -q \
    ffmpeg sox libsox-dev libgl1 libegl1 libgles2 libglib2.0-0 git git-lfs build-essential \
    pkg-config curl unzip jq nvme-cli awscli || \
  $SUDO DEBIAN_FRONTEND=noninteractive apt-get install -y -q \
    ffmpeg sox libsox-dev libgl1 libegl1 libgles2 libglib2.0-0 git git-lfs build-essential pkg-config curl unzip jq nvme-cli
  if ! command -v uv >/dev/null || [ "$(uv --version | awk '{print $2}')" != "$UV_VERSION" ]; then
    curl -LsSf "https://astral.sh/uv/$UV_VERSION/install.sh" | env UV_INSTALL_DIR=/usr/local/bin $SUDO sh
  fi
  $SUDO mkdir -p "$RITIM_ENVS" "$RITIM_REPOS" "$RITIM_MODELS" "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR"
  $SUDO chown -R "$(id -u):$(id -g)" /opt/ritim
}

step_storage() {
  # The DLAMI mounts the g6e instance-store NVMe (250 GB on xlarge) at /opt/dlami/nvme.
  # If it is not mounted (other AMI), format and mount the first empty instance-store disk.
  log "instance store"
  if ! mountpoint -q /opt/dlami/nvme; then
    local dev
    dev=$(lsblk -dnpo NAME,MODEL | awk '/Instance Storage/ {print $1; exit}')
    if [ -n "$dev" ] && ! lsblk -no FSTYPE "$dev" | grep -q .; then
      $SUDO mkfs.ext4 -q -L ritim-nvme "$dev"
      $SUDO mkdir -p /opt/dlami/nvme && $SUDO mount -o noatime "$dev" /opt/dlami/nvme
    else
      echo "no empty instance-store disk found; work dir stays on EBS ($RITIM_WORK)"
    fi
  fi
  $SUDO mkdir -p "$RITIM_WORK" && $SUDO chown -R "$(id -u):$(id -g)" "$(dirname "$RITIM_WORK")"
  df -h "$RITIM_WORK" "$RITIM_MODELS" | sed 's/^/  /'
}

step_base() {
  log "base env (orchestrator: ritim CLI, MediaPipe, boto3, anthropic, huggingface_hub)"
  new_env base 3.12
  uv pip install --python "$RITIM_ENVS/base/bin/python" -e "${REPO}[vision,dev]" "huggingface_hub>=0.30" gdown
  $SUDO ln -sf "$RITIM_ENVS/base/bin/ritim" /usr/local/bin/ritim
  mark_done base
}

step_audio() {
  is_done audio && return
  log "env audio: WhisperX + Demucs (torch 2.8.0 / CUDA 12.8)"
  new_env audio 3.12
  local py="$RITIM_ENVS/audio/bin/python"
  uv pip install --python "$py" --index-url "$TORCH_INDEX_CU128" torch==2.8.0 torchaudio==2.8.0 torchvision==0.23.0
  local specs; mapfile -t specs < <(cfg pip audio)
  uv pip install --python "$py" --extra-index-url "$TORCH_INDEX_CU128" --index-strategy unsafe-best-match \
    "torch==2.8.0" "${specs[@]}" "numba>=0.68" soundfile
  mark_done audio
}

step_qwen3tts() {
  is_done qwen3tts && return
  log "env qwen3tts: Qwen3-TTS (torch 2.8.0 / CUDA 12.8)"
  new_env qwen3tts 3.12
  local py="$RITIM_ENVS/qwen3tts/bin/python"
  uv pip install --python "$py" --index-url "$TORCH_INDEX_CU128" torch==2.8.0 torchaudio==2.8.0
  local specs; mapfile -t specs < <(cfg pip qwen3tts)
  uv pip install --python "$py" --extra-index-url "$TORCH_INDEX_CU128" --index-strategy unsafe-best-match \
    "torch==2.8.0" "${specs[@]}" soundfile
  # Optional: FlashAttention 2 (compiles ~1 h on 4 vCPU). Then set voice.qwen3tts.attn_implementation.
  if [ "${INSTALL_FLASH_ATTN:-0}" = 1 ]; then
    MAX_JOBS=4 uv pip install --python "$py" --no-build-isolation flash-attn
  fi
  mark_done qwen3tts
}

step_chatterbox() {
  is_done chatterbox && return
  log "env chatterbox: Chatterbox Multilingual V3 (torch 2.6.0)"
  read -r url commit < <(cfg git chatterbox)
  git_checkout chatterbox "$url" "$commit"
  new_env chatterbox 3.11
  uv pip install --python "$RITIM_ENVS/chatterbox/bin/python" -e "$RITIM_REPOS/chatterbox" soundfile
  mark_done chatterbox
}

step_cosyvoice() {
  is_done cosyvoice && return
  log "env cosyvoice: Fun-CosyVoice 3 (torch 2.3.1 / CUDA 12.1)"
  read -r url commit < <(cfg git cosyvoice)
  git_checkout CosyVoice "$url" "$commit"
  new_env cosyvoice 3.10
  uv pip install --python "$RITIM_ENVS/cosyvoice/bin/python" --index-strategy unsafe-best-match \
    -r "$RITIM_REPOS/CosyVoice/requirements.txt" "pykakasi==2.3.0"
  mark_done cosyvoice
}

step_latentsync() {
  is_done latentsync && return
  log "env latentsync: LatentSync 1.6 (torch 2.5.1 / CUDA 12.1), InsightFace removed"
  read -r url commit < <(cfg git latentsync)
  git_checkout LatentSync "$url" "$commit"
  new_env latentsync 3.10.13
  # insightface (and onnxruntime-gpu, only used by it) are left out: InsightFace weights are non-commercial.
  grep -v -E '^(insightface|onnxruntime-gpu)' "$RITIM_REPOS/LatentSync/requirements.txt" > "$RITIM_ENVS/latentsync.req.txt"
  uv pip install --python "$RITIM_ENVS/latentsync/bin/python" --index-strategy unsafe-best-match \
    -r "$RITIM_ENVS/latentsync.req.txt"
  cp "$REPO/infra/patches/latentsync_face_detector.py" "$RITIM_REPOS/LatentSync/latentsync/utils/face_detector.py"
  mkdir -p "$RITIM_MODELS/latentsync"
  ln -sfn "$RITIM_MODELS/latentsync" "$RITIM_REPOS/LatentSync/checkpoints"
  if grep -rq "insightface" "$RITIM_REPOS/LatentSync/latentsync" "$RITIM_REPOS/LatentSync/scripts"; then
    echo "insightface still referenced in LatentSync inference code"; exit 1
  fi
  mark_done latentsync
}

step_musetalk() {
  is_done musetalk && return
  log "env musetalk: MuseTalk 1.5 (torch 2.0.1 / CUDA 11.8, mmcv 2.0.1)"
  read -r url commit < <(cfg git musetalk)
  git_checkout MuseTalk "$url" "$commit"
  new_env musetalk 3.10 --seed
  local py="$RITIM_ENVS/musetalk/bin/python"
  uv pip install --python "$py" --index-url "$TORCH_INDEX_CU118" torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
  uv pip install --python "$py" -r "$RITIM_REPOS/MuseTalk/requirements.txt"
  uv pip install --python "$py" openmim
  "$RITIM_ENVS/musetalk/bin/mim" install mmengine "mmcv==2.0.1" "mmdet==3.1.0" "mmpose==1.1.0"
  mkdir -p "$RITIM_MODELS/musetalk"
  ln -sfn "$RITIM_MODELS/musetalk" "$RITIM_REPOS/MuseTalk/models"
  mark_done musetalk
}

step_models() {
  log "model weights (pinned by infra/models.lock.json)"
  local envs="base" d
  for d in "$RITIM_ENVS"/*/; do envs="$envs,$(basename "$d")"; done
  HF_HUB_OFFLINE=0 "$RITIM_ENVS/base/bin/python" "$REPO/infra/fetch_models.py" --envs "$envs"
}

step_verify() {
  log "verify"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
  local cuda_check='import torch,sys; ok=torch.cuda.is_available(); print(f"  torch {torch.__version__} cuda {torch.version.cuda} gpu {torch.cuda.get_device_name(0) if ok else None}"); sys.exit(0 if ok else 1)'
  declare -A imports=(
    [audio]="import whisperx, demucs"
    [qwen3tts]="import qwen_tts"
    [chatterbox]="from chatterbox.mtl_tts import ChatterboxMultilingualTTS"
    [cosyvoice]="import sys; sys.path[:0]=['$RITIM_REPOS/CosyVoice','$RITIM_REPOS/CosyVoice/third_party/Matcha-TTS']; from cosyvoice.cli.cosyvoice import AutoModel"
    [latentsync]="import sys; sys.path.insert(0,'$RITIM_REPOS/LatentSync'); from latentsync.utils.face_detector import FaceDetector; import mediapipe"
    [musetalk]="import mmpose, mmcv, mmdet"
  )
  for env in "${!imports[@]}"; do
    [ -x "$RITIM_ENVS/$env/bin/python" ] || continue
    echo "- $env"
    "$RITIM_ENVS/$env/bin/python" -c "$cuda_check"
    "$RITIM_ENVS/$env/bin/python" -c "${imports[$env]}" && echo "  imports ok"
  done
  ffmpeg -hide_banner -loglevel error -f lavfi -i testsrc=s=1280x720:d=1 -c:v h264_nvenc -f null - \
    && echo "- ffmpeg h264_nvenc ok" || echo "- ffmpeg h264_nvenc NOT available (assemble falls back to libx264)"
  ritim doctor || true
}

ALL_STEPS=(system storage base audio qwen3tts chatterbox cosyvoice latentsync musetalk models verify)
steps=("$@"); [ ${#steps[@]} -eq 0 ] && steps=("${ALL_STEPS[@]}")
for s in "${steps[@]}"; do
  if [ "$s" != system ] && [ "$s" != storage ] && [ "$s" != base ] && [ ! -x "$RITIM_ENVS/base/bin/python" ]; then
    step_base
  fi
  "step_$s"
done
log "bootstrap finished: ${steps[*]}"
