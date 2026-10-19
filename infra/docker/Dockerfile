# One image per model family, built from the same bootstrap steps as the bare-metal setup:
#   docker build -f infra/docker/Dockerfile --build-arg STAGE_ENV=audio      -t ritim-audio .
#   docker build -f infra/docker/Dockerfile --build-arg STAGE_ENV=latentsync -t ritim-latentsync .
# STAGE_ENV: audio | qwen3tts | chatterbox | cosyvoice | latentsync | musetalk
# Weights are not baked in: mount the model cache (EBS) at /opt/ritim/models and the work dir at /work.
#   docker run --gpus all -v /opt/ritim/models:/opt/ritim/models -v /opt/dlami/nvme/ritim/work:/work \
#     -e RITIM_WORK=/work ritim-audio ritim transcribe <course>
# On the g6e box the bare-metal envs are simpler (one host, one GPU); images are for reproducibility/CI.
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ARG STAGE_ENV=audio
ENV DEBIAN_FRONTEND=noninteractive \
    RITIM_ENVS=/opt/ritim/envs RITIM_REPOS=/opt/ritim/repos RITIM_MODELS=/opt/ritim/models \
    RITIM_WORK=/work UV_CACHE_DIR=/opt/ritim/uv-cache HF_HUB_OFFLINE=1

WORKDIR /opt/ritim/app
COPY pyproject.toml README.md ./
COPY src ./src
COPY workers ./workers
COPY infra ./infra
COPY config ./config
COPY glossary ./glossary

RUN infra/bootstrap.sh system base ${STAGE_ENV} \
 && rm -rf /opt/ritim/uv-cache /var/lib/apt/lists/*

ENTRYPOINT []
CMD ["ritim", "doctor"]
