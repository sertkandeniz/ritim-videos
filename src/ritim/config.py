"""Pipeline configuration: one YAML file, secrets only from the environment."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "config" / "pipeline.yaml"

TTS_BACKENDS = ("elevenlabs", "qwen3tts", "chatterbox", "cosyvoice", "mock")

_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def expand_env(value: str) -> str:
    """Expand ${VAR} and ${VAR:-default} (shell semantics: empty counts as unset) and ~."""

    def repl(m: re.Match[str]) -> str:
        env = os.environ.get(m.group(1))
        if env:
            return env
        return m.group(2) if m.group(2) is not None else ""

    return os.path.expanduser(_ENV_RE.sub(repl, value))


def _expand_tree(node: Any) -> Any:
    if isinstance(node, str):
        return expand_env(node)
    if isinstance(node, dict):
        return {k: _expand_tree(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_expand_tree(v) for v in node]
    return node


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PathsCfg(_Model):
    work_root: Path
    models_root: Path
    envs_root: Path
    repos_root: Path
    glossary_dir: Path = Path("glossary")


class S3Cfg(_Model):
    bucket: str
    region: str = "eu-central-1"
    inputs_prefix: str = "courses/"
    work_prefix: str = "work/"
    outputs_prefix: str = "outputs/"
    voice_prefix: str = "voice/"


class LanguageCfg(_Model):
    name: str
    tts_backend: str = "tbd"
    lipsync: bool = False
    chars_per_second: float = 14.0
    subtitle_line_chars: int = 42


class InventoryCfg(_Model):
    scene_threshold: float = 0.3
    scene_fps: float = 4
    silence_db: float = -35
    silence_min_s: float = 0.35
    face_backend: Literal["mediapipe", "none", "mock"] = "mediapipe"
    face_sample_fps: float = 1.0
    large_face_min_height: float = 0.22
    min_face_segment_s: float = 3.0


class ChaptersCfg(_Model):
    min_s: float = 180
    target_s: float = 420
    max_s: float = 600


class TranscribeCfg(_Model):
    backend: Literal["whisperx", "scribe", "mock"] = "whisperx"
    env: str = "audio"
    model: str = "large-v3"
    compute_type: str = "float16"
    batch_size: int = 16
    vad_method: str = "pyannote"
    align: bool = True
    initial_prompt: str = ""
    scribe_model: str = "scribe_v2"
    max_unit_s: float = 12.0
    unit_pause_s: float = 0.8


class SeparateCfg(_Model):
    backend: Literal["demucs", "passthrough"] = "demucs"
    env: str = "audio"
    model: str = "htdemucs"
    segment: int = 7  # demucs takes whole seconds; htdemucs max is 7.8
    shifts: int = 1


class TranslateCfg(_Model):
    backend: Literal["anthropic", "mock"] = "anthropic"
    model: str = "claude-sonnet-5-5"
    effort: str = "medium"
    max_tokens: int = 16000
    http_retries: int = 5
    validation_rounds: int = 2
    fallbacks: Literal["default", "none"] = "default"
    max_extend_s: float = 1.2
    budget_factor: float = 0.95
    context_units: int = 4


class ReferenceCfg(_Model):
    audio: Path
    text_file: Path
    language: str = "tr"

    def text(self) -> str:
        return self.text_file.read_text(encoding="utf-8").strip() if self.text_file.exists() else ""


class ElevenLabsCfg(_Model):
    voice_id: str = ""
    model_id: str = "eleven_multilingual_v2"
    output_format: str = "mp3_44100_192"
    send_language_code: bool = False
    stability: float = 0.5
    similarity_boost: float = 0.8
    style: float = 0.0
    use_speaker_boost: bool = True
    concurrency: int = 3


class Qwen3TTSCfg(_Model):
    env: str = "qwen3tts"
    model: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    attn_implementation: str = "sdpa"


class ChatterboxCfg(_Model):
    env: str = "chatterbox"
    model: str = "ResembleAI/chatterbox"
    t3_model: str = "v3"
    exaggeration: float = 0.5
    cfg_weight: float = 0.3


class CosyVoiceCfg(_Model):
    env: str = "cosyvoice"
    model: str = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"
    repo: str = "CosyVoice"


class VoiceCfg(_Model):
    sample_rate: int = 48000
    max_tempo: float = 1.12
    hard_max_tempo: float = 1.25
    shorten_rounds: int = 2
    min_gap_s: float = 0.06
    tempo_filter: Literal["atempo", "rubberband"] = "atempo"
    reference: ReferenceCfg
    elevenlabs: ElevenLabsCfg = Field(default_factory=ElevenLabsCfg)
    qwen3tts: Qwen3TTSCfg = Field(default_factory=Qwen3TTSCfg)
    chatterbox: ChatterboxCfg = Field(default_factory=ChatterboxCfg)
    cosyvoice: CosyVoiceCfg = Field(default_factory=CosyVoiceCfg)


class LatentSyncCfg(_Model):
    env: str = "latentsync"
    repo: str = "LatentSync"
    inference_steps: int = 20
    guidance_scale: float = 1.5
    deepcache: bool = True


class MuseTalkCfg(_Model):
    env: str = "musetalk"
    repo: str = "MuseTalk"
    bbox_shift: int = 0


class LipsyncCfg(_Model):
    backend: Literal["latentsync", "musetalk", "none", "mock"] = "latentsync"
    pad_s: float = 0.25
    max_chunk_s: float = 40
    crop_scale: float = 2.2
    latentsync: LatentSyncCfg = Field(default_factory=LatentSyncCfg)
    musetalk: MuseTalkCfg = Field(default_factory=MuseTalkCfg)


class AssembleCfg(_Model):
    background_gain_db: float = 0.0
    loudness_i: float = -16.0
    loudness_tp: float = -1.5
    loudness_lra: float = 11.0
    hls_segment_s: int = 6
    video_encoder: Literal["auto", "libx264", "h264_nvenc"] = "auto"
    video_quality: int = 23
    audio_bitrate: str = "128k"
    include_source_audio: bool = True


class QACfg(_Model):
    max_overlap_s: float = 0.05
    max_drift_s: float = 0.6
    max_tempo_warn: float = 1.15


class CostsCfg(_Model):
    gpu_hour_usd: float = 0.0
    claude_input_per_mtok_usd: float = 2.0
    claude_output_per_mtok_usd: float = 10.0
    claude_cache_read_per_mtok_usd: float = 0.2
    elevenlabs_per_1k_chars_usd: float = 0.0


class Config(_Model):
    paths: PathsCfg
    s3: S3Cfg
    source_language: str = "tr"
    languages: dict[str, LanguageCfg]
    inventory: InventoryCfg = Field(default_factory=InventoryCfg)
    chapters: ChaptersCfg = Field(default_factory=ChaptersCfg)
    transcribe: TranscribeCfg = Field(default_factory=TranscribeCfg)
    separate: SeparateCfg = Field(default_factory=SeparateCfg)
    translate: TranslateCfg = Field(default_factory=TranslateCfg)
    voice: VoiceCfg
    lipsync: LipsyncCfg = Field(default_factory=LipsyncCfg)
    assemble: AssembleCfg = Field(default_factory=AssembleCfg)
    qa: QACfg = Field(default_factory=QACfg)
    costs: CostsCfg = Field(default_factory=CostsCfg)
    models: dict[str, Any] = Field(default_factory=dict)

    def glossary_path(self) -> Path:
        p = self.paths.glossary_dir
        return p if p.is_absolute() else REPO_ROOT / p

    def env_python(self, env: str) -> Path:
        return self.paths.envs_root / env / "bin" / "python"

    def repo_dir(self, name: str) -> Path:
        return self.paths.repos_root / name

    def language(self, code: str) -> LanguageCfg:
        try:
            return self.languages[code]
        except KeyError:
            raise KeyError(f"language {code!r} is not in config.languages") from None

    def tts_backend_for(self, code: str) -> str:
        backend = self.language(code).tts_backend
        if backend not in TTS_BACKENDS:
            raise ValueError(
                f"no TTS backend chosen for {code!r} ({backend!r}); run the blind test and set "
                f"languages.{code}.tts_backend to one of {', '.join(TTS_BACKENDS)}"
            )
        return backend


def load_config(path: Path | None = None, overrides: dict[str, Any] | None = None) -> Config:
    """Load the YAML, expand ${ENV} references and apply dotted-key overrides (used by tests)."""
    raw = yaml.safe_load((path or DEFAULT_CONFIG).read_text(encoding="utf-8"))
    raw = _expand_tree(raw)
    for dotted, value in (overrides or {}).items():
        node = raw
        *parents, leaf = dotted.split(".")
        for key in parents:
            node = node.setdefault(key, {})
        node[leaf] = value
    return Config.model_validate(raw)


def require_env(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        raise RuntimeError(f"environment variable {name} is not set (see .env.example)")
    return value


def load_dotenv(path: Path) -> None:
    """Minimal .env loader: KEY=VALUE lines, existing environment wins."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip().strip("'\"")
        if value:
            os.environ.setdefault(key.strip(), value)
