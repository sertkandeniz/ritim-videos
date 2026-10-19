"""S3 sync for inputs, resumable work dirs and outputs (boto3; credentials from the instance role/env)."""

from __future__ import annotations

import json
import logging
import mimetypes
import os
from pathlib import Path
from typing import Any

from .config import Config

log = logging.getLogger(__name__)

CONTENT_TYPES = {
    ".m3u8": "application/vnd.apple.mpegurl",
    ".ts": "video/mp2t",
    ".vtt": "text/vtt; charset=utf-8",
    ".srt": "application/x-subrip; charset=utf-8",
    ".json": "application/json",
    ".html": "text/html; charset=utf-8",
}
# Never upload scratch files; job files hold local paths only.
SKIP_DIRS = {"tmp", "jobs"}
STATE_FILE = ".s3state.json"


def make_client(cfg: Config) -> Any:
    import boto3

    return boto3.client("s3", region_name=cfg.s3.region)


def _list(client: Any, bucket: str, prefix: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            out[obj["Key"]] = int(obj["Size"])
    return out


def download_prefix(client: Any, bucket: str, prefix: str, dest: Path) -> int:
    n = 0
    for key, size in _list(client, bucket, prefix).items():
        rel = key[len(prefix):]
        if not rel or rel.endswith("/"):
            continue
        target = dest / rel
        if target.exists() and target.stat().st_size == size:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".s3part")
        client.download_file(bucket, key, str(tmp))
        os.replace(tmp, target)
        n += 1
    return n


def upload_dir(client: Any, bucket: str, src: Path, prefix: str, state_path: Path | None = None) -> int:
    """Upload files changed since the last sync (size + mtime remembered in a local state file)."""
    state: dict[str, list[int]] = {}
    if state_path and state_path.exists():
        state = json.loads(state_path.read_text())
    n = 0
    for path in sorted(src.rglob("*")):
        if not path.is_file() or path.name in (STATE_FILE, ".lock") or path.name.endswith((".part.wav", ".tmp")):
            continue
        rel = path.relative_to(src)
        if rel.parts and rel.parts[0] in SKIP_DIRS:
            continue
        st = path.stat()
        sig = [st.st_size, st.st_mtime_ns]
        if state.get(str(rel)) == sig:
            continue
        ctype = CONTENT_TYPES.get(path.suffix) or mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        client.upload_file(str(path), bucket, prefix + rel.as_posix(), ExtraArgs={"ContentType": ctype})
        state[str(rel)] = sig
        n += 1
    if state_path:
        state_path.write_text(json.dumps(state))
    return n


def pull_course(cfg: Config, course_id: str, course_dir: Path, client: Any | None = None) -> None:
    client = client or make_client(cfg)
    b = cfg.s3.bucket
    n_work = download_prefix(client, b, f"{cfg.s3.work_prefix}{course_id}/", course_dir)
    n_src = download_prefix(client, b, f"{cfg.s3.inputs_prefix}{course_id}/", course_dir / "source")
    log.info("pull %s: %d source files, %d work files", course_id, n_src, n_work)


def push_work(cfg: Config, course_id: str, course_dir: Path, client: Any | None = None) -> None:
    client = client or make_client(cfg)
    n = upload_dir(client, cfg.s3.bucket, course_dir, f"{cfg.s3.work_prefix}{course_id}/",
                   course_dir / STATE_FILE)
    log.info("push work %s: %d files", course_id, n)


def push_outputs(cfg: Config, course_id: str, course_dir: Path, client: Any | None = None) -> None:
    client = client or make_client(cfg)
    prefix = f"{cfg.s3.outputs_prefix}{course_id}/"
    n = 0
    for sub in ("output", "subtitles", "qa"):
        if (course_dir / sub).exists():
            n += upload_dir(client, cfg.s3.bucket, course_dir / sub, prefix + ("" if sub == "output" else sub + "/"),
                            course_dir / f".s3state.{sub}.json")
    log.info("push outputs %s: %d files -> s3://%s/%s", course_id, n, cfg.s3.bucket, prefix)


def fetch_voice(cfg: Config, speaker: str = "hakan", client: Any | None = None) -> Path:
    client = client or make_client(cfg)
    dest = cfg.voice.reference.audio.parent
    download_prefix(client, cfg.s3.bucket, f"{cfg.s3.voice_prefix}{speaker}/", dest)
    return dest
