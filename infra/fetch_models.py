"""Download and pin model weights listed in config/pipeline.yaml (models.weights).

Pinning: infra/models.lock.json maps every weight to an exact Hugging Face commit sha (or a sha256 for
plain URLs). Without a lock entry the configured revision (usually "main") is resolved to its current
commit and written to the lock; with an entry, exactly that commit is fetched. Commit the lock file.

Libraries load most models by repo name with HF_HUB_OFFLINE=1, which resolves `refs/main` in the cache.
So for cache layouts we point refs/main at the locked commit: name-based loads then get the pinned files.

Run in the base env:  python infra/fetch_models.py --envs base,audio,latentsync [--check]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.request
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ritim.config import load_config

LOCK = Path(__file__).resolve().parent / "models.lock.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch_hf(w: dict[str, Any], models_root: Path, hf_home: Path, locked: dict[str, Any] | None) -> dict[str, Any]:
    from huggingface_hub import HfApi, snapshot_download

    token = os.environ.get("HF_TOKEN") or None
    revision = locked["sha"] if locked else w.get("revision", "main")
    sha = locked["sha"] if locked else HfApi().model_info(w["hf"], revision=revision, token=token).sha
    dest: str = w["dest"]
    kwargs: dict[str, Any] = {"repo_id": w["hf"], "revision": sha, "token": token}
    if w.get("files"):
        kwargs["allow_patterns"] = w["files"]
    if dest == "cache" or dest.startswith("cache:"):
        cache_dir = hf_home / "hub" if dest == "cache" else models_root / dest.split(":", 1)[1]
        path = snapshot_download(cache_dir=str(cache_dir), **kwargs)
        refs = cache_dir / ("models--" + w["hf"].replace("/", "--")) / "refs"
        refs.mkdir(parents=True, exist_ok=True)
        (refs / "main").write_text(sha)
    else:
        path = snapshot_download(local_dir=str(models_root / dest), **kwargs)
    print(f"  {w['name']}: {w['hf']}@{sha[:12]} -> {path}")
    return {"hf": w["hf"], "sha": sha, "files": w.get("files", []), "licence": w.get("licence", "")}


def fetch_file(w: dict[str, Any], models_root: Path, locked: dict[str, Any] | None) -> dict[str, Any]:
    target = models_root / w["dest"]
    expected = (locked or {}).get("sha256") or w.get("sha256")
    if not target.exists() or (expected and sha256(target) != expected):
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".part")
        if "gdrive" in w:
            import gdown

            gdown.download(id=w["gdrive"], output=str(tmp), quiet=True)
        else:
            urllib.request.urlretrieve(w["url"], tmp)
        tmp.replace(target)
    digest = sha256(target)
    if expected and digest != expected:
        raise SystemExit(f"{w['name']}: sha256 {digest} != pinned {expected}")
    print(f"  {w['name']}: {target} sha256 {digest[:12]}")
    source = {"gdrive": w["gdrive"]} if "gdrive" in w else {"url": w["url"]}
    return {**source, "sha256": digest, "licence": w.get("licence", "")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", default="", help="only weights whose env is in this list")
    ap.add_argument("--check", action="store_true", help="only report what is missing from the lock")
    args = ap.parse_args()
    cfg = load_config()
    models_root = cfg.paths.models_root
    hf_home = Path(os.environ.get("HF_HOME", models_root / "hf"))
    lock: dict[str, Any] = json.loads(LOCK.read_text()) if LOCK.exists() else {}
    wanted = set(filter(None, args.envs.split(",")))
    for w in cfg.models["weights"]:
        if wanted and w.get("env") not in wanted:
            continue
        locked = lock.get(w["name"])
        if args.check:
            print(f"{'locked  ' if locked else 'UNLOCKED'} {w['name']}")
            continue
        lock[w["name"]] = fetch_hf(w, models_root, hf_home, locked) if "hf" in w else \
            fetch_file(w, models_root, locked)
        LOCK.write_text(json.dumps(dict(sorted(lock.items())), indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
