"""Fun-CosyVoice 3 (0.5B-2512) cross-lingual voice clone. Env: cosyvoice (repo commit pinned, torch 2.3.1).

Runs with cwd = the CosyVoice checkout (its Matcha-TTS submodule must be importable).
Japanese input must be katakana for CosyVoice 3 (see the repo's example.py); we convert with pykakasi,
which is GPL-3.0 and used only as a runtime tool on our own server (not distributed).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from _common import atomic, load_job, run_items, save_wav

PROMPT = "You are a helpful assistant.<|endofprompt|>"


def to_katakana(text: str) -> str:
    import pykakasi

    kks = pykakasi.kakasi()
    return " ".join(part["kana"] for part in kks.convert(text))


def main() -> None:
    job, job_path = load_job()
    o = job["options"]
    repo = Path(o["repo"])
    sys.path[:0] = [str(repo), str(repo / "third_party" / "Matcha-TTS")]

    def load() -> Any:
        from cosyvoice.cli.cosyvoice import AutoModel

        return AutoModel(model_dir=o["model_dir"])

    def process(model: Any, it: dict[str, Any]) -> dict[str, Any]:
        import torch

        text = to_katakana(it["text"]) if it["lang"] == "ja" else it["text"]
        chunks = [out["tts_speech"] for out in
                  model.inference_cross_lingual(PROMPT + text, o["ref_audio"], stream=False)]
        wav = torch.cat(chunks, dim=1)
        with atomic(it["out"]) as tmp:
            save_wav(tmp, wav.cpu().numpy(), model.sample_rate)
        return {"sr": model.sample_rate}

    run_items(job, job_path, load, process)


if __name__ == "__main__":
    main()
