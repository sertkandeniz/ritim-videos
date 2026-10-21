"""Chatterbox Multilingual V3 (MIT; output carries Resemble's inaudible PerTh watermark).

Env: chatterbox (git commit pinned in config; torch 2.6.0). V3 is selected with t3_model="v3".
"""

from __future__ import annotations

from typing import Any

from _common import atomic, load_job, run_items, save_wav


def main() -> None:
    job, job_path = load_job()
    o = job["options"]

    def load() -> Any:
        import torch
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = ChatterboxMultilingualTTS.from_pretrained(device=device, t3_model=o.get("t3_model", "v3"))
        model.prepare_conditionals(o["ref_audio"], exaggeration=o["exaggeration"])
        return model

    def process(model: Any, it: dict[str, Any]) -> dict[str, Any]:
        wav = model.generate(it["text"], language_id=it["lang"], exaggeration=o["exaggeration"],
                             cfg_weight=o["cfg_weight"])
        with atomic(it["out"]) as tmp:
            save_wav(tmp, wav.cpu().numpy(), model.sr)
        return {"sr": model.sr}

    run_items(job, job_path, load, process)


if __name__ == "__main__":
    main()
