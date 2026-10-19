"""Qwen3-TTS 1.7B Base voice clone. Env: qwen3tts (qwen-tts==0.1.1, torch 2.8, CUDA 12.8)."""

from __future__ import annotations

from typing import Any

from _common import atomic, load_job, run_items, save_wav

# qwen-tts takes language names, not codes
LANGUAGE = {"en": "English", "de": "German", "fr": "French", "es": "Spanish", "ja": "Japanese",
            "zh": "Chinese", "it": "Italian", "ko": "Korean", "ru": "Russian", "pt": "Portuguese"}


def main() -> None:
    job, job_path = load_job()
    o = job["options"]

    def load() -> Any:
        import torch
        from qwen_tts import Qwen3TTSModel

        model = Qwen3TTSModel.from_pretrained(o["model"], device_map="cuda:0", dtype=torch.bfloat16,
                                              attn_implementation=o.get("attn_implementation", "sdpa"))
        # Build the speaker prompt once (reference clip + its transcript) and reuse it for every line.
        prompt = model.create_voice_clone_prompt(ref_audio=o["ref_audio"], ref_text=o["ref_text"] or None,
                                                 x_vector_only_mode=not o["ref_text"])
        return model, prompt

    def process(model: Any, it: dict[str, Any]) -> dict[str, Any]:
        tts, prompt = model
        wavs, sr = tts.generate_voice_clone(text=it["text"], language=LANGUAGE[it["lang"]],
                                            voice_clone_prompt=prompt)
        with atomic(it["out"]) as tmp:
            save_wav(tmp, wavs[0], sr)
        return {"sr": sr}

    run_items(job, job_path, load, process)


if __name__ == "__main__":
    main()
