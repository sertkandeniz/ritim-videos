"""WhisperX large-v3 + Turkish wav2vec2 alignment. Env: audio (whisperx==3.8.6, torch 2.8, CUDA 12.8)."""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any

from _common import atomic, load_job, run_items


def main() -> None:
    job, job_path = load_job()
    o = job["options"]

    def load() -> Any:
        import torch
        import whisperx

        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute = o["compute_type"] if device == "cuda" else "int8"
        asr = whisperx.load_model(
            o["model"], device, compute_type=compute, language=o["language"], vad_method=o["vad_method"],
            asr_options={"initial_prompt": o.get("initial_prompt") or None},
            download_root=o["download_root"], local_files_only=True,
        )
        align = whisperx.load_align_model(language_code=o["language"], device=device) if o["align"] else None
        return whisperx, asr, align, device

    def process(model: Any, it: dict[str, Any]) -> dict[str, Any]:
        whisperx, asr, align, device = model
        audio = whisperx.load_audio(it["audio"])
        result = asr.transcribe(audio, batch_size=o["batch_size"], language=o["language"])
        segments = result["segments"]
        if align is not None:
            segments = whisperx.align(segments, align[0], align[1], audio, device,
                                      return_char_alignments=False)["segments"]
        with atomic(it["out"]) as tmp:
            Path(tmp).write_text(json.dumps({"language": o["language"], "segments": segments},
                                            ensure_ascii=False), encoding="utf-8")
        gc.collect()
        return {"segments": len(segments)}

    run_items(job, job_path, load, process)


if __name__ == "__main__":
    main()
