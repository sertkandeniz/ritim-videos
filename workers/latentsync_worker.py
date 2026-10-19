"""LatentSync 1.6 (512 px, ~18 GB VRAM). Env: latentsync. Runs with cwd = the LatentSync checkout.

Mirrors scripts/inference.py of the pinned commit but builds the pipeline once for all clips.
The checkout's face detector is replaced by infra/patches/latentsync_face_detector.py (MediaPipe),
because upstream loads InsightFace models, whose weights are licensed for non-commercial use only.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from _common import atomic, load_job, run_items


def main() -> None:
    job, job_path = load_job()
    o = job["options"]
    sys.path.insert(0, str(Path(o["repo"])))

    def load() -> Any:
        import torch
        from accelerate.utils import set_seed
        from diffusers import AutoencoderKL, DDIMScheduler
        from latentsync.models.unet import UNet3DConditionModel
        from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
        from latentsync.utils import face_detector
        from latentsync.whisper.audio2feature import Audio2Feature
        from omegaconf import OmegaConf

        if "insightface" in Path(face_detector.__file__).read_text():
            raise RuntimeError("LatentSync face detector is not patched; re-run infra/bootstrap.sh latentsync")
        config = OmegaConf.load(o["unet_config"])
        dtype = torch.float16 if torch.cuda.get_device_capability()[0] > 7 else torch.float32
        scheduler = DDIMScheduler.from_pretrained("configs")
        whisper = Path(o["inference_ckpt"]).parent / "whisper" / (
            "small.pt" if config.model.cross_attention_dim == 768 else "tiny.pt")
        audio_encoder = Audio2Feature(model_path=str(whisper), device="cuda", num_frames=config.data.num_frames,
                                      audio_feat_length=config.data.audio_feat_length)
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=dtype)
        vae.config.scaling_factor = 0.18215
        vae.config.shift_factor = 0
        unet, _ = UNet3DConditionModel.from_pretrained(OmegaConf.to_container(config.model), o["inference_ckpt"],
                                                       device="cpu")
        pipeline = LipsyncPipeline(vae=vae, audio_encoder=audio_encoder, unet=unet.to(dtype=dtype),
                                   scheduler=scheduler).to("cuda")
        if o.get("deepcache"):
            from DeepCache import DeepCacheSDHelper

            helper = DeepCacheSDHelper(pipe=pipeline)
            helper.set_params(cache_interval=3, cache_branch_id=0)
            helper.enable()
        set_seed(o.get("seed", 1247))
        return pipeline, config, dtype

    def process(model: Any, it: dict[str, Any]) -> dict[str, Any]:
        pipeline, config, dtype = model
        with atomic(it["out"]) as tmp:
            pipeline(video_path=it["video"], audio_path=it["audio"], video_out_path=str(tmp),
                     num_frames=config.data.num_frames, num_inference_steps=o["inference_steps"],
                     guidance_scale=o["guidance_scale"], weight_dtype=dtype, width=config.data.resolution,
                     height=config.data.resolution, mask_image_path=config.data.mask_image_path,
                     temp_dir="temp")
        return {}

    run_items(job, job_path, load, process)


if __name__ == "__main__":
    main()
