"""Decode generated WAM latents after the large WAM process has exited."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from diffusers.video_processor import VideoProcessor

from robotwin_critic.two_stage_rft.generate_wam_candidates import save_image
from robotwin_critic.vlac_finetune.common import read_jsonl


def decode_latents(vae, processor, latents: torch.Tensor):
    latents = latents.to(device=next(vae.parameters()).device, dtype=vae.dtype)
    mean = torch.tensor(
        vae.config.latents_mean, device=latents.device, dtype=latents.dtype
    ).view(1, vae.config.z_dim, 1, 1, 1)
    inverse_std = (
        1.0
        / torch.tensor(
            vae.config.latents_std,
            device=latents.device,
            dtype=latents.dtype,
        )
    ).view(1, vae.config.z_dim, 1, 1, 1)
    latents = latents / inverse_std + mean
    video = vae.decode(latents, return_dict=False)[0]
    return processor.postprocess_video(video, output_type="np")[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    from wan_va.modules.utils import load_vae

    device = torch.device(args.device)
    vae = load_vae(
        str(args.model / "vae"),
        torch_dtype=torch.bfloat16,
        torch_device=device,
    )
    vae.eval()
    processor = VideoProcessor(vae_scale_factor=1)
    rows = read_jsonl(args.input)
    decoded = skipped = 0
    with torch.no_grad():
        for row in rows:
            output = Path(row["generated_image"])
            if output.is_file() and not args.overwrite:
                skipped += 1
                continue
            latents = torch.load(
                row["latent_path"], map_location="cpu", weights_only=False
            )
            frames = decode_latents(vae, processor, latents)
            save_image(output, frames[-1])
            decoded += 1
    result = {
        "candidates": len(rows),
        "decoded": decoded,
        "skipped_existing": skipped,
    }
    print(json.dumps(result, indent=2))
    print("ROBOTWIN_WAM_CANDIDATE_DECODE_OK")


if __name__ == "__main__":
    main()
