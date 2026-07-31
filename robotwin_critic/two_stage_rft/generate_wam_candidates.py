"""Generate chunk-level WAM video/action candidates from Stage-2 RGB contexts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image

from robotwin_critic.two_stage_rft.data_access import CAMERAS
from robotwin_critic.vlac_finetune.common import (
    VideoFrameReader,
    make_tshape_state,
    read_jsonl,
)


def action_matrix(value: np.ndarray) -> np.ndarray:
    """Return executable actions, excluding VA_Server's first condition block."""
    value = np.asarray(value, dtype=np.float32)
    if value.ndim != 3 or value.shape[0] != 16 or value.shape[1] < 2:
        raise ValueError(f"Expected WAM actions [16,F,N], got {value.shape}")
    flattened = np.moveaxis(value, 0, -1).reshape(-1, 16)
    return flattened[value.shape[2] :]


def save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.asarray(image)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    Image.fromarray(image).save(path, quality=95)


def current_observation(row: dict, reader: VideoFrameReader) -> tuple[dict, np.ndarray]:
    frame_index = int(row["history_frame_indices"][-1])
    images = {
        camera: reader.read(row["video_paths"][camera], frame_index)
        for camera in CAMERAS
    }
    return {"obs": [images]}, make_tshape_state([images[camera] for camera in CAMERAS])


def distributed_context() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return rank, local_rank, world_size


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contexts", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidates-per-context", type=int, default=8)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--max-contexts", type=int, default=0)
    parser.add_argument("--config-name", default="robotwin")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.candidates_per_context <= 0:
        raise ValueError("--candidates-per-context must be positive")

    # Imports are delayed so --help and CPU unit tests do not require the WAM runtime.
    from wan_va.configs import VA_CONFIGS
    from wan_va.distributed.util import init_distributed
    from wan_va.wan_va_server import VA_Server

    rank, local_rank, world_size = distributed_context()
    init_distributed(world_size, local_rank, rank)
    config = VA_CONFIGS[args.config_name]
    config.rank = rank
    config.local_rank = local_rank
    config.world_size = world_size
    config.wan22_pretrained_model_name_or_path = str(args.model.resolve())
    config.save_root = str(args.output_dir.resolve())
    server = VA_Server(config)
    contexts = read_jsonl(args.contexts)
    if args.max_contexts:
        contexts = contexts[: args.max_contexts]
    output_jsonl = args.output_dir / "candidates.jsonl"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if rank == 0 and output_jsonl.exists() and not args.resume:
        raise FileExistsError(f"Refusing to overwrite {output_jsonl}")
    completed = set()
    if output_jsonl.is_file() and args.resume:
        completed = {
            str(row["candidate_id"]) for row in read_jsonl(output_jsonl)
        }

    with VideoFrameReader(max_cached_videos=12) as reader:
        output_handle = (
            output_jsonl.open("a" if args.resume else "w", encoding="utf-8")
            if rank == 0
            else None
        )
        try:
            for context_index, context in enumerate(contexts):
                observation, current_mosaic = current_observation(context, reader)
                context_dir = args.output_dir / f"context_{context_index:07d}"
                current_path = context_dir / "current.jpg"
                if rank == 0:
                    save_image(current_path, current_mosaic)
                server._reset(prompt=context["text"])
                prompt_embeds = server.prompt_embeds.detach().clone()
                negative_prompt_embeds = (
                    None
                    if server.negative_prompt_embeds is None
                    else server.negative_prompt_embeds.detach().clone()
                )
                for candidate_index in range(args.candidates_per_context):
                    candidate_id = f"{context['context_id']}/{candidate_index}"
                    if candidate_id in completed:
                        continue
                    seed = (
                        args.base_seed
                        + context_index * args.candidates_per_context
                        + candidate_index
                    )
                    torch.manual_seed(seed)
                    torch.cuda.manual_seed_all(seed)
                    if candidate_index:
                        server._reset(prompt=None)
                        server.prompt_embeds = prompt_embeds
                        server.negative_prompt_embeds = negative_prompt_embeds
                    actions, latents = server._infer(observation, frame_st_id=0)
                    candidate_dir = context_dir / f"candidate_{candidate_index:02d}"
                    action_path = candidate_dir / "actions.npy"
                    latent_path = candidate_dir / "latents.pt"
                    text_emb_path = candidate_dir / "text_emb.pt"
                    generated_path = candidate_dir / "generated_final.jpg"
                    if rank == 0:
                        candidate_dir.mkdir(parents=True, exist_ok=True)
                        np.save(action_path, action_matrix(actions))
                        torch.save(latents.detach().cpu(), latent_path)
                        torch.save(server.prompt_embeds.detach().cpu(), text_emb_path)
                        record = {
                            **context,
                            "candidate_id": candidate_id,
                            "candidate_index": candidate_index,
                            "seed": seed,
                            "action_path": str(action_path.resolve()),
                            "action_semantics": (
                                "relative executable actions; first VA_Server "
                                "conditioning block removed"
                            ),
                            "latent_path": str(latent_path.resolve()),
                            "text_emb_path": str(text_emb_path.resolve()),
                            "current_image": str(current_path.resolve()),
                            "generated_image": str(generated_path.resolve()),
                            "wam_conditioning": {
                                "rgb_frame_indices": [
                                    context["history_frame_indices"][-1]
                                ],
                                "language": True,
                                "proprio": False,
                                "note": (
                                    "Official first-chunk VA_Server consumes the current "
                                    "RGB state and language; history/proprio remain in "
                                    "the context record for future architectures."
                                ),
                            },
                        }
                        output_handle.write(
                            json.dumps(record, ensure_ascii=False) + "\n"
                        )
                        output_handle.flush()
                    if world_size > 1:
                        dist.barrier()
        finally:
            if output_handle is not None:
                output_handle.close()
    if rank == 0:
        summary = {
            "contexts": len(contexts),
            "candidates_per_context": args.candidates_per_context,
            "candidates": len(contexts) * args.candidates_per_context,
            "resumed_candidates": len(completed),
            "model": str(args.model.resolve()),
            "output": str(output_jsonl.resolve()),
            "generated_images_decoded": False,
        }
        (args.output_dir / "generation_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        print("ROBOTWIN_WAM_CANDIDATES_OK")


if __name__ == "__main__":
    main()
