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
from robotwin_critic.two_stage_rft.protocol import sha256_file
from robotwin_critic.vlac_finetune.common import (
    VideoFrameReader,
    make_tshape_state,
    read_jsonl,
)


def action_matrix(
    value: np.ndarray, *, frame_chunk_size: int = 2, action_per_frame: int = 16
) -> np.ndarray:
    """Return executable actions, excluding VA_Server's first condition block."""
    value = np.asarray(value, dtype=np.float32)
    expected = (16, frame_chunk_size, action_per_frame)
    if value.shape != expected:
        raise ValueError(f"Expected WAM actions {expected}, got {value.shape}")
    flattened = np.moveaxis(value, 0, -1).reshape(-1, 16)
    executable = flattened[action_per_frame:]
    expected_steps = (frame_chunk_size - 1) * action_per_frame
    if executable.shape != (expected_steps, 16):
        raise AssertionError(f"Unexpected executable action shape {executable.shape}")
    return executable


def save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.asarray(image)
    if image.dtype != np.uint8:
        if (
            np.issubdtype(image.dtype, np.floating)
            and np.all(np.isfinite(image))
            and float(image.min()) >= 0.0
            and float(image.max()) <= 1.0 + 1e-6
        ):
            image = image * 255.0
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
    parser.add_argument("--inference-batch-size", type=int, default=1)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--max-contexts", type=int, default=0)
    parser.add_argument("--config-name", default="robotwin")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.candidates_per_context <= 0:
        raise ValueError("--candidates-per-context must be positive")
    if args.inference_batch_size <= 0:
        raise ValueError("--inference-batch-size must be positive")

    contexts = read_jsonl(args.contexts)
    if args.max_contexts:
        contexts = contexts[: args.max_contexts]
    context_file_sha256 = sha256_file(args.contexts)
    output_jsonl = args.output_dir / "candidates.jsonl"
    summary_path = args.output_dir / "generation_summary.json"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if output_jsonl.exists() and not args.resume:
        raise FileExistsError(f"Refusing to overwrite {output_jsonl}")
    if output_jsonl.exists() and args.resume:
        if not summary_path.is_file():
            raise RuntimeError(
                f"Cannot safely resume {output_jsonl}: generation summary is missing"
            )
        previous = json.loads(summary_path.read_text(encoding="utf-8"))
        if previous.get("context_file_sha256") != context_file_sha256:
            raise RuntimeError(
                "Cannot resume candidates generated from a different context file; "
                "use a new output directory"
            )
        resume_contract = {
            "candidates_per_context": args.candidates_per_context,
            "inference_batch_size": args.inference_batch_size,
            "base_seed": args.base_seed,
            "model": str(args.model.resolve()),
            "config_name": args.config_name,
        }
        mismatches = {
            key: (previous.get(key), value)
            for key, value in resume_contract.items()
            if previous.get(key) != value
        }
        if mismatches:
            raise RuntimeError(
                "Cannot resume with a changed generation contract: "
                + json.dumps(mismatches, sort_keys=True)
            )

    # Imports are delayed so --help and CPU unit tests do not require the WAM runtime.
    from wan_va.configs import VA_CONFIGS
    from wan_va.distributed.util import init_distributed
    from robotwin_critic.two_stage_rft.batched_wam_server import BatchedVAServer

    rank, local_rank, world_size = distributed_context()
    init_distributed(world_size, local_rank, rank)
    config = VA_CONFIGS[args.config_name]
    config.rank = rank
    config.local_rank = local_rank
    config.world_size = world_size
    config.wan22_pretrained_model_name_or_path = str(args.model.resolve())
    config.save_root = str(args.output_dir.resolve())
    server = BatchedVAServer(config)
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
            for batch_start in range(0, len(contexts), args.inference_batch_size):
                indexed_contexts = list(
                    enumerate(
                        contexts[batch_start : batch_start + args.inference_batch_size],
                        start=batch_start,
                    )
                )
                observations = []
                current_paths = []
                for context_index, context in indexed_contexts:
                    observation, current_mosaic = current_observation(context, reader)
                    context_dir = args.output_dir / f"context_{context_index:07d}"
                    current_path = context_dir / "current.jpg"
                    if rank == 0:
                        save_image(current_path, current_mosaic)
                    observations.append(observation)
                    current_paths.append(current_path)
                for candidate_index in range(args.candidates_per_context):
                    active = [
                        (local_index, context_index, context)
                        for local_index, (context_index, context) in enumerate(
                            indexed_contexts
                        )
                        if f"{context['context_id']}/{candidate_index}" not in completed
                    ]
                    if not active:
                        continue
                    active_observations = [
                        observations[local_index] for local_index, _, _ in active
                    ]
                    prompts = [context["text"] for _, _, context in active]
                    seeds = [
                        args.base_seed
                        + context_index * args.candidates_per_context
                        + candidate_index
                        for _, context_index, _ in active
                    ]
                    actions_batch, latents_batch = server.infer_batch(
                        active_observations, prompts, seeds
                    )
                    if rank == 0:
                        for active_index, (
                            local_index,
                            context_index,
                            context,
                        ) in enumerate(active):
                            seed = seeds[active_index]
                            candidate_id = (
                                f"{context['context_id']}/{candidate_index}"
                            )
                            context_dir = (
                                args.output_dir / f"context_{context_index:07d}"
                            )
                            candidate_dir = (
                                context_dir / f"candidate_{candidate_index:02d}"
                            )
                            action_path = candidate_dir / "actions.npy"
                            latent_path = candidate_dir / "latents.pt"
                            text_emb_path = candidate_dir / "text_emb.pt"
                            generated_path = candidate_dir / "generated_final.jpg"
                            candidate_dir.mkdir(parents=True, exist_ok=True)
                            executable_actions = action_matrix(
                                actions_batch[active_index].numpy(),
                                frame_chunk_size=int(config.frame_chunk_size),
                                action_per_frame=int(config.action_per_frame),
                            )
                            candidate_latents = latents_batch[
                                active_index : active_index + 1
                            ].detach().cpu()
                            np.save(action_path, executable_actions)
                            torch.save(candidate_latents, latent_path)
                            torch.save(
                                server.prompt_embeds[
                                    active_index : active_index + 1
                                ].detach().cpu(),
                                text_emb_path,
                            )
                            record = {
                                **context,
                                "candidate_id": candidate_id,
                                "candidate_index": candidate_index,
                                "seed": seed,
                                "inference_batch_size": len(active),
                                "action_path": str(action_path.resolve()),
                                "action_semantics": (
                                    "relative executable actions; first VA_Server "
                                    "conditioning block removed"
                                ),
                                "latent_frames": int(candidate_latents.shape[2]),
                                "executable_action_steps": int(
                                    executable_actions.shape[0]
                                ),
                                "latent_path": str(latent_path.resolve()),
                                "text_emb_path": str(text_emb_path.resolve()),
                                "current_image": str(
                                    current_paths[local_index].resolve()
                                ),
                                "generated_image": str(generated_path.resolve()),
                                "wam_conditioning": {
                                    "rgb_frame_indices": [
                                        context["history_frame_indices"][-1]
                                    ],
                                    "language": True,
                                    "proprio": False,
                                    "note": (
                                        "Official first-chunk VA_Server consumes the "
                                        "current RGB state and language; history/proprio "
                                        "remain in the context record."
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
            "inference_batch_size": args.inference_batch_size,
            "base_seed": args.base_seed,
            "candidates": len(contexts) * args.candidates_per_context,
            "resumed_candidates": len(completed),
            "model": str(args.model.resolve()),
            "config_name": args.config_name,
            "output": str(output_jsonl.resolve()),
            "context_file": str(args.contexts.resolve()),
            "context_file_sha256": context_file_sha256,
            "generated_images_decoded": False,
        }
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        print("ROBOTWIN_WAM_CANDIDATES_OK")


if __name__ == "__main__":
    try:
        main()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
