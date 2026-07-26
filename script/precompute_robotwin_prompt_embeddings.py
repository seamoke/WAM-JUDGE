#!/usr/bin/env python3
import argparse
import hashlib
import itertools
import json
import re
import sys
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--seed-cache", type=Path, required=True)
    parser.add_argument("--robotwin-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--test-num", type=int, default=20)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--list-only", action="store_true")
    parser.add_argument("--enumerate-all-seen", action="store_true")
    return parser.parse_args()


def prompt_key(prompt):
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def enumerate_seen_prompts(task_name, episode_info, robotwin_root):
    task_path = (
        robotwin_root / "description" / "task_instruction" / f"{task_name}.json"
    )
    task_data = json.loads(task_path.read_text(encoding="utf-8"))
    stripped = {key.strip("{}"): value for key, value in episode_info.items()}
    arm_params = {
        key for key in stripped if len(key) == 1 and "a" <= key <= "z"
    }
    non_arm_params = set(stripped) - arm_params
    output = set()

    for instruction in task_data.get("seen", []):
        placeholders = set(re.findall(r"{([^}]+)}", instruction))
        exact = placeholders == set(stripped)
        omits_only_arms = (
            bool(arm_params)
            and placeholders.union(arm_params) == set(stripped)
            and not arm_params.intersection(placeholders)
        )
        if not exact and not omits_only_arms:
            continue
        if not non_arm_params.issubset(placeholders):
            continue

        keys = []
        options = []
        for key, value in stripped.items():
            if key not in placeholders:
                continue
            description_path = (
                robotwin_root
                / "description"
                / "objects_description"
                / f"{value}.json"
            )
            if description_path.is_file():
                descriptions = json.loads(
                    description_path.read_text(encoding="utf-8")
                ).get("seen", [])
                values = [f"the {description}" for description in descriptions]
            elif len(key) == 1 and "a" <= key <= "z":
                values = [f"the {value} arm"]
            else:
                values = [str(value)]
            if not values:
                raise RuntimeError(
                    f"No seen descriptions for {description_path}"
                )
            keys.append(key)
            options.append(values)

        for combination in itertools.product(*options):
            prompt = instruction
            for key, value in zip(keys, combination):
                prompt = prompt.replace("{" + key + "}", value)
            output.add(prompt)
    if not output:
        raise RuntimeError(
            f"No valid seen prompts for task={task_name} info={episode_info}"
        )
    return sorted(output)


def collect_prompts(seed_cache, robotwin_root, test_num, enumerate_all_seen):
    sys.path.insert(0, str(robotwin_root))
    from description.utils.generate_episode_instructions import (
        generate_episode_descriptions,
    )

    with seed_cache.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    records = []
    prompts = {}
    for task_name, entries in payload["tasks"].items():
        if len(entries) < test_num:
            raise RuntimeError(
                f"{task_name} has {len(entries)} cached seeds, need {test_num}"
            )
        for episode, entry in enumerate(entries[:test_num], start=1):
            seed = int(entry["seed"])
            episode_info = entry["episode_info"]
            if enumerate_all_seen:
                choices = enumerate_seen_prompts(
                    task_name, episode_info, robotwin_root
                )
            else:
                descriptions = generate_episode_descriptions(
                    task_name, [episode_info], test_num
                )
                choices = [
                    str(np.random.default_rng(seed).choice(descriptions[0]["seen"]))
                ]
            for candidate, prompt in enumerate(choices, start=1):
                key = prompt_key(prompt)
                prompts.setdefault(key, prompt)
                records.append(
                    {
                        "task": task_name,
                        "episode": episode,
                        "seed": seed,
                        "candidate": candidate,
                        "prompt_key": key,
                        "prompt": prompt,
                    }
                )

    return payload, records, prompts


def encode_prompt(text_encoder, tokenizer, prompt, device, dtype):
    import torch
    from diffusers.pipelines.wan.pipeline_wan import prompt_clean

    cleaned = prompt_clean(prompt)
    text_inputs = tokenizer(
        [cleaned],
        padding="max_length",
        max_length=512,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    mask = text_inputs.attention_mask
    seq_len = int(mask.gt(0).sum(dim=1).item())
    with torch.inference_mode():
        embeds = text_encoder(
            text_inputs.input_ids.to(device), mask.to(device)
        ).last_hidden_state
        embeds = embeds.to(dtype=dtype, device=device)
        embeds = embeds[:, :seq_len]
        embeds = torch.cat(
            [
                embeds,
                embeds.new_zeros(
                    embeds.shape[0], 512 - embeds.shape[1], embeds.shape[2]
                ),
            ],
            dim=1,
        )
    return embeds.cpu().contiguous()


def main():
    args = parse_args()
    payload, records, prompts = collect_prompts(
        args.seed_cache,
        args.robotwin_root,
        args.test_num,
        args.enumerate_all_seen,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "format_version": 1,
        "task_config": payload.get("task_config"),
        "seed_cache": str(args.seed_cache),
        "test_num": args.test_num,
        "tasks": len(payload["tasks"]),
        "episodes": len(payload["tasks"]) * args.test_num,
        "candidate_records": len(records),
        "unique_prompts": len(prompts),
        "enumerate_all_seen": args.enumerate_all_seen,
        "records": records,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"Collected {len(records)} candidate prompts across "
        f"{len(payload['tasks'])} tasks; unique={len(prompts)}"
    )
    if args.list_only:
        print(f"Manifest: {manifest_path}")
        return
    if args.model_path is None:
        raise ValueError("--model-path is required unless --list-only is used")

    import torch
    from wan_va.modules.utils import load_text_encoder, load_tokenizer

    dtype = torch.bfloat16
    tokenizer = load_tokenizer(str(args.model_path / "tokenizer"))
    text_encoder = load_text_encoder(
        str(args.model_path / "text_encoder"),
        torch_dtype=dtype,
        torch_device=args.device,
    )
    text_encoder.eval()

    negative_path = args.output_dir / "negative.pt"
    if not negative_path.exists():
        torch.save(encode_prompt(text_encoder, tokenizer, "", args.device, dtype), negative_path)

    for index, (key, prompt) in enumerate(sorted(prompts.items()), start=1):
        output_path = args.output_dir / f"{key}.pt"
        if not output_path.exists():
            torch.save(
                encode_prompt(text_encoder, tokenizer, prompt, args.device, dtype),
                output_path,
            )
        if index == 1 or index % 25 == 0 or index == len(prompts):
            print(f"Encoded {index}/{len(prompts)} unique prompts", flush=True)

    del text_encoder
    torch.cuda.empty_cache()
    manifest["dtype"] = str(dtype)
    manifest["shape"] = list(torch.load(negative_path, map_location="cpu").shape)
    manifest["complete"] = True
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Prompt embedding cache complete: {manifest_path}")


if __name__ == "__main__":
    main()
