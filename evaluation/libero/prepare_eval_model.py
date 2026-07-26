#!/usr/bin/env python3
"""Prepare a full eval model dir from base weights + training checkpoint."""
import argparse
import json
import os
from pathlib import Path


def prepare_eval_model(checkpoint_path: str, base_model: str, out_root: str) -> str:
    checkpoint_path = os.path.abspath(checkpoint_path)
    base_model = os.path.abspath(base_model)
    out_root = os.path.abspath(out_root)

    ckpt_name = os.path.basename(checkpoint_path.rstrip("/"))
    eval_dir = os.path.join(out_root, ckpt_name)
    marker = os.path.join(eval_dir, ".ready")
    if os.path.isfile(marker):
        with open(marker) as f:
            recorded = f.read()
        expected = f"base={base_model}\ncheckpoint={checkpoint_path}\n"
        if recorded == expected:
            return eval_dir

    transformer_src = os.path.join(checkpoint_path, "transformer")
    if not os.path.isdir(transformer_src):
        raise FileNotFoundError(f"Missing transformer in checkpoint: {transformer_src}")

    os.makedirs(eval_dir, exist_ok=True)

    for sub in ("vae", "tokenizer", "text_encoder"):
        src = os.path.join(base_model, sub)
        dst = os.path.join(eval_dir, sub)
        if not os.path.isdir(src):
            raise FileNotFoundError(f"Missing base model component: {src}")
        if not os.path.lexists(dst):
            os.symlink(src, dst)

    transformer_dst = os.path.join(eval_dir, "transformer")
    os.makedirs(transformer_dst, exist_ok=True)

    config_src = os.path.join(transformer_src, "config.json")
    config_dst = os.path.join(transformer_dst, "config.json")
    with open(config_src) as f:
        config = json.load(f)
    config["attn_mode"] = "torch"
    with open(config_dst, "w") as f:
        json.dump(config, f, indent=2)
        f.write("\n")

    weight_name = "diffusion_pytorch_model.safetensors"
    weight_src = os.path.join(transformer_src, weight_name)
    weight_dst = os.path.join(transformer_dst, weight_name)
    if not os.path.isfile(weight_src):
        raise FileNotFoundError(f"Missing checkpoint weights: {weight_src}")
    if not os.path.lexists(weight_dst):
        os.symlink(weight_src, weight_dst)

    Path(marker).write_text(f"base={base_model}\ncheckpoint={checkpoint_path}\n")
    return eval_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Training checkpoint dir")
    parser.add_argument(
        "--base-model",
        default="/workspace/lingbot-va/checkpoints/lingbot-va-base",
        help="Base model with vae/tokenizer/text_encoder",
    )
    parser.add_argument(
        "--out-root",
        default="/workspace/lingbot-va/train_out/libero/eval_models",
        help="Directory for symlink-only eval layout (no weight copy)",
    )
    args = parser.parse_args()
    eval_dir = prepare_eval_model(args.checkpoint, args.base_model, args.out_root)
    print(eval_dir)


if __name__ == "__main__":
    main()
