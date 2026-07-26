#!/usr/bin/env python3
"""Launch wan_va_server with a custom checkpoint model path."""
import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=None, help="Full model dir (vae+transformer+...)")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Training checkpoint dir (transformer only); use with --base-model",
    )
    parser.add_argument(
        "--base-model",
        default=None,
        help="Base model dir with vae/tokenizer/text_encoder",
    )
    parser.add_argument("--config-name", default="libero")
    parser.add_argument("--port", type=int, default=29056)
    parser.add_argument("--save-root", default="./visualization")
    args = parser.parse_args()

    if args.model_path:
        model_path = os.path.abspath(args.model_path)
    elif args.checkpoint and args.base_model:
        from evaluation.libero.prepare_eval_model import prepare_eval_model

        cache_root = os.environ.get(
            "EVAL_MODEL_CACHE",
            os.path.join(os.environ.get("TMPDIR", "/tmp"), "lingbot_eval_symlinks"),
        )
        model_path = prepare_eval_model(args.checkpoint, args.base_model, cache_root)
    else:
        parser.error("Provide --model-path or both --checkpoint and --base-model")

    from wan_va.configs import VA_CONFIGS

    config = VA_CONFIGS[args.config_name]
    config.wan22_pretrained_model_name_or_path = model_path

    sys.argv = [
        "wan_va_server.py",
        "--config-name",
        args.config_name,
        "--port",
        str(args.port),
        "--save_root",
        args.save_root,
    ]

    from wan_va.wan_va_server import main as server_main

    server_main()


if __name__ == "__main__":
    main()
