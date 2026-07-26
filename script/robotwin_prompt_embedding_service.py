#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import socketserver
import threading
import time
from pathlib import Path

import torch
from diffusers.pipelines.wan.pipeline_wan import prompt_clean

from wan_va.modules.utils import load_text_encoder, load_tokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=31056)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def encode_prompt(text_encoder, tokenizer, prompt, device):
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
        embeds = embeds.to(dtype=torch.bfloat16, device=device)[:, :seq_len]
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


class PromptEmbeddingService:
    def __init__(self, model_path, cache_dir, device):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.lock = threading.Lock()
        self.tokenizer = load_tokenizer(str(model_path / "tokenizer"))
        self.text_encoder = load_text_encoder(
            str(model_path / "text_encoder"),
            torch_dtype=torch.bfloat16,
            torch_device=device,
        )
        self.text_encoder.eval()
        self._ensure_negative()

    def _save_atomic(self, tensor, output_path):
        temporary = output_path.with_name(
            f".{output_path.name}.{os.getpid()}.tmp"
        )
        torch.save(tensor, temporary)
        os.replace(temporary, output_path)

    def _ensure_negative(self):
        path = self.cache_dir / "negative.pt"
        if not path.is_file():
            self._save_atomic(
                encode_prompt(
                    self.text_encoder, self.tokenizer, "", self.device
                ),
                path,
            )

    def ensure(self, prompt):
        key = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        output_path = self.cache_dir / f"{key}.pt"
        if output_path.is_file():
            return key, False
        with self.lock:
            if output_path.is_file():
                return key, False
            started_at = time.perf_counter()
            self._save_atomic(
                encode_prompt(
                    self.text_encoder, self.tokenizer, prompt, self.device
                ),
                output_path,
            )
            print(
                json.dumps(
                    {
                        "event": "encoded",
                        "key": key,
                        "seconds": round(time.perf_counter() - started_at, 3),
                        "prompt": prompt,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        return key, True


class Handler(socketserver.StreamRequestHandler):
    def handle(self):
        try:
            request = json.loads(self.rfile.readline().decode("utf-8"))
            if request.get("op") == "ping":
                response = {"ok": True, "status": "ready"}
            else:
                prompt = request.get("prompt")
                if not isinstance(prompt, str):
                    raise TypeError("prompt must be a string")
                key, created = self.server.service.ensure(prompt)
                response = {"ok": True, "key": key, "created": created}
        except Exception as error:
            response = {
                "ok": False,
                "error": f"{type(error).__name__}: {error}",
            }
        self.wfile.write(
            (json.dumps(response, ensure_ascii=False) + "\n").encode("utf-8")
        )
        self.wfile.flush()


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    request_queue_size = 64
    daemon_threads = True


def main():
    args = parse_args()
    service = PromptEmbeddingService(
        args.model_path, args.cache_dir, args.device
    )
    with Server((args.host, args.port), Handler) as server:
        server.service = service
        print(
            json.dumps(
                {
                    "event": "ready",
                    "host": args.host,
                    "port": args.port,
                    "device": args.device,
                    "cache_dir": str(args.cache_dir),
                }
            ),
            flush=True,
        )
        server.serve_forever(poll_interval=0.5)


if __name__ == "__main__":
    main()
