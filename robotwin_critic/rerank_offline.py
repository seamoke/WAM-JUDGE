from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from robotwin_critic.common import DEFAULT_OUTPUT_ROOT, read_jsonl, stable_int, state_feature, text_feature, write_jsonl
from robotwin_critic.models import RobotWinConsistencyFilter, RobotWinProcessCritic


def load_process(path: Path, device: torch.device) -> RobotWinProcessCritic:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    args = ckpt.get("args", {})
    model = RobotWinProcessCritic(
        hidden_dim=int(args.get("hidden_dim", 512)),
        task_buckets=int(args.get("task_buckets", 4096)),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline rerank candidate chunks with a trained process critic.")
    parser.add_argument("--candidates-jsonl", type=Path, required=True)
    parser.add_argument("--process-checkpoint", type=Path, default=DEFAULT_OUTPUT_ROOT / "process_critic" / "best.pt")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_ROOT / "rerank_results.jsonl")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--task-buckets", type=int, default=4096)
    args = parser.parse_args()
    device = torch.device(args.device)
    model = load_process(args.process_checkpoint, device)
    rows = read_jsonl(args.candidates_jsonl)
    out = []
    with torch.no_grad():
        for row in rows:
            latents = row["latents"]
            final_ref = row.get("goal_ref", row)
            state = state_feature(latents, row["state_frame"]).unsqueeze(0).to(device)
            future = state_feature(latents, row["future_frame"]).unsqueeze(0).to(device)
            final_state = state_feature(final_ref["latents"], row.get("final_frame", final_ref["length"] - 1)).unsqueeze(0).to(device)
            text = text_feature(latents).unsqueeze(0).to(device)
            task_id = torch.tensor([stable_int(row["task_name"], args.task_buckets)], dtype=torch.long, device=device)
            u_state = model(state, final_state, text, task_id)
            u_future = model(future, final_state, text, task_id)
            scored = dict(row)
            scored["process_score"] = float((u_future - u_state).item())
            out.append(scored)
    out.sort(key=lambda r: r["process_score"], reverse=True)
    write_jsonl(args.output, out)
    print(json.dumps({"input": len(rows), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()

