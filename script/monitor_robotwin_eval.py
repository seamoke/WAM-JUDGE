#!/usr/bin/env python3
"""RoboTwin eval bottleneck monitor (non-invasive; attach to running jobs).

Parses client/server logs + nvidia-smi. Does not modify eval processes.

Usage:
  python script/monitor_robotwin_eval.py --once
  python script/monitor_robotwin_eval.py --watch --interval 15
  python script/monitor_robotwin_eval.py --analyze-batch 20260701_203822
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs" / "robotwin_eval"
REPORT_DIR = LOG_DIR / "monitor"
STATE_PATH = REPORT_DIR / "state.json"

STEP_RE = re.compile(r"step:\s*\x1b\[92m(\d+)\s*/\s*(\d+)")
STEP_RE_PLAIN = re.compile(r"step:\s*(\d+)\s*/\s*(\d+)")
TASK_RE = re.compile(r"Task Name:\s*(\w+)")
SUCCESS_RE = re.compile(r"Success rate:.*?(\d+)/(\d+)")
SHARD_RE = re.compile(r"shard(\d+)_gpu(\d+)")


def latest_batch() -> str | None:
    batches: list[tuple[float, str]] = []
    for p in LOG_DIR.glob("client_checkpoint_step_*_*.log"):
        m = re.search(r"_(\d{8}_\d{6})\.log$", p.name)
        if m:
            batches.append((p.stat().st_mtime, m.group(1)))
    if not batches:
        return None
    batches.sort()
    return batches[-1][1]


def client_logs(batch: str | None = None) -> list[Path]:
    batch = batch or latest_batch()
    if not batch:
        return []
    return sorted(LOG_DIR.glob(f"client_checkpoint_step_*_{batch}.log"))


def parse_client_log(text: str) -> dict:
    tasks = TASK_RE.findall(text)
    cur_task = tasks[-1] if tasks else None
    srs = SUCCESS_RE.findall(text)
    eps_done = int(srs[-1][1]) if srs else 0
    eps_succ = int(srs[-1][0]) if srs else 0
    steps = [int(x) for m in STEP_RE.finditer(text) for x in [m.group(1)]]
    if not steps:
        steps = [int(m.group(1)) for m in STEP_RE_PLAIN.finditer(text)]
    step_lim = None
    for m in STEP_RE.finditer(text):
        step_lim = int(m.group(2))
    if step_lim is None:
        for m in STEP_RE_PLAIN.finditer(text):
            step_lim = int(m.group(2))
    cur_step = max(steps) if steps else 0
    return {
        "task": cur_task,
        "episodes_done": eps_done,
        "episodes_succ": eps_succ,
        "cur_step": cur_step,
        "step_lim": step_lim,
        "render_well": "Render Well" in text,
        "device_lost": "DeviceLost" in text or "Aborted" in text,
        "success_lines": len(srs),
    }


def gpu_stats() -> list[dict]:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,utilization.gpu,utilization.memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []
    rows = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 4:
            rows.append(
                {
                    "gpu": int(parts[0]),
                    "mem_mib": int(parts[1]),
                    "gpu_util": int(parts[2]),
                    "mem_util": int(parts[3]),
                }
            )
    return rows


def proc_counts() -> dict[str, int]:
    def cnt(pat: str) -> int:
        try:
            return int(
                subprocess.check_output(["pgrep", "-c", "-f", pat], text=True).strip()
            )
        except subprocess.CalledProcessError:
            return 0

    return {
        "clients": cnt("eval_polict_client_openpi"),
        "servers": cnt("run_server_ckpt"),
        "orchestrator": cnt("run_robotwin_eval.sh"),
    }


@dataclass
class ShardWatch:
    shard: str
    server_gpu: str
    sim_gpu: str | None = None
    phase: str = "unknown"
    task: str | None = None
    cur_step: int = 0
    step_lim: int | None = None
    episodes_done: int = 0
    last_step_ts: float = 0.0
    last_step_val: int = 0
    step_rate_per_min: float = 0.0
    idle_sec: float = 0.0
    init_done: bool = False
    episode_durations: list[float] = field(default_factory=list)
    phase_seconds: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    last_phase: str = "unknown"
    last_phase_ts: float = field(default_factory=time.time)


def infer_sim_gpu(text_head: str) -> str | None:
    m = re.search(r"ROBOTWIN_VULKAN_GPU=(\d+)", text_head[:4000])
    return m.group(1) if m else None


def classify_phase(info: dict, idle_sec: float, sim_gpu_util: int | None = None) -> str:
    """Classify shard phase from log + optional sim-GPU util.

    Important: policy ``step`` only increments once per ``take_action()`` call, while
    the inner physics loop can run for minutes. Long plateaus at the same step are
    usually sim (physics+render), not websocket inference — use sim GPU util to tell.
    """
    if info["device_lost"]:
        return "crashed"
    if not info["render_well"]:
        return "sim_init"
    if info["cur_step"] == 0 and info["episodes_done"] == 0:
        return "env_reset"
    step_lim = info.get("step_lim")
    if step_lim and info["cur_step"] >= step_lim:
        return "episode_boundary"
    if idle_sec > 45 and info["cur_step"] > 0:
        # High util on sim GPU → physics/render inside take_action, not model.infer.
        if sim_gpu_util is not None and sim_gpu_util >= 50:
            return "sim_stepping"
        return "infer_or_stall"
    if info["cur_step"] > 0:
        return "sim_stepping"
    return "between_episodes"


def load_state() -> dict:
    if STATE_PATH.is_file():
        return json.loads(STATE_PATH.read_text())
    return {}


def save_state(state: dict) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2))


def snapshot_batch(batch: str | None = None) -> dict:
    batch = batch or latest_batch()
    shards = []
    for p in client_logs(batch):
        m = SHARD_RE.search(p.name)
        text = p.read_text(errors="ignore")
        info = parse_client_log(text)
        age_min = (time.time() - p.stat().st_mtime) / 60
        shards.append(
            {
                "shard": m.group(1) if m else "?",
                "server_gpu": m.group(2) if m else "?",
                "sim_gpu": infer_sim_gpu(text),
                "log": p.name,
                "log_age_min": round(age_min, 1),
                **info,
            }
        )
    return {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "batch": batch,
        "procs": proc_counts(),
        "gpus": gpu_stats(),
        "shards": shards,
    }


def update_watch(state: dict, batch: str, interval: float) -> dict:
    now = time.time()
    shard_state: dict = state.setdefault("shards", {})
    totals = defaultdict(float)
    gpu_utils = {g["gpu"]: g["gpu_util"] for g in gpu_stats()}

    for p in client_logs(batch):
        m = SHARD_RE.search(p.name)
        if not m:
            continue
        sid = m.group(1)
        text = p.read_text(errors="ignore")
        info = parse_client_log(text)
        sw = shard_state.get(sid)
        if sw is None:
            sw = ShardWatch(shard=sid, server_gpu=m.group(2), sim_gpu=infer_sim_gpu(text)).__dict__
            sw["episode_start_ts"] = now
            sw["last_eps"] = info["episodes_done"]
            shard_state[sid] = sw

        sw["sim_gpu"] = infer_sim_gpu(text) or sw.get("sim_gpu")
        sw["task"] = info["task"]
        sw["episodes_done"] = info["episodes_done"]
        sw["cur_step"] = info["cur_step"]
        sw["step_lim"] = info["step_lim"]
        sw["init_done"] = info["render_well"]

        idle = now - sw.get("last_step_ts", now) if sw.get("last_step_ts") else 0
        if info["cur_step"] != sw.get("last_step_val", 0):
            sw["last_step_ts"] = now
            sw["last_step_val"] = info["cur_step"]
            idle = 0
        sw["idle_sec"] = idle

        sim_gpu = sw.get("sim_gpu")
        sim_util = gpu_utils.get(int(sim_gpu)) if sim_gpu and str(sim_gpu).isdigit() else None
        phase = classify_phase(info, idle, sim_gpu_util=sim_util)
        dt = now - sw.get("last_phase_ts", now)
        prev = sw.get("last_phase", phase)
        sw.setdefault("phase_seconds", {})
        sw["phase_seconds"][prev] = sw["phase_seconds"].get(prev, 0) + dt
        sw["last_phase"] = phase
        sw["last_phase_ts"] = now
        sw["phase"] = phase

        if info["episodes_done"] > sw.get("last_eps", 0):
            dur = now - sw.get("episode_start_ts", now)
            sw.setdefault("episode_durations", []).append(dur)
            sw["last_eps"] = info["episodes_done"]
            sw["episode_start_ts"] = now

        if sw.get("last_step_ts") and info["cur_step"] > 0:
            span = max(now - sw["last_step_ts"], 1e-3)
            # rough instantaneous rate over interval window
            sw["step_rate_per_min"] = 60.0 / max(span / max(info["cur_step"] - sw.get("rate_base_step", 0), 1), 0.5)

        for k, v in sw.get("phase_seconds", {}).items():
            totals[k] += v

        shard_state[sid] = sw

    state["batch"] = batch
    state["updated"] = datetime.now().isoformat(timespec="seconds")
    state["totals"] = dict(totals)
    return state


def format_report(snap: dict, state: dict | None = None) -> str:
    lines = [
        f"=== RoboTwin monitor {snap['ts']} batch={snap['batch']} ===",
        f"procs: clients={snap['procs']['clients']} servers={snap['procs']['servers']}",
    ]
    for g in snap["gpus"]:
        lines.append(
            f"  GPU{g['gpu']}: util={g['gpu_util']}% mem={g['mem_mib']}MiB"
        )

    if state and state.get("totals"):
        t = state["totals"]
        total = sum(t.values()) or 1
        lines.append("\n--- accumulated phase time (all shards, since monitor start) ---")
        for ph in sorted(t, key=t.get, reverse=True):
            lines.append(f"  {ph:18s} {t[ph]/60:6.1f} min  ({100*t[ph]/total:4.1f}%)")

    lines.append("\n--- per shard ---")
    shard_state = (state or {}).get("shards", {})
    for s in sorted(snap["shards"], key=lambda x: int(x["shard"])):
        sid = s["shard"]
        sw = shard_state.get(sid, {})
        phase = sw.get("phase", "snapshot")
        ep_durs = sw.get("episode_durations", [])
        avg_ep = sum(ep_durs) / len(ep_durs) if ep_durs else 0
        lines.append(
            f"  shard{sid:>2} srv{s['server_gpu']} sim{s.get('sim_gpu','?')} "
            f"{phase:16s} task={s.get('task') or '-':24s} "
            f"ep={s['episodes_done']}/10 step={s['cur_step']}/{s.get('step_lim') or '?'}"
            + (f" avg_ep={avg_ep/60:.1f}m" if avg_ep else "")
            + (" DL!" if s.get("device_lost") else "")
        )
    return "\n".join(lines)


def analyze_batch(batch: str) -> str:
    """Retrospective coarse breakdown from log file ages + episode counts."""
    logs = client_logs(batch)
    if not logs:
        return f"No client logs for batch {batch}"
    try:
        t0 = min(p.stat().st_mtime for p in logs) - 3600
        t0 = min(p.stat().st_ctime for p in logs)
    except ValueError:
        t0 = time.time()
    # batch timestamp
    t_batch = time.mktime(time.strptime(batch, "%Y%m%d_%H%M%S"))
    t0 = min(t0, t_batch)

    lines = [f"=== Retrospective analyze batch {batch} ===", f"batch_start~{datetime.fromtimestamp(t_batch)}"]
    total_eps = 0
    phase_est = defaultdict(float)

    for p in logs:
        m = SHARD_RE.search(p.name)
        text = p.read_text(errors="ignore")
        info = parse_client_log(text)
        elapsed_h = max((time.time() - t_batch) / 3600, 1e-6)
        eps = info["episodes_done"]
        total_eps += eps
        # heuristic: if no render well yet all time in init; else episodes * avg + init
        if not info["render_well"]:
            init_frac = 1.0
        else:
            init_frac = 0.15  # typical ~10-15% first hour
        init_h = elapsed_h * init_frac if eps == 0 else min(elapsed_h * 0.1, 0.25)
        ep_h = max(elapsed_h - init_h, 0)
        avg_ep_min = (ep_h * 60 / eps) if eps else 0

        lines.append(
            f"  shard{m.group(1) if m else '?'} srv{m.group(2) if m else '?'} "
            f"sim={infer_sim_gpu(text)} task={info['task']} eps={eps} "
            f"elapsed={elapsed_h:.2f}h init~{init_h*60:.0f}m "
            f"avg_ep~{avg_ep_min:.1f}m step_lim={info['step_lim']}"
        )
        phase_est["sim_init_est"] += init_h
        phase_est["episode_est"] += ep_h

    cluster_h = max((time.time() - t_batch) / 3600, 1e-6)
    lines.append(f"\ncluster: {len(logs)} shards, {total_eps} episodes logged, {cluster_h:.2f}h since batch")
    if total_eps:
        lines.append(f"cluster avg ~{cluster_h*60*len(logs)/total_eps:.1f} shard-episodes/h")
        lines.append(f"~{total_eps/(cluster_h*len(logs)):.2f} ep/shard/h")
    lines.append("\nInterpretation:")
    lines.append("  - step in logs only ticks once per take_action(); inner physics can take minutes")
    lines.append("  - old monitor labeled that plateau as infer_or_stall — usually still sim (GPU0/3 busy)")
    lines.append("  - true infer wait: step flat AND sim-GPU util low for 45s+")
    lines.append("  - long step_lim tasks (800/900) inflate per-episode time")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Monitor RoboTwin eval bottlenecks")
    ap.add_argument("--batch", default=None, help="Log batch timestamp YYYYMMDD_HHMMSS")
    ap.add_argument("--once", action="store_true", help="Print one snapshot and exit")
    ap.add_argument("--watch", action="store_true", help="Loop and accumulate phase timings")
    ap.add_argument("--interval", type=float, default=15.0)
    ap.add_argument("--analyze-batch", metavar="BATCH", help="Retrospective log analysis")
    args = ap.parse_args()

    if args.analyze_batch:
        print(analyze_batch(args.analyze_batch))
        return 0

    batch = args.batch or latest_batch()
    if not batch:
        print("No client logs found", file=sys.stderr)
        return 1

    if args.once or not args.watch:
        snap = snapshot_batch(batch)
        state = load_state() if STATE_PATH.exists() else None
        report = format_report(snap, state)
        print(report)
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        out = REPORT_DIR / f"snapshot_{batch}.txt"
        out.write_text(report + "\n")
        print(f"\n(wrote {out})")
        if not args.watch:
            return 0

    state = load_state()
    print(f"Watching batch {batch} every {args.interval}s → {REPORT_DIR}")
    print("Ctrl+C to stop. Phase times accumulate in state.json\n")
    try:
        while True:
            snap = snapshot_batch(batch)
            state = update_watch(state, batch, args.interval)
            save_state(state)
            report = format_report(snap, state)
            # overwrite live report
            live = REPORT_DIR / f"live_{batch}.txt"
            live.write_text(report + "\n")
            jsonl = REPORT_DIR / f"metrics_{batch}.jsonl"
            with jsonl.open("a") as f:
                f.write(json.dumps({"snap": snap, "totals": state.get("totals")}) + "\n")
            print(report)
            print("-" * 60)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nStopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
