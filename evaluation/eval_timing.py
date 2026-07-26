"""Wall-clock timing for eval (infer vs sim). Enable with ROBOTWIN_EVAL_TIMING=1 or LIBERO_EVAL_TIMING=1."""

from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_TIMING_ENV_KEYS = ("ROBOTWIN_EVAL_TIMING", "LIBERO_EVAL_TIMING")


def timing_enabled() -> bool:
    for key in _TIMING_ENV_KEYS:
        val = os.environ.get(key)
        if val is not None and val.strip().lower() in ("1", "true", "yes", "on"):
            return True
    return False


def _fmt_sec(sec: float) -> str:
    if sec >= 120:
        return f"{sec / 60:.1f}m"
    return f"{sec:.1f}s"


class EvalTiming:
    """Accumulates wall time per phase for one eval run."""

    def __init__(
        self,
        task_name: str,
        save_root: str | Path,
        st_seed: int = 0,
        *,
        timing_subdir: str | None = None,
    ):
        self.task_name = task_name
        self.save_root = Path(save_root)
        self.st_seed = st_seed
        self.timing_subdir = timing_subdir
        self._ep_buckets: dict[str, float] = defaultdict(float)
        self._ep_counts: dict[str, int] = defaultdict(int)
        self._run_buckets: dict[str, float] = defaultdict(float)
        self._run_counts: dict[str, int] = defaultdict(int)
        self._ep_t0: float | None = None
        self._jsonl: Path | None = None

    def begin_episode(self) -> None:
        self._ep_buckets.clear()
        self._ep_counts.clear()
        self._ep_t0 = time.perf_counter()

    def add(self, bucket: str, sec: float, *, count: int = 1) -> None:
        if sec < 0:
            return
        self._ep_buckets[bucket] += sec
        self._ep_counts[bucket] += count
        self._run_buckets[bucket] += sec
        self._run_counts[bucket] += count

    @contextmanager
    def section(self, bucket: str, *, count: int = 1):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.add(bucket, time.perf_counter() - t0, count=count)

    def _jsonl_path(self) -> Path:
        if self._jsonl is None:
            if self.timing_subdir:
                d = self.save_root / self.timing_subdir
            else:
                d = self.save_root / f"stseed-{self.st_seed}" / "eval_timing"
            d.mkdir(parents=True, exist_ok=True)
            self._jsonl = d / f"{self.task_name}.jsonl"
        return self._jsonl

    def finish_episode(
        self,
        *,
        episode: int,
        policy_steps: int,
        step_lim: int,
        success: bool,
        seed: int,
    ) -> None:
        ep_total = (
            time.perf_counter() - self._ep_t0 if self._ep_t0 is not None else 0.0
        )
        buckets = dict(self._ep_buckets)
        other = max(0.0, ep_total - sum(buckets.values()))
        if other > 0.05:
            buckets["untracked"] = other

        sim = buckets.get("sim_take_action", 0.0)
        infer = (
            buckets.get("infer_action", 0.0)
            + buckets.get("infer_reset", 0.0)
            + buckets.get("infer_kv", 0.0)
        )
        env_setup = buckets.get("env_setup", 0.0)
        obs = buckets.get("obs", 0.0)
        post = buckets.get("post_episode", 0.0)

        pct = lambda x: (100.0 * x / ep_total) if ep_total > 0 else 0.0
        line = (
            f"[eval_timing] ep={episode}/{policy_steps}@{step_lim} "
            f"task={self.task_name} seed={seed} succ={success} "
            f"total={_fmt_sec(ep_total)} | "
            f"sim={_fmt_sec(sim)} ({pct(sim):.0f}%) "
            f"infer={_fmt_sec(infer)} ({pct(infer):.0f}%) "
            f"env_setup={_fmt_sec(env_setup)} obs={_fmt_sec(obs)} "
            f"post={_fmt_sec(post)}"
        )
        print(line, flush=True)

        record: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "task": self.task_name,
            "st_seed": self.st_seed,
            "episode": episode,
            "seed": seed,
            "success": success,
            "policy_steps": policy_steps,
            "step_lim": step_lim,
            "total_sec": round(ep_total, 3),
            "buckets_sec": {k: round(v, 3) for k, v in sorted(buckets.items())},
            "buckets_count": dict(self._ep_counts),
            "sim_sec": round(sim, 3),
            "infer_sec": round(infer, 3),
        }
        with open(self._jsonl_path(), "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def finish_run(self, *, episodes: int, successes: int) -> None:
        total = sum(self._run_buckets.values())
        if total <= 0:
            return
        sim = self._run_buckets.get("sim_take_action", 0.0)
        infer = (
            self._run_buckets.get("infer_action", 0.0)
            + self._run_buckets.get("infer_reset", 0.0)
            + self._run_buckets.get("infer_kv", 0.0)
        )
        parts = [
            f"[eval_timing] RUN task={self.task_name} eps={episodes} succ={successes}",
            f"total={_fmt_sec(total)}",
            f"sim={_fmt_sec(sim)} ({100 * sim / total:.0f}%)",
            f"infer={_fmt_sec(infer)} ({100 * infer / total:.0f}%)",
        ]
        for name in ("env_setup", "obs", "post_episode"):
            if name in self._run_buckets:
                v = self._run_buckets[name]
                parts.append(f"{name}={_fmt_sec(v)} ({100 * v / total:.0f}%)")
        print(" | ".join(parts), flush=True)
