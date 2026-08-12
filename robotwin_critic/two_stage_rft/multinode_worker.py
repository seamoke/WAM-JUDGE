"""Shared-filesystem command queue for the second online-RFT GPU node."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any


JOB_ID = re.compile(r"^[A-Za-z0-9_.-]+$")
SENSITIVE_ENV_FRAGMENTS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def exported_environment() -> dict[str, str]:
    """Forward runtime settings without persisting credentials in the queue."""
    return {
        name: value
        for name, value in os.environ.items()
        if not any(fragment in name.upper() for fragment in SENSITIVE_ENV_FRAGMENTS)
    }


def job_paths(queue_root: Path, job_id: str) -> tuple[Path, Path, Path, Path]:
    if not JOB_ID.fullmatch(job_id):
        raise ValueError(f"Unsafe job id: {job_id!r}")
    return (
        queue_root / "jobs" / f"{job_id}.json",
        queue_root / "started" / f"{job_id}.json",
        queue_root / "results" / f"{job_id}.json",
        queue_root / "logs" / f"{job_id}.log",
    )


def submit_job(queue_root: Path, job_id: str, command: list[str], cwd: Path) -> Path:
    job_path, started_path, result_path, _ = job_paths(queue_root, job_id)
    if job_path.exists() or started_path.exists() or result_path.exists():
        raise FileExistsError(f"Multinode job already exists: {job_id}")
    payload = {
        "schema_version": 1,
        "job_id": job_id,
        "command": command,
        "cwd": str(cwd.resolve()),
        "environment": exported_environment(),
        "submitted_at": time.time(),
        "submitter_pid": os.getpid(),
    }
    atomic_write_json(job_path, payload)
    return job_path


class HolderLease:
    """Renew a short holder pause; expiry restores protection after hard failure."""

    def __init__(
        self,
        control_file: Path,
        *,
        lease_seconds: int = 120,
        refresh_seconds: int = 30,
        settle_seconds: int = 6,
    ) -> None:
        if lease_seconds <= refresh_seconds:
            raise ValueError("lease_seconds must exceed refresh_seconds")
        self.control_file = control_file
        self.lease_seconds = lease_seconds
        self.refresh_seconds = refresh_seconds
        self.settle_seconds = settle_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _renew(self) -> None:
        while not self._stop.is_set():
            atomic_write_text(
                self.control_file,
                f"{int(time.time()) + self.lease_seconds}\n",
            )
            self._stop.wait(self.refresh_seconds)

    def __enter__(self) -> "HolderLease":
        self._renew_once()
        self._thread = threading.Thread(target=self._renew, daemon=True)
        self._thread.start()
        time.sleep(self.settle_seconds)
        return self

    def _renew_once(self) -> None:
        atomic_write_text(
            self.control_file,
            f"{int(time.time()) + self.lease_seconds}\n",
        )

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        del exc_type, exc_value, traceback
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.refresh_seconds + 5)
        atomic_write_text(self.control_file, "0\n")


def run_job(
    queue_root: Path,
    job: dict[str, Any],
    control_file: Path,
    *,
    lease_seconds: int = 120,
    refresh_seconds: int = 30,
    settle_seconds: int = 6,
    stop_event: threading.Event | None = None,
) -> dict[str, Any]:
    job_id = str(job["job_id"])
    _, started_path, result_path, log_path = job_paths(queue_root, job_id)
    command = [str(value) for value in job["command"]]
    environment = os.environ.copy()
    environment.update({str(k): str(v) for k, v in job["environment"].items()})
    started = {
        "job_id": job_id,
        "worker_pid": os.getpid(),
        "started_at": time.time(),
        "command": command,
    }
    atomic_write_json(started_path, started)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cancelled = False
    with log_path.open("a", encoding="utf-8") as log_handle:
        with HolderLease(
            control_file,
            lease_seconds=lease_seconds,
            refresh_seconds=refresh_seconds,
            settle_seconds=settle_seconds,
        ):
            process = subprocess.Popen(
                command,
                cwd=job["cwd"],
                env=environment,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            cancel_path = queue_root / "cancel" / job_id
            while process.poll() is None:
                should_stop = (queue_root / "STOP").exists() or cancel_path.exists()
                if stop_event is not None and stop_event.is_set():
                    should_stop = True
                if should_stop:
                    cancelled = True
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    break
                time.sleep(1)
            return_code = process.wait()
    result = {
        "job_id": job_id,
        "return_code": return_code,
        "cancelled": cancelled,
        "started_at": started["started_at"],
        "finished_at": time.time(),
        "log": str(log_path),
    }
    atomic_write_json(result_path, result)
    return result


def serve(args: argparse.Namespace) -> None:
    args.queue_root.mkdir(parents=True, exist_ok=True)
    lock_path = args.queue_root / ".worker.lock"
    lock_handle = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError(f"A multinode worker already owns {args.queue_root}") from error
    lock_handle.seek(0)
    lock_handle.truncate()
    lock_handle.write(f"pid={os.getpid()} host={os.uname().nodename}\n")
    lock_handle.flush()
    print(f"MULTINODE_WORKER_READY queue={args.queue_root}", flush=True)
    stop_event = threading.Event()

    def request_stop(signum, frame) -> None:
        del signum, frame
        stop_event.set()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    processed = 0
    while not stop_event.is_set() and not (args.queue_root / "STOP").exists():
        for job_path in sorted((args.queue_root / "jobs").glob("*.json")):
            job = json.loads(job_path.read_text(encoding="utf-8"))
            _, _, result_path, _ = job_paths(args.queue_root, str(job["job_id"]))
            if result_path.exists():
                continue
            result = run_job(
                args.queue_root,
                job,
                args.control_file,
                lease_seconds=args.lease_seconds,
                refresh_seconds=args.refresh_seconds,
                settle_seconds=args.settle_seconds,
                stop_event=stop_event,
            )
            print(
                f"MULTINODE_WORKER_JOB_DONE id={result['job_id']} "
                f"rc={result['return_code']}",
                flush=True,
            )
            processed += 1
            if args.once and processed >= 1:
                return
        time.sleep(args.poll_seconds)


def wait_for_result(queue_root: Path, job_id: str, timeout: float) -> dict[str, Any]:
    _, _, result_path, _ = job_paths(queue_root, job_id)
    deadline = time.monotonic() + timeout if timeout > 0 else None
    while not result_path.is_file():
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for multinode job {job_id}")
        time.sleep(1)
    return json.loads(result_path.read_text(encoding="utf-8"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    submit = subparsers.add_parser("submit")
    submit.add_argument("--queue-root", type=Path, required=True)
    submit.add_argument("--job-id", required=True)
    submit.add_argument("--cwd", type=Path, required=True)
    submit.add_argument("command", nargs=argparse.REMAINDER)

    wait = subparsers.add_parser("wait")
    wait.add_argument("--queue-root", type=Path, required=True)
    wait.add_argument("--job-id", required=True)
    wait.add_argument("--timeout", type=float, default=0)

    worker = subparsers.add_parser("serve")
    worker.add_argument("--queue-root", type=Path, required=True)
    worker.add_argument("--control-file", type=Path, required=True)
    worker.add_argument("--poll-seconds", type=float, default=1)
    worker.add_argument("--lease-seconds", type=int, default=120)
    worker.add_argument("--refresh-seconds", type=int, default=30)
    worker.add_argument("--settle-seconds", type=int, default=6)
    worker.add_argument("--once", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.mode == "submit":
        command = args.command[1:] if args.command[:1] == ["--"] else args.command
        if not command:
            parser.error("submit requires a command after --")
        path = submit_job(args.queue_root, args.job_id, command, args.cwd)
        print(f"MULTINODE_JOB_SUBMITTED id={args.job_id} path={path}")
        return
    if args.mode == "wait":
        result = wait_for_result(args.queue_root, args.job_id, args.timeout)
        print(json.dumps(result, sort_keys=True))
        raise SystemExit(int(result["return_code"]))
    serve(args)


if __name__ == "__main__":
    main()
