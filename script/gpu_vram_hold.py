#!/usr/bin/env python3
import argparse
import signal
import time

import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gib-per-gpu", type=float, default=16.0)
    parser.add_argument("--interval", type=float, default=60.0)
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    running = True

    def stop(_signum, _frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    elements = int(args.gib_per_gpu * 1024**3 / 2)
    buffers = []
    for device_id in range(torch.cuda.device_count()):
        with torch.cuda.device(device_id):
            buffer = torch.empty(elements, dtype=torch.float16, device=device_id)
            buffer.zero_()
            buffers.append(buffer)

    print(
        f"holding {args.gib_per_gpu:.1f} GiB on each of "
        f"{len(buffers)} GPU(s); interval={args.interval:.1f}s",
        flush=True,
    )
    while running:
        for device_id, buffer in enumerate(buffers):
            with torch.cuda.device(device_id):
                buffer[0].add_(1)
        torch.cuda.synchronize()
        print(f"heartbeat={time.strftime('%F %T %Z')}", flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
