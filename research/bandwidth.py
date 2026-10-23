#!/usr/bin/env python3
"""LLM lab, step 1: how fast can this laptop move weights around?
RAM -> RAM, RAM -> VRAM (pageable and pinned), VRAM -> VRAM. Prints GB/s."""
import time

import torch

GB = 1024 ** 3


def bench(fn, nbytes, reps=5):
    fn()  # warm-up
    torch.cuda.synchronize()
    best = 0.0
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        best = max(best, nbytes / (time.perf_counter() - t) / GB)
    return best


def main():
    torch.set_num_threads(torch.get_num_threads())
    n = 1 * GB
    cpu_a = torch.empty(n, dtype=torch.uint8)
    cpu_b = torch.empty(n, dtype=torch.uint8)
    pinned = torch.empty(n, dtype=torch.uint8).pin_memory()
    gpu_a = torch.empty(n, dtype=torch.uint8, device="cuda")
    gpu_b = torch.empty(n, dtype=torch.uint8, device="cuda")
    cpu_a.fill_(1)
    pinned.fill_(1)

    results = {
        "RAM -> RAM (copy, read+write)": bench(lambda: cpu_b.copy_(cpu_a), n * 2),
        "RAM -> VRAM (pageable)": bench(lambda: gpu_a.copy_(cpu_a), n),
        "RAM -> VRAM (pinned)": bench(lambda: gpu_a.copy_(pinned, non_blocking=True), n),
        "VRAM -> RAM (pinned)": bench(lambda: pinned.copy_(gpu_a, non_blocking=True), n),
        "VRAM -> VRAM (copy, read+write)": bench(lambda: gpu_b.copy_(gpu_a), n * 2),
    }
    free, total = torch.cuda.mem_get_info()
    print(f"GPU: {torch.cuda.get_device_name(0)}  VRAM free {free / GB:.1f} / {total / GB:.1f} GB")
    for k, v in results.items():
        print(f"{k:34s} {v:7.1f} GB/s")


if __name__ == "__main__":
    main()
