#!/usr/bin/env python3
"""LLM lab, step 2: are some MoE experts 'hot'? And would caching the hot ones in VRAM beat
llama.cpp's --n-cpu-moe (which keeps whole layers on the GPU)?

Trace format (from llama-moe-trace): "<token> <layer> <expert ids...>" per line.
Evaluation is leave-one-prompt-out: the cache is chosen from the other prompts and scored on the
held-out one, so we measure what would happen on a new, unseen request."""
import collections
import glob
import sys

TRACE_GLOB = sys.argv[1] if len(sys.argv) > 1 else "traces/qwen_*.txt"
GPU_LAYERS = int(sys.argv[2]) if len(sys.argv) > 2 else 7  # whole layers that fit (measured: --n-cpu-moe 33)


def load(path):
    acts = []  # list of (layer, expert)
    for line in open(path):
        parts = line.split()
        if len(parts) < 3:
            continue
        layer = int(parts[1])
        acts += [(layer, int(e)) for e in parts[2:]]
    return acts


def main():
    files = sorted((f for f in glob.glob(TRACE_GLOB) if f.rsplit("_", 1)[1].split(".")[0].isdigit() and f.count(".") == 1),
                   key=lambda f: int(f.rsplit("_", 1)[1].split(".")[0]))
    traces = [load(f) for f in files]
    allacts = [a for t in traces for a in t]
    layers = sorted({l for l, _ in allacts})
    n_layers = len(layers)
    n_expert = max(e for _, e in allacts) + 1
    per_token_per_layer = len(open(files[0]).readline().split()) - 2
    budget = GPU_LAYERS * n_expert  # same VRAM as GPU_LAYERS full layers of experts

    print(f"traces: {len(files)}  activations: {len(allacts)}")
    print(f"layers: {n_layers}  experts/layer: {n_expert}  experts used per token per layer: {per_token_per_layer}")
    print(f"VRAM budget = {GPU_LAYERS} full layers = {budget} experts ({budget / (n_layers * n_expert):.1%} of all)\n")

    # how skewed is usage? share of activations covered by the top X% of (layer, expert) pairs
    cnt = collections.Counter(allacts)
    total = sum(cnt.values())
    ranked = [c for _, c in cnt.most_common()]
    all_pairs = n_layers * n_expert
    print("skew (all prompts): top share of experts -> share of all work")
    for frac in (0.05, 0.10, 0.175, 0.25, 0.50):
        k = int(all_pairs * frac)
        print(f"  top {frac:5.1%} experts -> {sum(ranked[:k]) / total:6.1%} of activations")
    never = all_pairs - len(cnt)
    print(f"  experts never used at all: {never} of {all_pairs} ({never / all_pairs:.1%})\n")

    # leave-one-prompt-out: whole-layer cache (llama.cpp style) vs hot-expert cache
    rows = []
    for i, held in enumerate(traces):
        train = collections.Counter(a for j, t in enumerate(traces) if j != i for a in t)
        hot = {a for a, _ in train.most_common(budget)}
        first_layers = set(layers[:GPU_LAYERS])
        n = len(held)
        hit_layers = sum(1 for l, _ in held if l in first_layers) / n
        hit_hot = sum(1 for a in held if a in hot) / n
        rows.append((i + 1, hit_layers, hit_hot))
    print("held-out prompt | GPU share: whole layers (llama.cpp) | GPU share: hot experts (ours)")
    for p, a, b in rows:
        print(f"  prompt {p}       | {a:6.1%}                           | {b:6.1%}")
    avg_a = sum(r[1] for r in rows) / len(rows)
    avg_b = sum(r[2] for r in rows) / len(rows)
    print(f"  AVERAGE        | {avg_a:6.1%}                           | {avg_b:6.1%}")

    # crude speed model: time per token ~ share of expert work left on the CPU (CPU-bound, measured)
    # anchor: 0 GPU layers -> 28.5 tok/s, 7 GPU layers -> 34.7 tok/s (measured with llama-server)
    t0, t7 = 1 / 28.5, 1 / 34.7
    cpu_part = (t0 - t7) / (7 / n_layers)  # seconds per token for 100% of expert work on the CPU
    fixed = t0 - cpu_part
    for name, share in (("whole layers", avg_a), ("hot experts", avg_b)):
        print(f"  predicted speed with {name:12s}: {1 / (fixed + cpu_part * (1 - share)):5.1f} tok/s")


if __name__ == "__main__":
    main()
