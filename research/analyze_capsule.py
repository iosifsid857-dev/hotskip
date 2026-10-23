#!/usr/bin/env python3
"""LLM lab: Joe's 'per-conversation capsule' vs our global hot set.
For each held-out prompt: look at the experts used in its first W tokens, build a capsule
(those experts first, the rest of the budget filled from the global ranking of the other prompts),
and score the GPU share on the tokens after W. Same VRAM budget for every method.
Oracle = best possible fixed set for that prompt (its own most used experts) - an upper bound."""
import collections
import glob
import sys

PATTERN = sys.argv[1] if len(sys.argv) > 1 else "traces/qwen_*.txt"
BUDGET = int(sys.argv[2]) if len(sys.argv) > 2 else 1880  # hot experts that fit (= our 3.4 GiB file)


def load(path):
    rows = []  # (token, layer, expert)
    for line in open(path):
        p = line.split()
        if len(p) >= 3:
            t, l = int(p[0]), int(p[1])
            rows += [(t, l, int(e)) for e in p[2:]]
    return rows


files = sorted((f for f in glob.glob(PATTERN) if f.rsplit("_", 1)[1].split(".")[0].isdigit() and f.count(".") == 1),
               key=lambda f: int(f.rsplit("_", 1)[1].split(".")[0]))
traces = [load(f) for f in files]


def share(hot, rows):
    return sum(1 for _, l, e in rows if (l, e) in hot) / max(len(rows), 1)


print(f"budget {BUDGET} experts, {len(traces)} prompts\n")
print("window = tokens seen before the capsule is built (prompt tokens come first)")
results = collections.defaultdict(list)
for i, held in enumerate(traces):
    other = collections.Counter((l, e) for j, t in enumerate(traces) if j != i for _, l, e in t)
    global_rank = [a for a, _ in other.most_common()]
    global_hot = set(global_rank[:BUDGET])
    first_tok = min(t for t, _, _ in held)
    for w in (16, 32, 64):
        cut = first_tok + w
        seen = collections.Counter((l, e) for t, l, e in held if t < cut)
        rest = [r for r in held if r[0] >= cut]
        capsule = [a for a, _ in seen.most_common()][:BUDGET]
        cset = set(capsule)
        for a in global_rank:
            if len(cset) >= BUDGET:
                break
            cset.add(a)
        oracle = set(a for a, _ in collections.Counter((l, e) for _, l, e in rest).most_common(BUDGET))
        results[(w, "global")].append(share(global_hot, rest))
        results[(w, "capsule")].append(share(cset, rest))
        results[(w, "oracle")].append(share(oracle, rest))

for w in (16, 32, 64):
    g = sum(results[(w, "global")]) / len(traces)
    c = sum(results[(w, "capsule")]) / len(traces)
    o = sum(results[(w, "oracle")]) / len(traces)
    print(f"W={w:3d}: global hot {g:6.1%} | capsule {c:6.1%} | oracle {o:6.1%}")
