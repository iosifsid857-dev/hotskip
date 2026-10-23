#!/usr/bin/env python3
"""LLM lab, step 4: three tiers (VRAM / RAM / SSD) for a model bigger than memory.

For each held-out prompt (leave-one-out, ranking built from the other prompts):
  * VRAM tier: the H globally hottest experts (fixed, like our hot/cold split model).
  * RAM tier, R experts, three policies:
      static  - the next R hottest experts, pinned;
      lru     - plain LRU over the non-VRAM experts (roughly what mmap + page cache does);
      hybrid  - half pinned by popularity, half LRU.
  * everything else is read from SSD on use.
Reports SSD traffic per generated token and the temporal locality of routing
(how often an expert used by token t at layer L was also used by one of the previous k tokens)."""
import collections
import glob
import sys

TRACE_GLOB = sys.argv[1] if len(sys.argv) > 1 else "traces/qwen_*.txt"
EXPERT_MB = float(sys.argv[2]) if len(sys.argv) > 2 else 2.04
HOT = int(sys.argv[3]) if len(sys.argv) > 3 else 1880
RAM_GB = [2, 4, 6, 8, 10, 12]


def load(path):
    """-> list of tokens, each a list of (layer, expert)."""
    toks = collections.OrderedDict()
    for line in open(path):
        p = line.split()
        if len(p) < 3:
            continue
        toks.setdefault(int(p[0]), []).extend((int(p[1]), int(e)) for e in p[2:])
    return list(toks.values())


def simulate(tokens, hot, ram_static, lru_size):
    lru = collections.OrderedDict()
    miss = total = 0
    for tok in tokens:
        for key in tok:
            total += 1
            if key in hot or key in ram_static:
                continue
            if key in lru:
                lru.move_to_end(key)
                continue
            miss += 1
            if lru_size:
                lru[key] = None
                if len(lru) > lru_size:
                    lru.popitem(last=False)
    return miss, total


def main():
    files = sorted(f for f in glob.glob(TRACE_GLOB) if f.rsplit("_", 1)[1].split(".")[0].isdigit()
                   and f.count(".") == 1)
    traces = [load(f) for f in files]
    per_tok = len(traces[0][0])
    print(f"traces: {len(files)}, expert {EXPERT_MB} MB, {per_tok} experts per token "
          f"= {per_tok * EXPERT_MB:.0f} MB of expert weights per token; VRAM tier {HOT} experts\n")

    rows = collections.defaultdict(lambda: [0, 0])
    for i, held in enumerate(traces):
        cnt = collections.Counter(k for j, t in enumerate(traces) if j != i for tok in t for k in tok)
        rank = [k for k, _ in cnt.most_common()]
        hot = set(rank[:HOT])
        for gb in RAM_GB:
            r = int(gb * 1024 / EXPERT_MB)
            for pol, static, lru in (("static", set(rank[HOT:HOT + r]), 0),
                                     ("lru", set(), r),
                                     ("hybrid", set(rank[HOT:HOT + r // 2]), r - r // 2)):
                m, t = simulate(held, hot, static, lru)
                rows[(gb, pol)][0] += m
                rows[(gb, pol)][1] += t

    print("RAM tier |  static: miss  MB/tok |     lru: miss  MB/tok |  hybrid: miss  MB/tok")
    for gb in RAM_GB:
        cells = []
        for pol in ("static", "lru", "hybrid"):
            m, t = rows[(gb, pol)]
            cells.append(f"{m / t:6.1%} {m / t * per_tok * EXPERT_MB:7.1f}")
        print(f"{gb:5d} GB |      " + " |      ".join(cells))

    print("\ntemporal locality (expert of token t at layer L seen in the previous k tokens, same layer):")
    for k in (1, 2, 4, 8, 16, 32):
        hit = tot = 0
        for t in traces:
            for n in range(k, len(t)):
                prev = set().union(*map(set, t[n - k:n]))
                for key in t[n]:
                    tot += 1
                    hit += key in prev
        print(f"  k={k:2d}: {hit / tot:.1%}")


if __name__ == "__main__":
    main()
