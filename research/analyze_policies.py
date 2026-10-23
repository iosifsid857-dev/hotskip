#!/usr/bin/env python3
"""HOTskip: which RAM-slot policy misses least? Replays the routing traces (leave-one-prompt-out:
popularity from the other prompts) through one layer-local cache of S slots per layer, VRAM tier fixed.

  pinned+lru  - current HOTskip: P most popular pinned, the rest LRU
  lru         - plain LRU
  lfu-decay   - evict the slot with the lowest decayed use count (count *= DECAY per call of the layer),
                seeded with global popularity, so it blends "popular" and "recent" without a fixed split
  arc-ish     - two LRU lists: seen once / seen again; evict from the 'once' list first
Prompts are replayed back to back (as one long chat) so caches warm up like in real use.
Usage: analyze_policies.py "traces/qwen_*.txt" hot_experts slots"""
import collections
import glob
import sys

TRACE_GLOB = sys.argv[1] if len(sys.argv) > 1 else "traces/qwen_*.txt"
HOT = int(sys.argv[2]) if len(sys.argv) > 2 else 1880
SLOTS = int(sys.argv[3]) if len(sys.argv) > 3 else 80


def load(path):
    calls = []  # (layer, [experts]) in execution order
    for line in open(path):
        p = line.split()
        if len(p) >= 3:
            calls.append((int(p[1]), [int(e) for e in p[2:]]))
    return calls


class PinnedLRU:
    def __init__(self, rank, pinned):
        self.pin = set(rank[:pinned])
        self.cap = SLOTS - pinned
        self.lru = collections.OrderedDict()

    def access(self, e):
        if e in self.pin:
            return True
        if e in self.lru:
            self.lru.move_to_end(e)
            return True
        self.lru[e] = None
        if len(self.lru) > self.cap:
            self.lru.popitem(last=False)
        return False


class LFUDecay:
    def __init__(self, rank, decay, prior):
        self.decay, self.score, self.res = decay, {}, set()
        for i, e in enumerate(rank):  # popularity prior, strongest for the most used
            self.score[e] = prior * (len(rank) - i) / len(rank)
        for e in rank[:SLOTS]:
            self.res.add(e)

    def tick(self):
        for e in self.res:
            self.score[e] *= self.decay

    def access(self, e):
        self.score[e] = self.score.get(e, 0) + 1
        if e in self.res:
            return True
        if len(self.res) >= SLOTS:
            self.res.remove(min(self.res, key=self.score.get))
        self.res.add(e)
        return False


class ARCish:
    def __init__(self, rank):
        self.once, self.again = collections.OrderedDict(), collections.OrderedDict((e, None) for e in rank[:SLOTS])

    def access(self, e):
        for d in (self.again, self.once):
            if e in d:
                del d[e]
                self.again[e] = None
                return True
        self.once[e] = None
        if len(self.once) + len(self.again) > SLOTS:
            (self.once if len(self.once) > SLOTS // 4 or not self.again else self.again).popitem(last=False)
        return False


def main():
    files = sorted(f for f in glob.glob(TRACE_GLOB) if f.rsplit("_", 1)[1].split(".")[0].isdigit()
                   and f.count(".") == 1)
    traces = [load(f) for f in files]
    policies = {f"pinned{p}+lru": (lambda r, p=p: PinnedLRU(r, p)) for p in (0, SLOTS // 4, SLOTS // 2, 3 * SLOTS // 4)}
    for d in (0.9, 0.97, 0.99):
        for prior in (1, 5):
            policies[f"lfu d={d} prior={prior}"] = lambda r, d=d, prior=prior: LFUDecay(r, d, prior)
    policies["arc-ish"] = ARCish

    res = collections.defaultdict(lambda: [0, 0])
    for i in range(len(traces)):
        cnt = collections.Counter((l, e) for j, t in enumerate(traces) if j != i for l, es in t for e in es)
        ranked = [k for k, _ in cnt.most_common()]
        hot = set(ranked[:HOT])
        layers = {l for t in traces for l, _ in t}
        rank = {l: [e for (ll, e) in ranked[HOT:] if ll == l] for l in layers}
        # replay: held-out prompt last, after the others, like a long conversation ending on new topic
        order = [t for j, t in enumerate(traces) if j != i] + [traces[i]]
        for name, make in policies.items():
            caches = {l: make(rank[l]) for l in layers}
            for k, t in enumerate(order):
                last = k == len(order) - 1
                for l, es in t:
                    c = caches[l]
                    if hasattr(c, "tick"):
                        c.tick()
                    for e in es:
                        if (l, e) in hot:
                            continue
                        hit = c.access(e)
                        if last:
                            res[name][0] += not hit
                            res[name][1] += 1
    print(f"{len(files)} traces, VRAM {HOT} experts, {SLOTS} RAM slots/layer; miss rate of cold accesses "
          f"on the held-out prompt:")
    for name, (m, t) in sorted(res.items(), key=lambda kv: kv[1][0] / kv[1][1]):
        print(f"  {name:24s} {m / t:6.1%}")


if __name__ == "__main__":
    main()
