#!/usr/bin/env python3
"""In a hot/cold split GGUF, point cold-map entries of hot experts to -1 (CPU skips them)
instead of the zero expert. Edits the small map tensors in place."""
import os
import sys

sys.path.insert(0, os.path.expanduser("~/Projects/src/llama.cpp/gguf-py"))
import gguf  # noqa: E402

r = gguf.GGUFReader(sys.argv[1], "r+")
n = 0
for t in r.tensors:
    if t.name.endswith("ffn_moe_cmap.weight"):
        zero_slot = int(t.data.max())  # hot experts point at the zero slot = len(cold)
        mask = t.data == zero_slot
        t.data[mask] = -1
        n += int(mask.sum())
print("cold-map entries set to -1:", n)
