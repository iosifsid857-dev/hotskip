#!/usr/bin/env python3
"""Rewrite a MoE GGUF so that every layer has two expert stacks:
  blk.N.ffn_{gate,up,down}_hexps.weight  - hot experts (meant for the GPU) + 1 zero expert
  blk.N.ffn_{gate,up,down}_cexps.weight  - cold experts (stay in RAM)     + 1 zero expert
  blk.N.ffn_moe_hmap.weight / ffn_moe_cmap.weight (I32[n_expert]) - expert id -> slot in that stack,
  or the zero slot if the expert lives in the other stack.
Per-expert biases (gpt-oss: blk.N.ffn_{gate,up,down}_exps.bias) are split the same way, zero row for the zero slot.
Hot experts are the globally most used ones in the routing traces, up to a byte budget.

Usage: split_hot_experts.py in.gguf out.gguf budget_gib "traces/qwen_*.txt"
"""
import collections
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.expanduser("~/Projects/src/llama.cpp/gguf-py"))
import gguf  # noqa: E402

KINDS = ("gate", "up", "down")


def load_counts(pattern):
    cnt = collections.Counter()
    for f in glob.glob(pattern):
        base = os.path.basename(f)
        if base.count(".") != 1:  # skip *.answer.txt / *.log
            continue
        for line in open(f):
            p = line.split()
            if len(p) >= 3:
                layer = int(p[1])
                for e in p[2:]:
                    cnt[(layer, int(e))] += 1
    return cnt


def main():
    src, dst, budget_gib, pattern = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
    reader = gguf.GGUFReader(src)
    arch = bytes(reader.fields["general.architecture"].parts[-1]).decode()
    n_layer = int(reader.fields[f"{arch}.block_count"].parts[-1][0])
    n_expert = int(reader.fields[f"{arch}.expert_count"].parts[-1][0])
    tensors = {t.name: t for t in reader.tensors}

    # bytes of one expert (gate + up + down) per layer
    def exps(il, kind):
        return tensors[f"blk.{il}.ffn_{kind}_exps.weight"]

    expert_bytes = {il: sum(exps(il, k).n_bytes for k in KINDS) // n_expert for il in range(n_layer)}

    counts = load_counts(pattern)
    budget = budget_gib * 1024 ** 3
    hot = collections.defaultdict(list)
    used = 0
    for (il, e), _ in counts.most_common():
        # one zero expert per layer also lives on the GPU
        extra = expert_bytes[il] * (2 if not hot[il] else 1)
        if used + extra > budget:
            continue
        hot[il].append(e)
        used += extra
    n_hot = sum(len(v) for v in hot.values())
    print(f"{arch}: {n_layer} layers x {n_expert} experts; hot experts: {n_hot} ({used / 1024 ** 3:.2f} GiB)")

    writer = gguf.GGUFWriter(dst, arch=arch)
    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        vt = field.types[0]
        sub = field.types[-1] if vt == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), vt, sub_type=sub)
    writer.add_key_value("hotexperts.budget_gib", budget_gib, gguf.GGUFValueType.FLOAT32)

    plan = []  # (name, kind of data producer)
    split_names = {f"blk.{il}.ffn_{k}_exps.{w}" for il in range(n_layer) for k in KINDS for w in ("weight", "bias")}
    for t in reader.tensors:
        if t.name not in split_names:
            plan.append((t.name, ("copy", t)))
            writer.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
            continue
        il = int(t.name.split(".")[1])
        kind = t.name.split(".")[2].split("_")[1]
        suffix = t.name.split(".")[3]  # weight / bias
        h = sorted(hot[il])
        c = [e for e in range(n_expert) if e not in set(h)]
        for tag, ids in (("h", h), ("c", c)):
            shape = (len(ids) + 1,) + t.data.shape[1:]
            nbytes = int(np.prod(shape)) * t.data.dtype.itemsize
            name = f"blk.{il}.ffn_{kind}_{tag}exps.{suffix}"
            writer.add_tensor_info(name, shape, t.data.dtype, nbytes, t.tensor_type)
            plan.append((name, ("experts", t, ids)))
        if kind == "down" and suffix == "weight":  # maps once per layer
            for tag, ids in (("h", h), ("c", c)):
                m = np.full(n_expert, len(ids), dtype=np.int32)  # default: zero slot
                for slot, e in enumerate(ids):
                    m[e] = slot
                name = f"blk.{il}.ffn_moe_{tag}map.weight"
                writer.add_tensor_info(name, m.shape, m.dtype, m.nbytes)
                plan.append((name, ("array", m)))

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for i, (name, job) in enumerate(plan):
        if job[0] == "copy":
            writer.write_tensor_data(job[1].data, tensor_endianess=reader.endianess)
        elif job[0] == "experts":
            t, ids = job[1], job[2]
            data = t.data
            out = np.empty((len(ids) + 1,) + data.shape[1:], dtype=data.dtype)
            if ids:
                out[:-1] = data[ids]
            out[-1] = 0  # zero expert: all-zero quant blocks decode to 0.0
            writer.write_tensor_data(out)
        else:
            writer.write_tensor_data(job[1])
        if i % 50 == 0:
            print(f"  {i}/{len(plan)} {name}", flush=True)
    writer.close()
    print("done", dst)


if __name__ == "__main__":
    main()
