#!/usr/bin/env python3
"""Expert store, step 1: from a hot/cold split GGUF build
  * a skeleton GGUF where every cold expert stack (blk.N.ffn_*_cexps.*) has only S zeroed slots -
    the RAM budget; llama.cpp fills the slots at run time from the split file (EXPSTORE_INDEX);
  * an index file for llama.cpp: per layer, where each cold expert lives in the split file,
    the expert id -> cold-stack index map, and the cold experts ordered by popularity (pinning).

Usage: make_store.py split.gguf skeleton.gguf index.txt slots_per_layer "traces/qwen_*.txt"
"""
import collections
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.expanduser("~/Projects/src/llama.cpp/gguf-py"))
import gguf  # noqa: E402

PARTS = [(k, w) for w in ("weight", "bias") for k in ("gate", "up", "down")]  # arg order in llama.cpp


def load_counts(pattern):
    cnt = collections.Counter()
    for f in glob.glob(pattern):
        if os.path.basename(f).count(".") != 1:
            continue
        for line in open(f):
            p = line.split()
            if len(p) >= 3:
                for e in p[2:]:
                    cnt[(int(p[1]), int(e))] += 1
    return cnt


def main():
    src, dst, index, slots, pattern = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
    reader = gguf.GGUFReader(src)
    arch = bytes(reader.fields["general.architecture"].parts[-1]).decode()
    n_layer = int(reader.fields[f"{arch}.block_count"].parts[-1][0])
    n_expert = int(reader.fields[f"{arch}.expert_count"].parts[-1][0])
    tensors = {t.name: t for t in reader.tensors}
    counts = load_counts(pattern)

    with open(index, "w") as f:
        f.write(f"expstore 1 {os.path.abspath(src)} {n_layer} {n_expert} {slots}\n")
        for il in range(n_layer):
            parts = [tensors[n] for n in (f"blk.{il}.ffn_{k}_cexps.{w}" for k, w in PARTS) if n in tensors]
            cmap = tensors[f"blk.{il}.ffn_moe_cmap.weight"].data.astype(np.int64)
            n_cold = parts[0].data.shape[0] - 1  # last slot of the split stack is the zero expert
            cmap = np.where((cmap < 0) | (cmap >= n_cold), -1, cmap)  # hot experts -> -1
            cold = [e for e in range(n_expert) if cmap[e] >= 0]
            rank = sorted(cold, key=lambda e: -counts[(il, e)])
            f.write(f"layer {il} {len(parts)}\n")
            for t in parts:
                f.write(f"part {t.name} {t.data_offset} {t.n_bytes // (n_cold + 1)}\n")
            f.write("cmap " + " ".join(map(str, cmap)) + "\n")
            f.write("rank " + " ".join(map(str, rank)) + "\n")

    writer = gguf.GGUFWriter(dst, arch=arch)
    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        vt = field.types[0]
        writer.add_key_value(field.name, field.contents(), vt,
                             sub_type=field.types[-1] if vt == gguf.GGUFValueType.ARRAY else None)
    writer.add_key_value("expstore.slots", slots, gguf.GGUFValueType.UINT32)
    jobs = []
    for t in reader.tensors:
        if "_cexps." in t.name:
            shape = (slots,) + t.data.shape[1:]
            writer.add_tensor_info(t.name, shape, t.data.dtype, int(np.prod(shape)) * t.data.dtype.itemsize,
                                   t.tensor_type)
            jobs.append(("zeros", shape, t.data.dtype))
        else:
            writer.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
            jobs.append(("copy", t))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for job in jobs:
        if job[0] == "copy":
            writer.write_tensor_data(job[1].data, tensor_endianess=reader.endianess)
        else:
            writer.write_tensor_data(np.zeros(job[1], dtype=job[2]))
    writer.close()
    print(f"skeleton {dst}: {os.path.getsize(dst) / 1e9:.2f} GB, index {index}")


if __name__ == "__main__":
    main()
