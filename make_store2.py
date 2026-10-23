#!/usr/bin/env python3
"""Expert store from the ORIGINAL MoE GGUF (no 60+ GB split copy needed).

Writes a skeleton GGUF with, per layer:
  blk.N.ffn_{gate,up,down}_hexps.{weight,bias} - hot experts (for the GPU) + 1 zero expert, copied;
  blk.N.ffn_{gate,up,down}_cexps.{weight,bias} - S zeroed RAM slots, filled at run time from the original file;
  blk.N.ffn_moe_{h,c}map.weight                - expert id -> stack index (zero slot / -1 when in the other stack);
and an index for llama.cpp (EXPSTORE_INDEX) pointing at the original ffn_*_exps tensors.
Hot experts: globally most used in the routing traces, up to budget_gib.

Usage: make_store2.py model.gguf skeleton.gguf index.txt budget_gib slots "traces/<model>_*.txt"
"""
import collections
import glob
import os
import sys

import numpy as np

for _p in (os.path.join(os.path.dirname(os.path.realpath(__file__)), "llama.cpp", "gguf-py"),
           os.path.expanduser("~/Projects/src/llama.cpp/gguf-py")):
    if os.path.isdir(_p):
        sys.path.insert(0, _p)
        break
import gguf  # noqa: E402

KINDS = ("gate", "up", "down")
PARTS = [(k, w) for w in ("weight", "bias") for k in KINDS]  # arg order in llama.cpp


def load_counts(pattern):
    cnt = collections.Counter()
    for f in glob.glob(pattern):
        if os.path.basename(f).count(".") != 1:  # skip *.answer.txt / *.log
            continue
        for line in open(f):
            p = line.split()
            if len(p) >= 3:
                for e in p[2:]:
                    cnt[(int(p[1]), int(e))] += 1
    return cnt


def write_hole(writer, nbytes):
    """Write an all-zero tensor as a filesystem hole: the RAM slots take no disk space (sparse file)."""
    fout = writer.fout[0]
    name = next(iter(writer.tensors[0]))
    assert writer.tensors[0].pop(name).nbytes == nbytes
    writer.write_padding(fout, fout.tell())
    fout.flush()
    fout.seek(nbytes, os.SEEK_CUR)
    writer.write_padding(fout, nbytes)


def main():
    src, dst, index = sys.argv[1], sys.argv[2], sys.argv[3]
    budget, slots, pattern = float(sys.argv[4]) * 1024 ** 3, int(sys.argv[5]), sys.argv[6]
    reader = gguf.GGUFReader(src)
    arch = bytes(reader.fields["general.architecture"].parts[-1]).decode()
    n_layer = int(reader.fields[f"{arch}.block_count"].parts[-1][0])
    n_expert = int(reader.fields[f"{arch}.expert_count"].parts[-1][0])
    n_used = int(reader.fields[f"{arch}.expert_used_count"].parts[-1][0])
    tensors = {t.name: t for t in reader.tensors}
    moe_layers = [il for il in range(n_layer) if f"blk.{il}.ffn_down_exps.weight" in tensors
                  and f"blk.{il}.nextn.eh_proj.weight" not in tensors]  # MTP layers stay whole
    expert_bytes = {il: sum(tensors[f"blk.{il}.ffn_{k}_exps.weight"].n_bytes for k in KINDS) // n_expert
                    for il in moe_layers}

    counts = load_counts(pattern)
    if not counts:
        sys.exit(f"no traces match {pattern}")
    hot = collections.defaultdict(list)
    used = 0
    # every layer gets at least n_used real hot experts: big batches fill a token's cold positions with
    # distinct hot slots it does not use (CUDA batched mul_mat_id must not see an expert twice per token)
    for il in moe_layers:
        top = sorted(range(n_expert), key=lambda e: -counts[(il, e)])[:n_used]
        hot[il] = top
        used += expert_bytes[il] * (n_used + 1)
    for (il, e), _ in counts.most_common():
        if il not in expert_bytes or e in hot[il]:
            continue
        extra = expert_bytes[il] * (2 if not hot[il] else 1)  # + the zero expert of the hot stack
        if used + extra <= budget:
            hot[il].append(e)
            used += extra
    print(f"{arch}: {len(moe_layers)} MoE layers x {n_expert} experts; hot {sum(map(len, hot.values()))} "
          f"({used / 1024 ** 3:.2f} GiB); RAM slots {slots}/layer = "
          f"{sum(expert_bytes.values()) * slots / 1024 ** 3:.2f} GiB")

    with open(index, "w") as f:
        f.write(f"expstore 1 {os.path.abspath(src)} {n_layer} {n_expert} {slots}\n")
        for il in moe_layers:
            h = set(hot[il])
            parts = [tensors[n] for n in (f"blk.{il}.ffn_{k}_exps.{w}" for k, w in PARTS) if n in tensors]
            cold = [e for e in range(n_expert) if e not in h]
            f.write(f"layer {il} {len(parts)}\n")
            for t in parts:
                f.write(f"part {t.name} {t.data_offset} {t.n_bytes // n_expert}\n")
            f.write("cmap " + " ".join(str(-1 if e in h else e) for e in range(n_expert)) + "\n")
            f.write("rank " + " ".join(map(str, sorted(cold, key=lambda e: -counts[(il, e)]))) + "\n")

    writer = gguf.GGUFWriter(dst, arch=arch)
    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        vt = field.types[0]
        writer.add_key_value(field.name, field.contents(), vt,
                             sub_type=field.types[-1] if vt == gguf.GGUFValueType.ARRAY else None)
    writer.add_key_value("expstore.slots", slots, gguf.GGUFValueType.UINT32)

    jobs = []

    def add(name, shape, dtype, ttype, job):
        nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
        writer.add_tensor_info(name, shape, np.dtype(dtype), nbytes, ttype)
        jobs.append(job)

    split_names = {f"blk.{il}.ffn_{k}_exps.{w}" for il in moe_layers for k, w in PARTS}
    for t in reader.tensors:
        if t.name not in split_names:
            add(t.name, t.data.shape, t.data.dtype, t.tensor_type, ("copy", t))
            continue
        il, kind, suffix = int(t.name.split(".")[1]), t.name.split(".")[2].split("_")[1], t.name.split(".")[3]
        h = sorted(hot[il])
        add(f"blk.{il}.ffn_{kind}_hexps.{suffix}", (len(h) + 1,) + t.data.shape[1:], t.data.dtype, t.tensor_type,
            ("hot", t, h))
        add(f"blk.{il}.ffn_{kind}_cexps.{suffix}", (slots,) + t.data.shape[1:], t.data.dtype, t.tensor_type,
            ("zeros", (slots,) + t.data.shape[1:], t.data.dtype))
        if kind == "down" and suffix == "weight":
            hm = np.full(n_expert, len(h), dtype=np.int32)  # default: zero slot of the hot stack
            for slot, e in enumerate(h):
                hm[e] = slot
            cm = np.where(np.isin(np.arange(n_expert), h), -1, 0).astype(np.int32)  # unused by the store
            for tag, m in (("h", hm), ("c", cm)):
                add(f"blk.{il}.ffn_moe_{tag}map.weight", m.shape, m.dtype, None, ("array", m))

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for job in jobs:
        if job[0] == "copy":
            writer.write_tensor_data(job[1].data, tensor_endianess=reader.endianess)
        elif job[0] == "hot":
            data, ids = job[1].data, job[2]
            out = np.zeros((len(ids) + 1,) + data.shape[1:], dtype=data.dtype)  # last = zero expert
            if ids:
                out[:-1] = data[ids]
            writer.write_tensor_data(out)
        elif job[0] == "zeros":
            write_hole(writer, int(np.prod(job[1])) * np.dtype(job[2]).itemsize)
        else:
            writer.write_tensor_data(job[1])
    fout = writer.fout[0]
    fout.flush()
    fout.truncate(fout.tell())  # a trailing hole still has to count towards the file size
    writer.close()
    print(f"skeleton {dst}: {os.path.getsize(dst) / 1e9:.2f} GB "
          f"({os.stat(dst).st_blocks * 512 / 1e9:.2f} GB on disk), index {index}")


if __name__ == "__main__":
    main()
