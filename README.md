# HOTskip

Run Mixture-of-Experts LLMs that are **bigger than your RAM + VRAM** on an ordinary gaming laptop, at
usable speed. HOTskip is a patch set for [llama.cpp](https://github.com/ggml-org/llama.cpp) plus a small
command-line tool.

> Status: research prototype, measured on one machine (see below). Русская версия: [README.ru.md](README.ru.md).

## The idea

In a MoE model every token uses only a few experts per layer (Qwen3.6-35B-A3B: 8 of 256, gpt-oss-120b:
4 of 128). Routing traces show that expert use is very uneven: in our traces the top 5% of experts do 28%
of the work, and 10% are never used. HOTskip keeps the experts where they are cheapest to use:

| Tier | What lives there | How it is chosen |
|---|---|---|
| **VRAM** | the globally hottest experts, from all layers | routing traces, under a VRAM budget |
| **RAM** | a fixed number of *slots* per layer: part pinned to popular experts, the rest LRU | popularity + recency |
| **SSD** | everything else, read on demand with `O_DIRECT`, in parallel | — |

On top of that:

* **Cold experts are computed on the CPU, right where they are.** Moving experts to the GPU over PCIe is
  slower than computing them from RAM on a laptop (measured: PCIe ~12 GB/s vs RAM ~36 GB/s), which is why
  llama.cpp's own dynamic `--moe-cache` loses on this kind of hardware.
* **Prefetch.** Each layer applies the *next* layer's router to its own hidden state to guess the next
  layer's experts (58% of misses guessed with 8 candidates); background threads start reading them.
* **Grouped prompt processing.** Big batches split the needed cold experts of a layer into groups that fit
  the RAM slots, so each expert is read from SSD at most once per batch. Those groups are then computed
  **on the GPU** (llama.cpp's op offload, patched to understand the store's skipped experts): long prompts
  read 2.2× faster on DeepSeek-V4.
* **No 60 GB copies.** The tool writes a small "skeleton" GGUF (hot experts + empty slots); cold experts
  are read straight from the original model file.

## Results on a 32 GB RAM / 8 GB VRAM laptop

Laptop: i9-13900H, RTX 4070 Laptop 8 GB, 32 GB DDR5, NVMe (~3.7 GB/s random reads at queue depth ≥ 4).
No artificial limits; the desktop itself uses 1–2 GB of VRAM.

| Model | File | vs. RAM+VRAM | stock llama.cpp | **HOTskip** |
|---|---|---|---|---|
| Qwen3.6-35B-A3B Q4_K_M | 21 GB | fits (we use ~7 GB RAM) | — | **17–19 tok/s** |
| gpt-oss-120b MXFP4 | 65 GB | 1.6× bigger | 1.42 tok/s | **4.9–7 tok/s (×3.5–5)** |
| DeepSeek-V4-Flash (284B) Q2_K | 98 GB | 2.4× bigger | 0.49 tok/s | **~3 tok/s (×6)** |
| **Kimi K2.6 (1T)** IQ2_XXS, 7 files | 271 GB | **6.8× bigger** | not measured | **~1.05 tok/s** |

Stock = best llama.cpp setup for the machine (`--cpu-moe`, mmap). Prompt processing: gpt-oss 0.6 → 6.4
tok/s, DeepSeek 0.8 → 4.8 tok/s. The bigger the model is relative to memory, the bigger the gain.
gpt-oss ranges 5–7 tok/s depending on how much VRAM the desktop leaves free (hot-expert budget).
Kimi K2.6 is a one-trillion-parameter model (384 experts × 60 layers, 8 per token): every token needs
~5.5 GB of expert weights, ~2.5 GB of which still come from the SSD, so the disk is the limit there
(0.89 tok/s with all 8 experts, 1.05 tok/s with 7 experts + skip 0.4 at unchanged quality, see below).

### Quality

HOTskip computes the same thing as llama.cpp; only the optional *skip* (below) trades accuracy for speed.
Checked per model against stock llama.cpp:

| Model | HOTskip, no skip | with skip | skip used |
|---|---|---|---|
| Qwen3.6-35B | PPL 4.38 vs stock 4.39 | PPL 6.10 vs 6.14 at 0.64 | 0.64 |
| gpt-oss-120b | KLD 0.088, top-1 88% (stock vs itself at another batch size: 0.081, 89%) | KLD 0.17 at 0.4 | off |
| DeepSeek-V4-Flash | PPL 4.10 | 4.02 at 0.4; 4.28 at 0.64 | 0.4 |
| Kimi K2.6 | PPL 4.90 | 4.64 at 0.4; 7.34 at 0.64 ✗ | 0.4, 7 experts (PPL 5.21) |

*Skip:* a cold expert that is not in RAM and whose routing weight is below `f ×` the token's mean
weight is skipped instead of read from SSD. Models with few experts per token (gpt-oss: 4) tolerate it
least, so the threshold is set per model after a quality check. Kimi also loses quality quickly when
running fewer experts per token (6 experts: PPL 8.18), so only 7 is used. One generated story is not a
quality check: several settings that "read fine" doubled the perplexity.

## Lab test: Qwen3.6-35B-A3B squeezed into 7 GB of RAM

The model (21 GB) is forced to run in **7 GB of RAM** (cgroup memory limit incl. page cache) to study a
model that does not fit, quickly.

| Setup (7 GB RAM) | Generation | SSD read per token |
|---|---|---|
| stock llama.cpp, best layer offload (`--n-cpu-moe 33`, mmap) | 1.4 tok/s | 1072 MB |
| hot experts in VRAM, cold via mmap | 2.4 tok/s | 605 MB |
| **HOTskip** (80 slots/layer) | 16.4 tok/s | 51 MB |
| **HOTskip + prefetch** | **17.8 tok/s** | — |

Prompt processing: 28.8 → **71.8 tok/s** with grouped batches. Quality is unchanged: perplexity on a
small test text 6.14 (stock) vs 6.06–6.13 (HOTskip, all modes). (A proper wikitext run is still to do.)

In everyday chat (no artificial limit, ~7 GB RAM used instead of 21 GB): 17–19 tok/s.

Things that did **not** help (useful to know):

* smarter eviction policies — LRU vs LFU-with-decay vs ARC-like differ by < 1.5% misses;
* predicting 2–3 layers ahead — SSD bandwidth, not latency, is the limit;
* prefetch on DeepSeek-V4 — only ~46% of guessed experts are used, and the wasted reads cost more than
  the hits gain once the SSD is saturated (it helps on Qwen and gpt-oss);
* reading the three parts of an expert in parallel — the bottleneck is the volume of SSD reads;
* keeping more of DeepSeek's attention on the GPU — moving 6 layers of attention to the CPU costs ~4%.

Rule of thumb from all of this: once a model spills to SSD, only fewer bytes read per token makes it
faster (more RAM, better hot/pinned sets, skipping weak experts).

## Install

Requirements: Linux with systemd, an NVIDIA GPU with CUDA, Python 3 with `numpy`, and a fast NVMe SSD.
Tested on Fedora 44, i9-13900H, RTX 4070 Laptop 8 GB, 32 GB DDR5.

```
git clone https://github.com/iosifsid857-dev/hotskip && cd hotskip
scripts/build.sh                                  # patched llama.cpp → ./llama.cpp/build/bin
ln -s "$PWD/hotskip.py" ~/.local/bin/hotskip
```

With GCC newer than your CUDA supports (e.g. GCC 16 + nvcc 13.0) add
`HOTSKIP_CMAKE_FLAGS="-DCMAKE_CUDA_FLAGS=-std=c++17"`. Stores go to `~/.local/share/hotskip`
(`HOTSKIP_HOME` to change), binaries are looked up in `./llama.cpp/build/bin` (`HOTSKIP_LLAMA_BIN`).

## Usage

```
hotskip build model.gguf --title "My model"   # trace routing, size tiers for this machine, write store
hotskip run <name>                            # chat in the browser (llama.cpp web UI + OpenAI API)
hotskip start <name> / hotskip stop           # background server, used by the app-menu entries
hotskip list
hotskip tune <name> skip=0.4 io_threads=0 ngl=14 ub=1024 ctx=32768   # per-model settings
```

Split models (`model-00001-of-00007.gguf`) are read from all their shards directly — no merged copy is
needed. Extra per-model keys in `hotskip.json` for very big models: `ot` (tensor placement, e.g. Kimi keeps
its hot experts on the GPU and everything else on the CPU: `"hexps=CUDA0,cexps=CPU,moe_.map=CPU"` with
`ngl 0`), `n_used_override` (experts per token), `no_think_kwargs` (chat-template switch for models that
ignore `--reasoning off`, Kimi: `{"thinking": false}`), `extra_args`. `MIN_HOT=1` when building a store
lets the hot experts fit a small GPU (then run with `ub=8`).

The server speaks the OpenAI API (`http://127.0.0.1:8080/v1`), so agents such as Cline work with it.

`build` traces 18 varied prompts in the model's own chat format, then picks the VRAM budget from free
VRAM and the RAM slots from available RAM. Every run is wrapped in a cgroup memory limit, so a mistake
kills the server, not your desktop.

## Limitations

* Research prototype, tested on one laptop and four models (Qwen3.6-35B-A3B, gpt-oss-120b, DeepSeek-V4-Flash,
  Kimi K2.6). Supported architectures: `qwen35moe`, `gpt-oss`, `deepseek4`, `deepseek2` (Kimi K2.x and
  other DeepSeek-V3-style models); others need a few lines in their graph code.
* Keep 1.5–2 GB of VRAM free for the desktop: with the GPU completely full, the NVIDIA 615 driver on our
  machine sometimes got stuck until reboot (new GPU apps crashed).
* Expert prediction is only ~40% accurate on Kimi, so prefetching does not pay off there yet.
* NVIDIA/CUDA only for now. Linux only (`O_DIRECT`, systemd scopes).
* Agents work but are slow on the biggest models: generation, not prompt reading, is the limit there.
* The patch is against a fixed llama.cpp commit (`patches/LLAMA_CPP_BASE_COMMIT`).

## Repository layout

* `hotskip.py` — the command-line tool; `make_store2.py` — builds a store (skeleton + index);
  `prompts.txt` — the 18 prompts used for routing traces.
* `patches/llama.cpp-hotskip.patch` + `llama.cpp-additions/` (the `moe-trace` tool) — the llama.cpp changes.
* `research/` — benchmark and analysis scripts with the raw results behind every number in this README.

## How it is built

* `src/llama-expstore.{h,cpp}` — the store: slot bookkeeping, `O_DIRECT` reads, prefetch threads,
  grouped batches; plugged into the graph as `ggml_custom` ops.
* `src/llama-graph.cpp` — hot/cold split of the MoE FFN (two expert stacks + id maps).
* CPU `mul_mat_id`: id `-1` means "skip, output zero".
* CUDA `mul_mat_id` / `add_id`: the same `-1` handling, so long prompts can run on the GPU.

## Support

If HOTskip helped you, you can support the author (Ukraine 🇺🇦): [Monobank jar](https://send.monobank.ua/jar/ABkwFDAY2V).

## Credits

Built on [llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT). Developed together with Claude (Anthropic)
as a coding and research assistant. Thanks to the authors of the open models used for testing.

## Related work

PowerInfer (hot/cold neurons), ktransformers (CPU/GPU MoE offload), MoE-Infinity / Pre-gated MoE
(expert prefetch), llama.cpp `--n-cpu-moe` and `--moe-cache`. HOTskip's angle: a laptop with a slow PCIe
link and a fast NVMe — static hot set in VRAM, compute cold experts where they lie, SSD as a third tier.
