# research/

The scripts and raw results behind the numbers in the main README, kept as they were used during the
experiments (some contain the author's local paths — adjust before running).

* `run_bench.py` — starts `llama-server` with given flags and measures prompt/generation speed, VRAM and
  SSD bytes per token; `MEMMAX=7G` runs it under a cgroup RAM limit (page cache included).
* `bandwidth.py` — RAM, PCIe, VRAM and NVMe bandwidth of the machine.
* `analyze_traces.py`, `analyze_tiers.py`, `analyze_policies.py`, `analyze_capsule.py` — routing-trace
  analyses: hot experts vs whole layers, VRAM/RAM/SSD tier simulation, eviction policies, per-chat "capsules".
* `split_hot_experts.py`, `set_cold_skip.py`, `make_store.py` — earlier versions of the store builder.
* `sweep_mem.sh`, `sweep_store.sh` — the RAM-cap and slot-count sweeps.
* `results/` — JSON lines / text output of every run; `traces/` — Qwen3.6-35B routing traces
  (`token layer expert…` per line).
