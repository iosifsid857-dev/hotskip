#!/bin/bash
# Expert store: speed vs RAM slots under a fixed RAM cap. Results -> results_store.jsonl
export DISABLE_MANGOHUD=1; unset MANGOHUD
export LLAMA_SERVER=~/Projects/src/llama.cpp/build-cuda/bin/llama-server EXPSTORE_STATS=/tmp/expstore.json
D=~/Data/AI/llm-lab
for S in ${SLOTS:-64 80}; do
  [ -f $D/qwen35-store$S.gguf ] || nice python3 make_store.py $D/qwen35-hot3.4.gguf $D/qwen35-store$S.gguf $D/qwen35-store$S.idx $S "traces/qwen_*.txt" >&2
  for mem in ${MEMS:-7G}; do
    EXPSTORE_INDEX=$D/qwen35-store$S.idx EXPSTORE_PINNED=$((S - 32)) MEMMAX=$mem timeout 900 \
      python3 run_bench.py $D/qwen35-store$S.gguf s_store${S}_$mem -- -ngl 99 -ot cexps=CPU -ot "moe_.map=CPU" \
      -t 8 --no-op-offload --load-mode none -ub 4
    echo "  stats: $(cat /tmp/expstore.json)"
  done
done | tee -a results_store.jsonl
