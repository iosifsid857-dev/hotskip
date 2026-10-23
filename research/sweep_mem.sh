#!/bin/bash
# Speed vs RAM cap: standard layer offload vs our hot/cold split. Results -> results_memcap.jsonl
export DISABLE_MANGOHUD=1; unset MANGOHUD
export LLAMA_SERVER=~/Projects/src/llama.cpp/build-cuda/bin/llama-server
ORIG=~/.lmstudio/models/llmfan46/Qwen3.6-35B-A3B-uncensored-heretic-GGUF/Qwen3.6-35B-A3B-uncensored-heretic-Q4_K_M.gguf
HOT=~/Data/AI/llm-lab/qwen35-hot3.4.gguf
for mem in ${MEMS:-none 14G 10G 7G}; do
  if [ "$mem" = none ]; then unset MEMMAX; else export MEMMAX=$mem; fi
  timeout 900 python3 run_bench.py "$ORIG" "m_ncmoe33_$mem" -- -ngl 99 --n-cpu-moe 33 -t 8 --no-op-offload
  timeout 900 python3 run_bench.py "$HOT" "m_HOTskip_$mem" -- -ngl 99 -ot cexps=CPU -ot "moe_.map=CPU" -t 8 --no-op-offload
done | tee -a results_memcap.jsonl
