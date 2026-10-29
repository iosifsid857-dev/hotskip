#!/bin/bash
# Overnight: Kimi K2.6 routing traces -> HOTskip store -> speed/sanity test -> power off.
#   systemd-run --user --unit kimi-night scripts/night-kimi.sh /path/to/Kimi-K2.6-IQ2_XXS-00001-of-00007.gguf
# Log: ~/.cache/hotskip/kimi-night.log. Rerun after a power cut: finished traces are skipped.
set -u
LOG=~/.cache/hotskip/kimi-night.log
MODEL=${1:?usage: night-kimi.sh <first shard of the model, e.g. Kimi-K2.6-IQ2_XXS-00001-of-00007.gguf>}
NAME=kimi-k2.6
mkdir -p "$(dirname "$LOG")"
exec >>"$LOG" 2>&1
echo "=== $(date '+%F %T') start"

# power guard: on battery for 2 minutes (the Bluetti ran dry, or no Bluetti) -> stop everything, power off
(
    off=0
    while sleep 30; do
        if [ "$(cat /sys/class/power_supply/AC0/online)" = 0 ] && \
           ! grep -qx 1 /sys/class/power_supply/ucsi-source-psy-USBC000:00*/online 2>/dev/null; then
            off=$((off + 30))
        else
            off=0
        fi
        if [ $off -ge 120 ]; then
            echo "$(date '+%T') no mains for 2 min -> stopping and powering off"
            pkill -x llama-moe-trace
            pkill -x llama-server
            sync
            systemctl poweroff
        fi
    done
) &
GUARD=$!

ask() {  # $1 = question -> answer + speed into the log
    curl -s -m 1800 http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' \
        -d "$(python3 -c 'import json,sys; print(json.dumps({"messages":[{"role":"user","content":sys.argv[1]}],"max_tokens":160,"temperature":0}))' "$1")" |
        python3 -c 'import json,sys; r=json.load(sys.stdin); t=r["timings"]
print(r["choices"][0]["message"]["content"].strip())
print("--- gen {:.2f} tok/s, prompt {:.1f} tok/s".format(t["predicted_per_second"], t["prompt_per_second"]))'
}

test_run() {  # $1 = label
    echo "--- $(date '+%T') test: $1"
    hotskip run $NAME --no-browser --ctx 8192 &
    for _ in $(seq 900); do curl -sf http://127.0.0.1:8080/health >/dev/null && break; sleep 2; done
    ask "Привет! Кто ты? Ответь в двух предложениях."
    ask "Explain in three sentences why the sky is blue."
    pkill -x llama-server; wait %1 2>/dev/null; sleep 5   # background jobs ignore SIGINT: stop the server itself
}

if [ ! -f ~/.local/share/hotskip/$NAME/hotskip.json ]; then
    # traces with the attention of 30 layers on the GPU. At run time all layers stay on the CPU and the GPU
    # holds 4 GiB of hot experts (MIN_HOT=1: not 8 per layer); 20 RAM slots per layer (688 MB per slot row)
    MIN_HOT=1 systemd-inhibit --what=sleep:idle:handle-lid-switch --who=hotskip --why="Kimi traces" \
        hotskip build "$MODEL" --name $NAME --title "Kimi K2.6 1T" --ngl 0 --trace-ngl 30 \
        --hot 4.0 --slots 20 --prompts 18 --trace-tokens 32 || { echo "BUILD FAILED"; kill $GUARD; exit 1; }
    # quality-checked settings (README); ub 8 because of MIN_HOT. Kimi also needs
    # "no_think_kwargs": {"thinking": false} in hotskip.json (tune only sets plain values)
    hotskip tune $NAME io_threads=0 ub=8 ctx=8192 skip=0.4 n_used_override=7 ngl=0 \
        "ot=hexps=CUDA0,cexps=CPU,moe_.map=CPU"
    python3 -c "import json,os; p=os.path.expanduser('~/.local/share/hotskip/$NAME/hotskip.json'); c=json.load(open(p)); c['no_think_kwargs']={'thinking': False}; json.dump(c,open(p,'w'),indent=1)"
fi

systemd-inhibit --what=sleep:idle:handle-lid-switch --who=hotskip --why="Kimi test" bash -c "$(declare -f ask test_run); NAME=$NAME; test_run 'kimi'"

echo "=== $(date '+%F %T') done, powering off"
kill $GUARD
sync
systemctl poweroff
