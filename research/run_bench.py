#!/usr/bin/env python3
"""LLM lab: start LM Studio's bundled llama-server with a given offload config, generate a fixed
answer, and report prompt / generation speed and VRAM use. One JSON line per run.
Usage: run_bench.py model.gguf name -- <extra llama-server args>"""
import json
import os
import subprocess
import sys
import time
import urllib.request

B = os.path.expanduser("~/.lmstudio/extensions/backends/llama.cpp-linux-x86_64-nvidia-cuda12-avx2-2.60.0")
V = os.path.expanduser("~/.lmstudio/extensions/backends/vendor/linux-llama-cuda12-vendor-v1")
PORT = 8077
PROMPT = ("Explain in detail how an ultrasonic distance sensor HC-SR04 works with an ESP32, "
          "including wiring, the voltage divider on the echo pin, and example Arduino code.")


def vram_used_mb():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    return int(out.strip().splitlines()[0])


def read_bytes(pid):
    with open(f"/proc/{pid}/io") as f:
        return int(next(l for l in f if l.startswith("read_bytes")).split()[1])


def post(path, payload, timeout=900):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def main():
    model, name = sys.argv[1], sys.argv[2]
    extra = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    base_vram = vram_used_mb()
    server = os.environ.get("LLAMA_SERVER", f"{B}/llama-server")
    env = dict(os.environ) if "LLAMA_SERVER" in os.environ else dict(os.environ, LD_LIBRARY_PATH=f"{B}:{V}")
    cmd = [server, "-m", model, "--port", str(PORT), "-c", "4096", "--no-webui"] + extra
    memmax = os.environ.get("MEMMAX")  # e.g. "10G": cap RAM incl. page cache, forcing SSD reads
    if memmax:
        fd = os.open(model, os.O_RDONLY)  # evict the model from page cache so the cap really bites
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(fd)
        cmd = ["systemd-run", "--user", "--scope", "-q", "-p", f"MemoryMax={memmax}",
               "-p", "MemorySwapMax=0"] + cmd
    log = open(f"/tmp/llmlab-{name}.log", "w")
    srv = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
    try:
        t0 = time.time()
        while True:
            if srv.poll() is not None:
                print(json.dumps({"name": name, "error": "server exited", "log": f"/tmp/llmlab-{name}.log"}))
                return
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2) as r:
                    if json.loads(r.read()).get("status") == "ok":
                        break
            except Exception:
                pass
            time.sleep(1)
        load_s = time.time() - t0
        post("/completion", {"prompt": "Hi", "n_predict": 8})  # warm-up
        rb0 = read_bytes(srv.pid)
        prompt = open(os.environ["PROMPT_FILE"]).read() if os.environ.get("PROMPT_FILE") else PROMPT
        res = post("/completion", {"prompt": prompt, "n_predict": int(os.environ.get("N_PREDICT", 256)), "temperature": 0, "cache_prompt": False})
        rb1 = read_bytes(srv.pid)
        tm = res["timings"]
        print(json.dumps({
            "name": name, "args": " ".join(extra), "load_s": round(load_s, 1),
            "prompt_tok_s": round(tm["prompt_per_second"], 1), "gen_tok_s": round(tm["predicted_per_second"], 2),
            "gen_tokens": tm["predicted_n"], "vram_mb": vram_used_mb() - base_vram,
            "memmax": memmax, "ssd_mb_per_tok": round((rb1 - rb0) / 1e6 / max(1, tm["predicted_n"] + tm["prompt_n"]), 1),
        }), flush=True)
        with open(f"/tmp/llmlab-{name}.answer.txt", "w") as f:
            f.write(res["content"])
    finally:
        srv.terminate()
        try:
            srv.wait(30)
        except subprocess.TimeoutExpired:
            srv.kill()


if __name__ == "__main__":
    main()
