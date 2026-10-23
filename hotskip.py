#!/usr/bin/env python3
"""HOTskip - run MoE models bigger than RAM + VRAM on a laptop.

The hottest experts live in VRAM, a slot cache of experts in RAM (part pinned by popularity,
the rest LRU), everything else is read from SSD on demand, with the next layer's experts guessed
and prefetched in the background. Built on a patched llama.cpp (scripts/build.sh).

  hotskip build <model.gguf> [--name N] [--hot GiB] [--slots N]   trace routing, size tiers, write store
  hotskip run <name> [--port 8080] [--ctx 16384] [--think]        start the chat (web UI + OpenAI API)
  hotskip start <name> / stop                                     background server for the app menu
  hotskip tune <name> key=value ...                               per-model settings (skip, ngl, ub, ctx, ...)
  hotskip menu                                                    (re)create app menu entries
  hotskip list                                                    built models
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
import webbrowser

LAB = os.path.dirname(os.path.realpath(__file__))
HOME = os.path.expanduser("~")


def _first_dir(*cands):
    for c in cands:
        if c and os.path.isdir(c):
            return c
    return cands[-1]


# patched llama.cpp binaries (scripts/build.sh puts them in <repo>/llama.cpp/build/bin) and where stores go
BIN = _first_dir(os.environ.get("HOTSKIP_LLAMA_BIN"), os.path.join(LAB, "llama.cpp", "build", "bin"),
                 os.path.join(LAB, "..", "llama.cpp", "build", "bin"), f"{HOME}/Projects/src/llama.cpp/build-cuda/bin")
STORE_DIR = os.environ.get("HOTSKIP_HOME") or f"{HOME}/.local/share/hotskip"
PROMPTS = next((p for p in (os.path.join(LAB, "prompts.txt"), os.path.join(LAB, "..", "prompts.txt"))
                if os.path.isfile(p)), os.path.join(LAB, "prompts.txt"))
MAKE_STORE = next((p for p in (os.path.join(LAB, "make_store2.py"), os.path.join(LAB, "make_store.py"))
                   if os.path.isfile(p)), os.path.join(LAB, "make_store2.py"))
GIB = 1024 ** 3

for _p in (os.path.join(os.path.dirname(os.path.realpath(__file__)), "llama.cpp", "gguf-py"),
           os.path.expanduser("~/Projects/src/llama.cpp/gguf-py")):
    if os.path.isdir(_p):
        sys.path.insert(0, _p)
        break


def mem_available():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("no MemAvailable")


def vram_free():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout
    return int(out.split()[0]) * 1024 * 1024


def model_info(path):
    import gguf
    r = gguf.GGUFReader(path)
    arch = bytes(r.fields["general.architecture"].parts[-1]).decode()
    f = lambda k: int(r.fields[f"{arch}.{k}"].parts[-1][0])  # noqa: E731
    n_expert, n_used = f("expert_count"), f("expert_used_count")
    t = {x.name: x for x in r.tensors}
    layers = [il for il in range(f("block_count")) if f"blk.{il}.ffn_down_exps.weight" in t]
    per_layer = {il: sum(t[n].n_bytes for n in t if n.startswith(f"blk.{il}.ffn_") and "_exps." in n) // n_expert
                 for il in layers}
    experts = sum(x.n_bytes for x in r.tensors if "_exps." in x.name)
    return {"arch": arch, "n_expert": n_expert, "n_used": n_used, "moe_layers": len(layers),
            "expert_row_bytes": sum(per_layer.values()),  # one expert in every MoE layer
            "experts_bytes": experts, "other_bytes": sum(x.n_bytes for x in r.tensors) - experts}


def scope(mem_bytes):
    """Run under a RAM cap: if anything goes wrong only this process dies, not the desktop."""
    return ["systemd-run", "--user", "--scope", "-q", "-p", f"MemoryMax={mem_bytes // (1024 * 1024)}M",
            "-p", "MemorySwapMax=0"]


def build(a):
    model = os.path.abspath(a.model)
    name = a.name or os.path.basename(model).removesuffix(".gguf")
    d = f"{STORE_DIR}/{name}"
    os.makedirs(f"{d}/traces", exist_ok=True)
    info = model_info(model)
    print(f"{name}: {info['arch']}, {info['moe_layers']} MoE layers x {info['n_expert']} experts "
          f"({info['n_used']} per token), experts {info['experts_bytes'] / 1e9:.1f} GB, "
          f"rest {info['other_bytes'] / 1e9:.1f} GB")

    # 1. routing traces: which experts does this model actually use?
    prompts = [p for p in open(PROMPTS).read().splitlines() if p.strip()]
    if a.prompts:  # evenly spread over the list: tech, chat, stories, other languages
        prompts = [prompts[round(i * (len(prompts) - 1) / max(1, a.prompts - 1))] for i in range(a.prompts)]
    cap = max(4 * GIB, mem_available() - 4 * GIB)
    for i, p in enumerate(prompts, 1):
        out = f"{d}/traces/t_{i}.txt"
        if os.path.exists(out):
            continue
        print(f"tracing prompt {i}/{len(prompts)} ...", flush=True)
        cmd = scope(cap) + [f"{BIN}/llama-moe-trace", "-m", model, "-ngl", str(a.trace_ngl or a.ngl), "--cpu-moe", "-t", "8",
                            "-c", "2048", "-n", str(a.trace_tokens), "--temp", "0", "--chat", "-p", p,
                            "--trace-out", out + ".tmp"]
        with open(f"{d}/traces/t_{i}.log", "w") as log:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True, env=clean_env())
        os.rename(out + ".tmp", out)

    # 2. size the tiers for this machine
    reserve_vram = 1.5 * GIB  # KV cache + compute buffers
    hot = a.hot if a.hot is not None else max(0.5, (vram_free() - info["other_bytes"] - reserve_vram) / GIB)
    if a.slots is not None:
        slots = a.slots
    else:
        ram = mem_available() - 6 * GIB  # leave room for the desktop and the rest of the process
        slots = int(ram // info["expert_row_bytes"])
    min_lru = 4 * info["n_used"]  # -ub 4: one batch may need this many distinct experts per layer
    slots = max(min_lru + 8, min(slots, info["n_expert"]))
    pinned = min(slots // 4, slots - min_lru)
    print(f"tiers: VRAM {hot:.2f} GiB of hot experts, RAM {slots} slots/layer "
          f"({slots * info['expert_row_bytes'] / GIB:.1f} GiB, {pinned} pinned), rest on SSD")

    # 3. skeleton + index
    subprocess.run([sys.executable, MAKE_STORE, model, f"{d}/skeleton.gguf", f"{d}/store.idx",
                    f"{hot:.3f}", str(slots), f"{d}/traces/t_*.txt"], check=True)
    cfg = dict(info, model=model, hot_gib=hot, slots=slots, pinned=pinned, built=time.strftime("%Y-%m-%d %H:%M"),
               title=a.title or name, ngl=a.ngl)
    json.dump(cfg, open(f"{d}/hotskip.json", "w"), indent=1)
    menu(None)
    print(f"done: hotskip run {name}  (or from the app menu)")


def clean_env():
    env = dict(os.environ, DISABLE_MANGOHUD="1")
    env.pop("MANGOHUD", None)  # the MangoHud Vulkan layer crashes llama.cpp
    return env


def server_cmd(name, port, ctx, io_threads, think, ngl=None):
    d = f"{STORE_DIR}/{name}"
    cfg = json.load(open(f"{d}/hotskip.json"))
    env = clean_env()
    # per-model tuning lives in hotskip.json: skip threshold (quality-checked per model) and prefetch on/off
    if io_threads is None:
        io_threads = cfg.get("io_threads", 4)
    env.update(EXPSTORE_INDEX=f"{d}/store.idx", EXPSTORE_PINNED=str(cfg["pinned"]),
               EXPSTORE_PREFETCH=str(io_threads), EXPSTORE_PREDICT=str(cfg["n_used"] if io_threads else 0),
               EXPSTORE_SKIP=str(cfg.get("skip", 0)))
    # RAM: the slots + every non-expert weight that does not sit on the GPU (fewer GPU layers = more RAM) + buffers
    need = cfg["slots"] * cfg["expert_row_bytes"] + cfg["other_bytes"] + 3 * GIB
    cap = max(need, mem_available() - 2 * GIB)
    # big micro-batches: the store serves cold experts in groups that fit RAM (each read once per batch)
    cmd = scope(cap) + [f"{BIN}/llama-server", "-m", f"{d}/skeleton.gguf", "--host", "127.0.0.1",
                        "--port", str(port), "-c", str(ctx or cfg.get("ctx", 16384)), "-ngl", str(ngl if ngl is not None else cfg.get("ngl", 99)), "-ot", "cexps=CPU",
                        "-ot", "moe_.map=CPU", "-t", "8", "--load-mode", "none",
                        "-ub", str(cfg.get("ub", 512)), "-b", "2048", "--jinja", "--reasoning", "on" if think else "off"]
    return cmd, env


def healthy(port):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
            return json.loads(r.read()).get("status") == "ok"
    except Exception:
        return False


def run(a):
    cmd, env = server_cmd(a.name, a.port, a.ctx, a.io_threads, a.think)
    print(f"starting {a.name} on http://127.0.0.1:{a.port} (Ctrl+C to stop)", flush=True)
    srv = subprocess.Popen(cmd, env=env)
    try:
        while srv.poll() is None and not healthy(a.port):
            time.sleep(1)
        if srv.poll() is None:
            print("ready", flush=True)
            if not a.no_browser:
                webbrowser.open(f"http://127.0.0.1:{a.port}")
        srv.wait()
    except KeyboardInterrupt:
        srv.terminate()
        srv.wait()


STATE = f"{HOME}/.cache/hotskip/state.json"


def notify(text):
    subprocess.run(["notify-send", "-a", "HOTskip", "-i", f"{HOME}/.local/share/icons/hotskip.svg", "HOTskip", text],
                   check=False)


def running():
    try:
        st = json.load(open(STATE))
        os.kill(st["pid"], 0)
        return st
    except Exception:
        return None


def start(a):
    """Background server for the menu: open the chat when ready; one model at a time (VRAM)."""
    st = running()
    if st and st["name"] == a.name:
        webbrowser.open(f"http://127.0.0.1:{st['port']}")
        return
    if st:
        stop(None)
    cfg = json.load(open(f"{STORE_DIR}/{a.name}/hotskip.json"))
    os.makedirs(os.path.dirname(STATE), exist_ok=True)
    notify(f"Loading {a.name}…")
    # the desktop's own VRAM use varies (browser, live wallpaper...): if the GPU is short, keep fewer layers on it
    n_layers = cfg["moe_layers"]
    ngl = min(cfg.get("ngl", 99), n_layers)
    while True:
        cmd, env = server_cmd(a.name, a.port, a.ctx, a.io_threads, a.think, ngl)
        logpath = f"{HOME}/.cache/hotskip/{a.name}.log"
        srv = subprocess.Popen(cmd, env=env, stdout=open(logpath, "w"), stderr=subprocess.STDOUT, start_new_session=True)
        json.dump({"name": a.name, "pid": srv.pid, "port": a.port}, open(STATE, "w"))
        t0 = time.time()
        while srv.poll() is None and not healthy(a.port):
            if time.time() - t0 > 600:
                break
            time.sleep(1)
        if srv.poll() is not None and "out of memory" in open(logpath, errors="replace").read() and ngl > 4:
            ngl -= 4
            continue
        break
    if srv.poll() is not None or not healthy(a.port):
        notify(f"Could not start {a.name}, log: ~/.cache/hotskip/{a.name}.log")
        sys.exit(1)
    notify(f"{a.name} is ready, opening the chat")
    webbrowser.open(f"http://127.0.0.1:{a.port}")


def stop(_):
    st = running()
    if not st:
        if _ is not None:
            notify("Nothing is running")
        return
    os.kill(st["pid"], 15)
    for _i in range(30):
        try:
            os.kill(st["pid"], 0)
            time.sleep(0.5)
        except OSError:
            break
    os.remove(STATE)
    notify(f"{st['name']} stopped, memory is free")


ICON = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">
<defs><linearGradient id="g" x1="0" y1="1" x2="0" y2="0"><stop offset="0" stop-color="#7aa2f7"/><stop offset="1" stop-color="#bb9af7"/></linearGradient></defs>
<rect x="4" y="4" width="56" height="56" rx="14" fill="#1a1b26"/>
<path d="M32 12c2 8 12 13 12 25a12 12 0 0 1-24 0c0-6 3-9 6-12 0 5 2 8 5 8-2-7 1-15 1-21z" fill="url(#g)"/>
</svg>
"""


def menu(_):
    apps = f"{HOME}/.local/share/applications"
    os.makedirs(apps, exist_ok=True)
    os.makedirs(f"{HOME}/.local/share/icons", exist_ok=True)
    open(f"{HOME}/.local/share/icons/hotskip.svg", "w").write(ICON)
    me = f"{HOME}/.local/bin/hotskip"
    for f in glob.glob(f"{apps}/hotskip-*.desktop"):
        os.remove(f)
    entries = {"stop": ("HOTskip: остановить", f"{me} stop", "Выгрузить модель и освободить память")}
    for f in sorted(glob.glob(f"{STORE_DIR}/*/hotskip.json")):
        name = os.path.basename(os.path.dirname(f))
        title = json.load(open(f)).get("title", name)
        entries[name] = (f"HOTskip: {title}", f"{me} start {name}", "Локальная нейросеть, чат в браузере")
    for key, (title, cmd, comment) in entries.items():
        open(f"{apps}/hotskip-{key}.desktop", "w").write(
            f"[Desktop Entry]\nType=Application\nName={title}\nComment={comment}\nExec={cmd}\n"
            f"Icon={HOME}/.local/share/icons/hotskip.svg\nTerminal=false\nCategories=Utility;Development;\n")
    subprocess.run(["update-desktop-database", apps], check=False, capture_output=True)
    print("menu:", ", ".join(t for t, _c, _m in entries.values()))


def tune(a):
    p = f"{STORE_DIR}/{a.name}/hotskip.json"
    cfg = json.load(open(p))
    for kv in a.kv:
        k, v = kv.split("=", 1)
        cfg[k] = json.loads(v) if v.replace(".", "", 1).isdigit() else v
    json.dump(cfg, open(p, "w"), indent=1)
    print({k: cfg[k] for k in ("skip", "io_threads", "ngl") if k in cfg})


def list_models(_):
    for f in sorted(glob.glob(f"{STORE_DIR}/*/hotskip.json")):
        c = json.load(open(f))
        print(f"{os.path.basename(os.path.dirname(f)):40s} {c['arch']:12s} VRAM {c['hot_gib']:.1f} GiB, "
              f"RAM {c['slots']} slots, built {c['built']}")


def main():
    p = argparse.ArgumentParser(prog="hotskip", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("model")
    b.add_argument("--name")
    b.add_argument("--title", help="name shown in the app menu")
    b.add_argument("--hot", type=float, help="GiB of hot experts in VRAM (default: fit free VRAM)")
    b.add_argument("--slots", type=int, help="RAM slots per layer (default: fit available RAM)")
    b.add_argument("--trace-tokens", type=int, default=96)
    b.add_argument("--prompts", type=int, help="trace only N of the prompts (evenly spread), for very big models")
    b.add_argument("--ngl", type=int, default=99, help="layers on the GPU when the non-expert weights do not fit VRAM")
    b.add_argument("--trace-ngl", type=int, help="GPU layers while tracing (no hot experts yet, so more fit)")
    b.set_defaults(fn=build)
    r = sub.add_parser("run")
    r.add_argument("name")
    r.add_argument("--port", type=int, default=8080)
    r.add_argument("--ctx", type=int, help="context size (default: per model, 16384)")
    r.add_argument("--io-threads", type=int, help="background prefetch threads, 0 = off (default: per model)")
    r.add_argument("--no-browser", action="store_true")
    r.add_argument("--think", action="store_true", help="let the model reason before answering (slower)")
    r.set_defaults(fn=run)
    for cmd, fn in (("start", start),):
        st = sub.add_parser(cmd)
        st.add_argument("name")
        st.add_argument("--port", type=int, default=8080)
        st.add_argument("--ctx", type=int)
        st.add_argument("--io-threads", type=int)
        st.add_argument("--think", action="store_true")
        st.set_defaults(fn=fn)
    sub.add_parser("stop").set_defaults(fn=stop)
    t = sub.add_parser("tune", help="set per-model options, e.g. hotskip tune deepseek-v4-flash skip=0.25 io_threads=0")
    t.add_argument("name")
    t.add_argument("kv", nargs="+")
    t.set_defaults(fn=tune)
    sub.add_parser("menu").set_defaults(fn=menu)
    sub.add_parser("list").set_defaults(fn=list_models)
    a = p.parse_args()
    if not shutil.which("systemd-run"):
        sys.exit("systemd-run is required")
    a.fn(a)


if __name__ == "__main__":
    main()
