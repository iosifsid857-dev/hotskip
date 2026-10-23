#!/bin/bash
# Build the HOTskip-patched llama.cpp into ./llama.cpp/build/bin (where hotskip.py looks for it).
#   scripts/build.sh                 CUDA build (NVIDIA)
#   HOTSKIP_CMAKE_FLAGS="..." scripts/build.sh   extra CMake flags, e.g. for a newer GCC than your nvcc supports:
#       HOTSKIP_CMAKE_FLAGS="-DCMAKE_CUDA_FLAGS=-std=c++17" scripts/build.sh
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$PWD
BASE=$(cat patches/LLAMA_CPP_BASE_COMMIT)

if [ ! -d llama.cpp/.git ]; then
    git clone https://github.com/ggml-org/llama.cpp llama.cpp
fi
cd llama.cpp
if ! git cat-file -e "$BASE^{commit}" 2>/dev/null; then
    git fetch origin "$BASE"
fi

if git apply --reverse --check "$ROOT/patches/llama.cpp-hotskip.patch" 2>/dev/null; then
    echo "patch already applied"
else
    git checkout -q "$BASE"
    git apply "$ROOT/patches/llama.cpp-hotskip.patch"
    echo "patch applied on $BASE"
fi
cp -r "$ROOT/llama.cpp-additions/." .

# shellcheck disable=SC2086
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release ${HOTSKIP_CMAKE_FLAGS:-}
cmake --build build -j "$(nproc)" --target llama-server llama-moe-trace llama-perplexity
echo
echo "done: $ROOT/llama.cpp/build/bin"
echo "next: ln -s $ROOT/hotskip.py ~/.local/bin/hotskip && hotskip build /path/to/model.gguf"
