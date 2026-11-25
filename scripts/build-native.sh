#!/usr/bin/env bash
set -euo pipefail

suffix="${1:?usage: build-native.sh <suffix> <cpu|cuda>}"
mode="${2:?usage: build-native.sh <suffix> <cpu|cuda>}"
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"

build_dir="$root/backend/build"
args=(-S "$root/backend" -B "$build_dir" -DCMAKE_BUILD_TYPE=Release)

if [[ "$mode" == "cuda" ]]; then
    args+=(-DTSGRAD_CUDA=ON "-DCMAKE_CUDA_ARCHITECTURES=75;80;86;89;90")
elif [[ "$mode" == "cpu" ]]; then
    args+=(-DTSGRAD_CUDA=OFF)
else
    echo "mode must be cpu or cuda" >&2
    exit 1
fi

if [[ -n "${CROSS:-}" ]]; then
    args+=("-DCMAKE_TOOLCHAIN_FILE=$root/backend/cmake/${CROSS}.cmake")
fi

if [[ -n "${CMAKE_OSX_ARCHITECTURES:-}" ]]; then
    args+=("-DCMAKE_OSX_ARCHITECTURES=${CMAKE_OSX_ARCHITECTURES}")
fi

if [[ "${OS:-}" == "Windows_NT" ]]; then
    export MSYS2_ARG_CONV_EXCL='*'
    args+=(-G "Visual Studio 17 2022" -A x64)
fi

if [[ "$(uname -s)" == "Darwin" && -n "${OpenMP_ROOT:-}" ]]; then
    args+=(
        "-DOpenMP_ROOT=${OpenMP_ROOT}"
        "-DOpenMP_CXX_FLAGS=-Xpreprocessor -fopenmp -I${OpenMP_ROOT}/include"
        "-DOpenMP_CXX_LIB_NAMES=libomp"
        "-DOpenMP_libomp_LIBRARY=${OpenMP_ROOT}/lib/libomp.dylib"
    )
fi

omp_lib="/c/Program Files/LLVM/lib/libomp.lib"
if [[ "${OS:-}" == "Windows_NT" && -f "$omp_lib" ]]; then
    args+=("-DOpenMP_libomp_LIBRARY=$(cygpath -m "$omp_lib")")
fi

cmake "${args[@]}"
cmake --build "$build_dir" --config Release --parallel

dest="$root/npm/$suffix"
mkdir -p "$dest"
bundle=(node "$root/scripts/bundle-native.mjs" "$root/backend/build/Release/tsgrad_native.node" "$dest")
if [[ "$mode" == "cuda" ]]; then
    bundle+=(--cuda)
fi
if [[ -n "${CROSS:-}" ]]; then
    bundle+=(--cross "$CROSS")
fi
"${bundle[@]}"
ls -lh "$dest"
