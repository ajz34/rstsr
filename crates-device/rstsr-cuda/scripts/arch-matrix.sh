#!/usr/bin/env bash
#
# Arch matrix: run the hardware test suite compiled for every arch this card can host.
#
# Why this exists
# ---------------
# There are two accept/reject gates between generated source and a running kernel:
#
#     source --[NVRTC]--> PTX --[driver load + JIT]--> SASS
#                 gate 1                    gate 2
#
# Gate 1 runs on the machine that wrote the kernel. Gate 2 runs on the user's machine, at first use,
# and a driver can reject PTX that NVRTC accepted (`max.NaN.f64` -> CUDA_ERROR_INVALID_PTX on one
# driver). So "it compiled here" says nothing about the gate that actually breaks.
#
# PTX is forward-compatible: a driver will JIT PTX for any arch at or below its own. That is what
# makes a matrix possible from one card — set `RSTSR_CUDA_ARCH`, and the same suite recompiles,
# reloads and re-checks its answers for every older target.
#
# Each run gets its own cache directory on purpose. The disk cache is keyed on the arch, so sharing
# it would be *correct* — but a run that silently served the previous arch's PTX would look exactly
# like a pass, and proving that cannot happen is cheaper than reasoning about it: the script counts
# the PTX files each run produced and refuses to call an empty compile a success.
#
# Usage
# -----
#   scripts/arch-matrix.sh              # every arch at or below the detected card
#   scripts/arch-matrix.sh 70 80 86     # explicit list
#   CUDARC_CUDA_VERSION=12040 scripts/arch-matrix.sh
#
set -euo pipefail

cd "$(dirname "$0")/../../.."

BINARIES=(test_kernels test_matmul_tiled)
LADDER=(70 75 80 86 87 89 90 100 120)

: "${CUDARC_CUDA_VERSION:=12060}"
export CUDARC_CUDA_VERSION

# Preflight: the repo pins a toolchain in rust-toolchain.toml, and on a machine where that channel
# lacks `cargo` every run below would fail with the same confusing message. Say it once, up front.
# `RUSTUP_TOOLCHAIN` is inherited on purpose — the CI workflows set it to stable for the same reason.
if ! cargo --version >/dev/null 2>&1; then
    echo "arch-matrix: 'cargo' does not run on the active toolchain:" >&2
    rustc --version 2>&1 | head -2 | sed 's/^/  /' >&2
    echo "arch-matrix: set RUSTUP_TOOLCHAIN (e.g. RUSTUP_TOOLCHAIN=stable $0)." >&2
    exit 2
fi

# Two sources, in order of cost. `compute_cap` is a recent nvidia-smi field, so on an older driver
# the fallback asks the driver through the crate's own query — the same one the launch path uses, so
# the answer cannot disagree with what the kernels target. `examples/arch.rs` prints bare digits.
detect_cap() {
    local cap
    cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '. \r')
    if [[ "$cap" =~ ^[0-9]+$ ]]; then
        echo "$cap"
        return 0
    fi
    cap=$(cargo run -q -p rstsr-cuda --example arch 2>/dev/null | tail -1 | tr -d '. \r')
    [[ "$cap" =~ ^[0-9]+$ ]] || return 1
    echo "$cap"
}

if [[ $# -gt 0 ]]; then
    ARCHES=("$@")
    DEV=""
else
    if ! DEV=$(detect_cap); then
        echo "arch-matrix: cannot detect the device compute capability (nvidia-smi field missing, and" >&2
        echo "             'cargo run --example arch' did not print a number either)." >&2
        echo "arch-matrix: pass the arches explicitly, e.g. '$0 70 80 86'." >&2
        exit 2
    fi
    ARCHES=()
    for a in "${LADDER[@]}"; do
        (( a <= DEV )) && ARCHES+=("$a")
    done
fi

if [[ ${#ARCHES[@]} -eq 0 ]]; then
    echo "arch-matrix: no arches to run" >&2
    exit 2
fi

LOGDIR=$(mktemp -d "${TMPDIR:-/tmp}/rstsr-arch-matrix.XXXXXX")
echo "arch-matrix: $(cargo --version)  CUDARC_CUDA_VERSION=$CUDARC_CUDA_VERSION  device=compute_${DEV:-unknown}  arches=${ARCHES[*]}"
echo "arch-matrix: logs in $LOGDIR"
echo

RESULTS=()
FAILED=0

for a in "${ARCHES[@]}"; do
    arch="compute_$a"
    cache=$(mktemp -d "${TMPDIR:-/tmp}/rstsr-arch-cache-$a.XXXXXX")
    started=$SECONDS
    status="ok"
    for bin in "${BINARIES[@]}"; do
        log="$LOGDIR/$arch.$bin.log"
        if ! RSTSR_CUDA_ARCH="$arch" RSTSR_CUDA_CACHE_DIR="$cache" \
            cargo test -p rstsr-cuda --test "$bin" >"$log" 2>&1; then
            status="FAILED"
            echo "  $arch / $bin: failed (see $log)"
            # the failure text names the entry symbol; show it immediately rather than at the end
            grep -E "panicked|Error::|error\[|NVRTC|INVALID_PTX|named symbol" "$log" | head -6 | sed 's/^/    /'
        fi
    done

    ptx=$(find "$cache" -name 'k_*.ptx' | wc -l | tr -d ' ')
    # A run that compiled nothing proves nothing — it means the cache answered for another arch, or
    # no kernel was ever launched. Both are the failure this script is supposed to detect.
    if [[ "$status" == "ok" && "$ptx" -eq 0 ]]; then
        status="NO-COMPILE"
        echo "  $arch: suite passed but produced no PTX — nothing was compiled, so the run proves nothing"
    fi
    [[ "$status" != "ok" ]] && FAILED=1

    RESULTS+=("$(printf '  %-13s %-11s %4s PTX %5ss' "$arch" "$status" "$ptx" "$((SECONDS - started))")")
    rm -rf "$cache"
done

echo
echo "arch matrix summary"
echo "  arch          status      compiled  time"
for r in "${RESULTS[@]}"; do echo "$r"; done
echo

if [[ "$FAILED" -ne 0 ]]; then
    echo "arch-matrix: at least one arch failed; logs in $LOGDIR" >&2
    exit 1
fi

echo "arch-matrix: every arch at or below compute_${DEV:-requested} compiles, loads and passes."
echo "Note what this does NOT cover: driver versions (the JIT is host-side) and arches above this card."
exit 0
