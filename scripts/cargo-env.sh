#!/bin/sh
# Rust build environment for Genesis.
#
# Genesis needs a C toolchain (to link) and libclang (for v4l2-sys-mit's
# bindgen). This host now provides both through the system LLVM/Clang
# packages, so no Zig shims or custom bindgen include paths are needed.
#
# Usage:  . ./scripts/cargo-env.sh && cargo test --release

if ! command -v clang >/dev/null 2>&1; then
    echo "cargo-env.sh: clang not found" >&2
    return 1 2>/dev/null || exit 1
fi

CC=${CC:-clang}
CXX=${CXX:-clang++}
AR=${AR:-ar}
RANLIB=${RANLIB:-ranlib}
export CC CXX AR RANLIB

# bindgen needs the directory containing libclang. Prefer the installed
# LLVM config; keep an explicit LIBCLANG_PATH override working.
if [ -z "${LIBCLANG_PATH:-}" ]; then
    LLVM_CONFIG=${LLVM_CONFIG:-}
    if [ -z "$LLVM_CONFIG" ]; then
        LLVM_CONFIG=$(command -v llvm-config 2>/dev/null || command -v llvm-config-18 2>/dev/null || true)
    fi
    if [ -n "$LLVM_CONFIG" ]; then
        LIBCLANG_PATH=$("$LLVM_CONFIG" --libdir)
        export LIBCLANG_PATH
    fi
fi

if [ -z "${LIBCLANG_PATH:-}" ] || [ ! -d "$LIBCLANG_PATH" ]; then
    echo "cargo-env.sh: libclang directory not found; install libclang-dev or set LIBCLANG_PATH" >&2
    return 1 2>/dev/null || exit 1
fi

unset BINDGEN_EXTRA_CLANG_ARGS
