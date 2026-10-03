#!/bin/sh
# Rust build environment for hosts with no system C toolchain.
#
# Genesis needs `cc` (to link) and `libclang` (for v4l2-sys-mit's
# bindgen). Where no system compiler exists, Zig ships a working clang;
# this shims `cc`/`c++`/`ar`/`ranlib` to it and points bindgen at Zig's
# libc headers.
#
# Usage:  . ./scripts/cargo-env.sh && cargo test --release
#
# Verified on this host with Zig 0.16.0 under ~/tools. Substitute the two
# roots below if yours differ; everything else is mechanical. See the
# Toolchain section of AGENTS.md.
ZIG_ROOT=${ZIG_ROOT:-/home/mindy/tools/zig-x86_64-linux-0.16.0}
LIBCLANG_ROOT=${LIBCLANG_ROOT:-/home/mindy/tools/libclang/clang/native}
ZIG=$ZIG_ROOT/zig
SHIM=${SHIM_DIR:-/tmp/zigshim}

if [ ! -x "$ZIG" ]; then
    echo "cargo-env.sh: zig not found at $ZIG" >&2
    return 1 2>/dev/null || exit 1
fi

mkdir -p "$SHIM"
for t in cc c++ ar ranlib; do
    case $t in
        cc) sub=cc ;;
        c++) sub=c++ ;;
        *) sub=$t ;;
    esac
    printf '#!/bin/sh\nexec %s %s "$@"\n' "$ZIG" "$sub" > "$SHIM/$t"
    chmod +x "$SHIM/$t"
done

PATH="$SHIM:$PATH"
export PATH
LIBCLANG_PATH="$LIBCLANG_ROOT"
export LIBCLANG_PATH

# bindgen's include set must match `zig cc -E -v -x c /dev/null` exactly --
# a partial set fails on __STD_TYPE inside stddef.h.
Z="$ZIG_ROOT/lib"
BINDGEN_EXTRA_CLANG_ARGS="-I$Z/include \
-I$Z/libc/include/x86-linux-gnu -I$Z/libc/include/generic-glibc \
-I$Z/libc/include/x86-linux-any -I$Z/libc/include/any-linux-any"
export BINDGEN_EXTRA_CLANG_ARGS