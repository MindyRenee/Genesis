#!/bin/sh
# Rust build environment for hosts with no system C toolchain.
#
# Genesis needs `cc` (to link) and `libclang` (for v4l2-sys-mit's
# bindgen). Where no system compiler exists, Zig ships a working clang;
# this shims `cc`/`c++`/`ar`/`ranlib` to it and points bindgen at the
# headers zig's clang actually searches.
#
# Usage:  . ./scripts/cargo-env.sh && cargo test --release
#
# Everything is discovered rather than assumed: $ZIG_ROOT/zig if
# executable, else `zig` on PATH; the bindgen include set parsed from
# `zig cc -E -v -x c /dev/null` (the <...> search list); libclang from
# a scan of candidate directories. Set ZIG_ROOT / LIBCLANG_ROOT to
# force the roots. See the Toolchain section of AGENTS.md.
ZIG_ROOT=${ZIG_ROOT:-$HOME/tools/zig-x86_64-linux-0.16.0}
LIBCLANG_ROOT=${LIBCLANG_ROOT:-}
ZIG=$ZIG_ROOT/zig
SHIM=${SHIM_DIR:-/tmp/zigshim}

# Fall back to a zig on PATH (e.g. a distro package) when the tarball
# root is absent.
[ -x "$ZIG" ] || ZIG=$(command -v zig 2>/dev/null || true)
if [ -z "$ZIG" ] || [ ! -x "$ZIG" ]; then
    echo "cargo-env.sh: zig not found at $ZIG_ROOT/zig or on PATH" >&2
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

# libclang: honour LIBCLANG_ROOT, else take the first candidate dir
# holding the C library. libclang-cpp must not match -- it lacks the
# C API bindgen binds.
has_libclang() {
    for f in "$1"/libclang.so* "$1"/libclang-[0-9]*.so*; do
        [ -e "$f" ] && return 0
    done
    return 1
}
if [ -z "$LIBCLANG_ROOT" ]; then
    for d in "$HOME/tools/libclang/clang/native" \
             /usr/lib/llvm-*/lib /usr/lib/*-linux-gnu /usr/lib64 /usr/lib; do
        if has_libclang "$d"; then LIBCLANG_ROOT=$d; break; fi
    done
fi
if [ -n "$LIBCLANG_ROOT" ]; then
    LIBCLANG_PATH=$LIBCLANG_ROOT
    export LIBCLANG_PATH
fi

# bindgen's include set must match `zig cc -E -v -x c /dev/null`
# exactly -- a partial set fails on __STD_TYPE inside stddef.h. A
# tarball zig lists its bundled libc headers ($ZIG_ROOT/lib/libc/
# include/<arch>-linux-{gnu,any}, generic-glibc, any-linux-any); a
# distro zig lists the system headers instead. Parsing the real
# search list handles both, on any arch.
BINDGEN_EXTRA_CLANG_ARGS=$($ZIG cc -E -v -x c /dev/null 2>&1 \
    | sed -n '/#include <\.\.\.> search starts here:/,/End of search list/p' \
    | sed -n 's|^ *\(/[^ ]*\).*|-I\1|p' \
    | tr '\n' ' ')
if [ -z "$BINDGEN_EXTRA_CLANG_ARGS" ]; then
    Z=$($ZIG env 2>/dev/null | sed -n 's/.*"lib_dir": *"\([^"]*\)".*/\1/p')
    [ -d "$Z" ] || Z=$ZIG_ROOT/lib
    BINDGEN_EXTRA_CLANG_ARGS="-I$Z/include"
    echo "cargo-env.sh: could not parse zig's include list; using $BINDGEN_EXTRA_CLANG_ARGS" >&2
fi
export BINDGEN_EXTRA_CLANG_ARGS
