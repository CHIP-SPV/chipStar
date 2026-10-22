#!/bin/sh
# usage: build.sh <outdir> [cc]     cc defaults to the aarch64 cross gcc
set -e
out="$1"; cc="${2:-aarch64-linux-gnu-gcc}"; mkdir -p "$out"
here="$(dirname "$(readlink -f "$0")")"
sed 's/.*/STUB(&)/' "$here/cl-symbols.txt" > "$out/cl-symbols.h"
"$cc" -shared -nostdlib -I"$out" -o "$out/libOpenCL.so.new" -Wl,-soname,libmali.so.0 "$here/libmali-stub.c"
# keep the old mtime when unchanged so ninja does not relink libCHIP.so
cmp -s "$out/libOpenCL.so.new" "$out/libOpenCL.so" && rm "$out/libOpenCL.so.new" || mv "$out/libOpenCL.so.new" "$out/libOpenCL.so"
ln -sf libOpenCL.so "$out/libmali.so.0"
