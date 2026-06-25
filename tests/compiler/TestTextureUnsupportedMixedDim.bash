#!/usr/bin/env bash
set -u
out=/tmp/TestTextureUnsupportedMixedDim.$$.out
trap 'rm -f "$out"' EXIT

@CMAKE_BINARY_DIR@/bin/hipcc -c \
  @CMAKE_CURRENT_SOURCE_DIR@/TestTextureUnsupportedMixedDim.hip \
  -o /dev/null > "$out" 2>&1
rc=$?
cat "$out"

test $rc -ne 0
! grep -Eq "UNREACHABLE|Assertion .* failed|PLEASE submit a bug report" "$out"
grep -q "unsupported HIP texture object use" "$out"
