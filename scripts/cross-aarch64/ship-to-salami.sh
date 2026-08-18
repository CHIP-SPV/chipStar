#!/bin/bash
# Ship a cross-built chipStar tree to salami and make it runnable there.
#
# Usage: ship-to-salami.sh <build-dir> <sha> [host]
#
# Three things have to change between the build machine and the target:
#   1. Every CTestTestfile.cmake bakes the builder's absolute paths
#      (/work/cross-<sha>/...). Rewrite them to the target prefix.
#   2. Tests that invoke a compiler at test time (hipcc, cucc, opt, the
#      compile-only tests under tests/compiler, the .bash tests that shell
#      out to hipcc) cannot run on the target: the toolchain in the tree is
#      x86, and a compiler that ran on aarch64 would test the aarch64 host
#      compiler, not Mali. Every one of them already runs on the x86 gate
#      with a working hipcc. Keep only tests that execute a prebuilt binary,
#      identified by their command shape rather than by name so a new test
#      is classified by what it does.
#   3. .o files and CMakeFiles are build intermediates; leave them behind.
set -e
BUILD="${1:?build dir}"; SHA="${2:?sha}"; HOST="${3:-salami}"
PREFIX="/home/pvelesko/ci-stage/$SHA"
# The paths baked into the tree are the CONTAINER's view of the build dir
# (/work/cross-<sha>), not this host's mount of it, so read them from the
# cache rather than from $PWD.
BUILDER_PREFIX="$(sed -n 's|^CMAKE_CACHEFILE_DIR:INTERNAL=||p' "$BUILD/CMakeCache.txt")"

STAGE="$(mktemp -d)"
rsync -a --exclude='*.o' --exclude='CMakeFiles' --exclude='*.ninja' "$BUILD/" "$STAGE/"

# 1. relocate
grep -rl "$BUILDER_PREFIX" "$STAGE" --include=CTestTestfile.cmake \
  | xargs -r sed -i "s|$BUILDER_PREFIX|$PREFIX|g"
# the source tree path is baked too (tests reference $SRC files by absolute path)
SRC_PREFIX="$(sed -n 's|^CMAKE_HOME_DIRECTORY:INTERNAL=||p' "$BUILD/CMakeCache.txt")"
grep -rl "$SRC_PREFIX" "$STAGE" --include=CTestTestfile.cmake \
  | xargs -r sed -i "s|$SRC_PREFIX|$PREFIX/src|g"

# 2. keep only tests that run a prebuilt binary through the doubles wrapper
#    (or a bare binary under tests/ or samples/). Everything else is a
#    toolchain invocation and is dropped from the target's test set.
python3 - "$STAGE" "$PREFIX" <<'PY'
import re, sys, pathlib
stage, prefix = sys.argv[1], sys.argv[2]
keep_re = re.compile(
    r'^add_test\(\[=\[([^\]]+)\]=\] "(?:%s/bin/spirv-extractor" "--check-for-doubles" ")?%s/(?:tests|samples)/[^"]+"' % (re.escape(prefix), re.escape(prefix)))
kept = dropped = 0
for f in pathlib.Path(stage).rglob('CTestTestfile.cmake'):
    out = []
    lines = f.read_text().splitlines()
    i = 0
    while i < len(lines):
        l = lines[i]
        if l.startswith('add_test('):
            name = re.match(r'add_test\(\[=\[([^\]]+)\]=\]', l).group(1)
            if keep_re.match(l):
                out.append(l); kept += 1
                # keep its set_tests_properties line(s)
                while i + 1 < len(lines) and lines[i+1].startswith('set_tests_properties('):
                    i += 1; out.append(lines[i])
            else:
                dropped += 1
                while i + 1 < len(lines) and lines[i+1].startswith('set_tests_properties('):
                    i += 1
        else:
            out.append(l)
        i += 1
    f.write_text('\n'.join(out) + '\n')
print(f"tests kept={kept} dropped(toolchain-invoking)={dropped}")
PY

# 3. ship, plus the source tree the surviving tests reference (test inputs)
ssh "$HOST" "mkdir -p '$PREFIX'"
rsync -a --delete "$STAGE/" "$HOST:$PREFIX/"
rsync -a --delete --exclude='.git' "$(dirname "$BUILD")/src-$SHA/" "$HOST:$PREFIX/src/"
rm -rf "$STAGE"
ssh "$HOST" "cd '$PREFIX' && LD_LIBRARY_PATH=\$PWD ./hipInfo 2>/dev/null | grep -m1 'Name:'"
echo "SHIP-OK $HOST:$PREFIX"
