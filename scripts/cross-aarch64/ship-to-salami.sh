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
#      A test can also reach the compiler without its ctest command showing
#      it, by calling hipRTC in process, and those look exactly like a plain
#      prebuilt binary. They are classified the same way, by behaviour: a
#      binary that imports a hiprtc* symbol compiles at run time, so it is
#      dropped too. What that costs the target is the device execution of a
#      kernel hipRTC produced; the compilation itself is host side and is
#      covered on the x86 gate.
#   3. .o files and CMakeFiles are build intermediates; leave them behind.
set -e
BUILD="${1:?build dir}"; SHA="${2:?sha}"; HOST="${3:-salami}"
# The ELF reader used to classify test binaries. Named here rather than looked
# up inside the filter so the tool this depends on is visible at the top, and
# overridable when binutils lives somewhere else.
case "${READELF:=/usr/bin/readelf}" in
  /*) ;;
  *) echo "ship-to-salami: READELF must be an absolute path, got '$READELF'" >&2; exit 1 ;;
esac
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
python3 - "$STAGE" "$PREFIX" "$READELF" <<'PY'
import re, subprocess, sys, pathlib
stage, prefix, readelf = sys.argv[1], sys.argv[2], sys.argv[3]
keep_re = re.compile(
    r'^add_test\(\[=\[([^\]]+)\]=\] "(?:%s/bin/spirv-extractor" "--check-for-doubles" ")?%s/(?:tests|samples)/[^"]+"' % (re.escape(prefix), re.escape(prefix)))
# The FIRST argument under the target prefix is the test executable. The
# doubles wrapper that may precede it lives under <prefix>/bin, which this does
# not match, and anything after the executable is its own argument, which may
# well be an input file under tests/.
bin_re = re.compile(r'"(%s/(?:tests|samples)/[^"]+)"' % re.escape(prefix))

def imports_hiprtc(binary):
    """True when the binary imports a hipRTC entry point, so it compiles when
    it runs.

    readelf rather than nm: it reads the ELF's own tables, so it works on the
    aarch64 binaries from the x86 builder whatever targets the local binutils
    was configured for. Undefined symbols only: a binary that merely defines
    something with hiprtc in the name does not call the runtime compiler."""
    out = subprocess.run([readelf, "-sW", "--dyn-syms", str(binary)],
                         capture_output=True, text=True, timeout=120)
    if out.returncode != 0:
        raise RuntimeError("readelf failed on %s: %s"
                           % (binary, out.stderr.strip()[:200]))
    for ln in out.stdout.splitlines():
        f = ln.split()
        # Num: Value Size Type Bind Vis Ndx Name
        # The name may carry a version suffix, as in hiprtcVersion@@HIP_1.0.
        if len(f) >= 8 and f[6] == "UND" and \
           f[7].split("@")[0].startswith("hiprtc"):
            return True
    return False

def compiles_at_runtime(line):
    """True when this add_test line runs a binary that compiles at run time.

    Raises rather than returning False when the question cannot be answered:
    silently keeping a test would ship exactly what this filter exists to keep
    off the target."""
    m = bin_re.findall(line)
    if not m:
        return False
    staged = pathlib.Path(stage) / pathlib.Path(m[0]).relative_to(prefix)
    # No existence probe: readelf raises if the binary is not there, and that
    # propagates, because shipping a test set this filter could not classify is
    # the failure it exists to prevent.
    return imports_hiprtc(staged)

kept = dropped = dropped_rtc = 0
for f in pathlib.Path(stage).rglob('CTestTestfile.cmake'):
    out = []
    lines = f.read_text().splitlines()
    i = 0
    while i < len(lines):
        l = lines[i]
        if l.startswith('add_test('):
            name = re.match(r'add_test\(\[=\[([^\]]+)\]=\]', l).group(1)
            if keep_re.match(l) and compiles_at_runtime(l):
                dropped += 1; dropped_rtc += 1
                while i + 1 < len(lines) and lines[i+1].startswith('set_tests_properties('):
                    i += 1
            elif keep_re.match(l):
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
print(f"tests kept={kept} dropped(toolchain-invoking)={dropped} of which runtime-compiling={dropped_rtc}")
PY

# 3. ship, plus the source tree the surviving tests reference (test inputs)
ssh "$HOST" "mkdir -p '$PREFIX'"
rsync -a --delete "$STAGE/" "$HOST:$PREFIX/"
rsync -a --delete --exclude='.git' "$(dirname "$BUILD")/src-$SHA/" "$HOST:$PREFIX/src/"
rm -rf "$STAGE"
ssh "$HOST" "cd '$PREFIX' && LD_LIBRARY_PATH=\$PWD ./hipInfo 2>/dev/null | grep -m1 'Name:'"
echo "SHIP-OK $HOST:$PREFIX"
