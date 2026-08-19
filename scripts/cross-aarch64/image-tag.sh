#!/bin/bash
# Print the docker tag for the cross image, derived from everything that
# determines its contents.
#
# The image bakes in an LLVM 23 built from the llvm-patches/llvm-23 series via
# configure_llvm.sh, so a change to any patch changes the compiler that builds
# the Mali test binaries. Tagging by a digest of the inputs makes a stale image
# impossible: a changed input yields a tag that does not exist locally, so the
# image is rebuilt. An unchanged one hits the cache.
#
# Usage: image-tag.sh [chipstar-root]
set -euo pipefail
ROOT="${1:-$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)}"
cd "$ROOT"
{
  # The patch series and the script that pins LLVM's refs and applies it.
  find llvm-patches/llvm-23 -type f -exec sha256sum {} + | sort -k2
  sha256sum scripts/configure_llvm.sh
  # The image recipe itself.
  sha256sum scripts/cross-aarch64/Dockerfile \
            scripts/cross-aarch64/Dockerfile.chipstar \
            scripts/cross-aarch64/image-llvm.sh \
            scripts/cross-aarch64/image-spirv-tools.sh
} | sha256sum | cut -c1-12
