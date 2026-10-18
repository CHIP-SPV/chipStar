#!/bin/sh
# Usage: llvm_stage_fingerprint.sh <llvm-version>
# Git object ids of the committed inputs (configure_llvm.sh, the patches, the workflow that passes it build options) LLVM <version> is built from.
git -C "$(dirname "$0")" rev-parse HEAD:scripts/configure_llvm.sh "HEAD:llvm-patches/llvm-$1" HEAD:.github/workflows/test-llvm-patches.yml
