#!/bin/sh
# Usage: llvm_stage_fingerprint.sh <llvm-version>
# Git object ids of the committed inputs configure_llvm.sh builds LLVM <version> from.
git -C "$(dirname "$0")" rev-parse HEAD:scripts/configure_llvm.sh "HEAD:llvm-patches/llvm-$1"
