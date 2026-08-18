# Ubuntu 22.04 x86_64 build host with an aarch64-linux-gnu cross toolchain.
#
# 22.04 is not incidental: its g++-aarch64-linux-gnu targets glibc 2.35 and
# ships the gcc 11 libstdc++, which is exactly salami's (ODROID-N2+, Ubuntu
# 22.04 arm64) userland. Matching it here is what lets the cross build run
# without a sysroot copied off the target.
#
# Serves two cross builds:
#   cross-build.sh     the aarch64-hosted LLVM 23 that runs ON salami
#   cross-chipstar.sh  chipStar + its tests, built on x86 FOR salami
# The second needs an x86-hosted clang that can emit aarch64, which the
# LLVM 23 on the runners cannot (targets-built "SPIRV X86"). That toolchain
# is built into the image by image-llvm.sh, once, and cached.
FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake ninja-build python3 git ca-certificates rsync \
    g++-aarch64-linux-gnu gcc-aarch64-linux-gnu binutils-aarch64-linux-gnu \
    libssl-dev pkg-config python3-yaml xxd \
    && rm -rf /var/lib/apt/lists/*
