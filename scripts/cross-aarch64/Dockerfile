# Ubuntu 22.04 x86_64 build host with an aarch64-linux-gnu cross toolchain.
#
# 22.04 is not incidental: its g++-aarch64-linux-gnu targets glibc 2.35 and
# ships the gcc 11 libstdc++, which is exactly salami's (ODROID-N2+, Ubuntu
# 22.04 arm64) userland. Matching it here is what lets the cross build run
# without a sysroot copied off the target.
FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake ninja-build python3 git ca-certificates \
    g++-aarch64-linux-gnu gcc-aarch64-linux-gnu binutils-aarch64-linux-gnu \
    && rm -rf /var/lib/apt/lists/*
