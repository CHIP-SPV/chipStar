#!/bin/bash
# Copyright (c) 2023 chipStar developers
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

# USAGE: $0 NAME INPUT OUTPUT_SOURCE OUTPUT_HEADER
set -eu

NAME=${1:-"Misses name argument!"}
INPUT=${2:-"Misses input file argument!"}
OUTPUT_SOURCE=${3:-"Misses output source file argument!"}
OUTPUT_HEADER=${4:-"Misses output header file argument!"}

SIZE=$(wc -c < "${INPUT}")

{
    cat <<CPPSOURCE
// This file is auto-generated!
#include <array>
#include <cstddef>

namespace chipstar {

extern const std::array<unsigned char, ${SIZE}> ${NAME} = {
CPPSOURCE

    xxd --include - < "$INPUT"

    cat <<CPPSOURCE
};
} // namespace chipstar
CPPSOURCE
} > "$OUTPUT_SOURCE"


{
    cat <<CPPSOURCE
// This file is auto-generated!
#ifndef CHIP_EMBEDDED_BINARY_${NAME}_H
#define CHIP_EMBEDDED_BINARY_${NAME}_H
#include <array>
#include <cstddef>

namespace chipstar {

extern const std::array<unsigned char, ${SIZE}> ${NAME};

} // namespace chipstar
#endif // CHIP_EMBEDDED_BINARY_${NAME}_H
CPPSOURCE
} > "$OUTPUT_HEADER"
