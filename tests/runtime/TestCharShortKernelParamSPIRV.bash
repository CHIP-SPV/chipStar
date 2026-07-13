#!/bin/bash
# Regression test for issue #1261: char (i8) and short (i16) kernel parameters
# are valid SPIR-V OpTypeInt widths and must NOT be promoted to i32. This test
# compiles a HIP kernel with char/short parameters and checks the generated
# SPIR-V to ensure the narrow integer parameters are preserved.

set -e

HIPCC="@CMAKE_BINARY_DIR@/bin/hipcc"
SPIRV_DIS="@SPIRV_DIS@"
SRC="@CMAKE_CURRENT_SOURCE_DIR@/TestCharShortKernelParam.hip"
WORKDIR=$(mktemp -d)
trap "rm -rf $WORKDIR" EXIT

if [ ! -x "$SPIRV_DIS" ]; then
  echo "HIP_SKIP_THIS_TEST: spirv-dis not found at $SPIRV_DIS"
  exit 0
fi

cd "$WORKDIR"

if [ "@USE_NEW_OFFLOAD_DRIVER@" = "ON" ]; then
    "$HIPCC" --offload-device-only -c "$SRC" -o device.o 2>/dev/null
    "@CLANG_OFFLOAD_BUNDLER@" -unbundle --type=o \
                              --targets=hip-@OFFLOAD_TRIPLE@--generic \
                              --inputs=device.o --output=spv_binary.out
    SPV_FILE=spv_binary.out
else
    "$HIPCC" --save-temps "$SRC" -o test_charshort 2>/dev/null
    SPV_FILE=$(ls *.out 2>/dev/null | head -1)
fi

if [ -z "$SPV_FILE" ]; then
  echo "FAIL: No .out SPIR-V file produced by hipcc --save-temps"
  exit 1
fi

DISASM=$("$SPIRV_DIS" "$SPV_FILE" 2>/dev/null)
DISASM_FILE="$WORKDIR/disasm.txt"
echo "$DISASM" > "$DISASM_FILE"

# The charKernel entry point should have an 8-bit integer parameter and
# shortKernel a 16-bit integer parameter, i.e. they were not promoted.
# Grab the type ids for the 8- and 16-bit integer types.
I8_TYPE=$(grep -oE "%[a-zA-Z0-9_]+ = OpTypeInt 8 0" "$DISASM_FILE" | \
  head -1 | awk '{print $1}')
I16_TYPE=$(grep -oE "%[a-zA-Z0-9_]+ = OpTypeInt 16 0" "$DISASM_FILE" | \
  head -1 | awk '{print $1}')

if [ -z "$I8_TYPE" ]; then
  echo "FAIL: No 8-bit integer type (OpTypeInt 8 0) found; char param was promoted."
  exit 1
fi
if [ -z "$I16_TYPE" ]; then
  echo "FAIL: No 16-bit integer type (OpTypeInt 16 0) found; short param was promoted."
  exit 1
fi

# Ensure at least one kernel entry point actually takes the narrow type as a
# function parameter (not only used internally).
if ! grep -q "OpFunctionParameter ${I8_TYPE}\b" "$DISASM_FILE"; then
  echo "FAIL: No kernel takes an 8-bit integer parameter; char param was promoted."
  exit 1
fi
if ! grep -q "OpFunctionParameter ${I16_TYPE}\b" "$DISASM_FILE"; then
  echo "FAIL: No kernel takes a 16-bit integer parameter; short param was promoted."
  exit 1
fi

echo "PASS: char/short kernel parameters preserved as i8/i16 (not promoted)"
