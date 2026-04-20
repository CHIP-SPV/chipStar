#!/bin/bash
# Test that bool kernel parameters are promoted away from OpTypeBool in SPIR-V.
# Per the OpenCL SPIR-V environment spec, kernel entry point parameters
# must not use OpTypeBool. Using OpTypeBool causes CL_INVALID_ARG_SIZE
# on conformant OpenCL implementations (e.g. rusticl).
#
# This test compiles a HIP kernel with bool parameters and checks the
# generated SPIR-V to ensure no kernel entry point has OpTypeBool params.

set -e

HIPCC="@CMAKE_BINARY_DIR@/bin/hipcc"
SPIRV_DIS="@SPIRV_DIS@"
SRC="@CMAKE_CURRENT_SOURCE_DIR@/TestBoolKernelParam.hip"
WORKDIR=$(mktemp -d)
trap "rm -rf $WORKDIR" EXIT

if [ ! -x "$SPIRV_DIS" ]; then
  echo "HIP_SKIP_THIS_TEST: spirv-dis not found at $SPIRV_DIS"
  exit 0
fi

# Compile with --save-temps to get the .out SPIR-V file
cd "$WORKDIR"
"$HIPCC" --save-temps "$SRC" -o test_bool 2>/dev/null

# Find the SPIR-V binary (native backend produces *.spv, translator produces *.out)
SPV_FILE=$(ls *.spv 2>/dev/null | head -1)
if [ -z "$SPV_FILE" ]; then
  SPV_FILE=$(ls *.out 2>/dev/null | head -1)
fi
if [ -z "$SPV_FILE" ]; then
  echo "FAIL: No SPIR-V file produced by hipcc --save-temps"
  exit 1
fi

# Disassemble
DISASM=$("$SPIRV_DIS" "$SPV_FILE" 2>/dev/null)

# Extract user kernel entry point function IDs
KERNEL_IDS=$(echo "$DISASM" | grep "OpEntryPoint Kernel" | \
  grep -v "__chip_" | \
  sed 's/.*Kernel \(%[0-9]*\).*/\1/')

if [ -z "$KERNEL_IDS" ]; then
  echo "FAIL: No user kernel entry points found in SPIR-V"
  exit 1
fi

# Write disassembly to a file so we can grep it reliably
DISASM_FILE="$WORKDIR/disasm.txt"
echo "$DISASM" > "$DISASM_FILE"

# For each kernel entry point, extract its parameters and check for %bool
for KID in $KERNEL_IDS; do
  # Find the line number of this function definition
  FUNC_LINE=$(grep -n "^[[:space:]]*${KID} = OpFunction" "$DISASM_FILE" | head -1 | cut -d: -f1)
  if [ -z "$FUNC_LINE" ]; then
    continue
  fi

  # Extract lines after the OpFunction until OpLabel (the parameters)
  PARAMS=$(tail -n +"$((FUNC_LINE + 1))" "$DISASM_FILE" | \
    sed '/OpLabel/q' | grep "OpFunctionParameter" || true)

  if echo "$PARAMS" | grep -q "OpFunctionParameter %bool"; then
    KERNEL_NAME=$(grep "OpEntryPoint Kernel ${KID} " "$DISASM_FILE" | \
      sed 's/.*"\(.*\)".*/\1/')
    echo "FAIL: Kernel \"$KERNEL_NAME\" ($KID) has OpTypeBool parameter(s)."
    echo "OpenCL SPIR-V environment spec forbids OpTypeBool in kernel args."
    echo "$PARAMS" | grep "OpFunctionParameter %bool"
    exit 1
  fi
done

echo "PASS: No kernel entry points have OpTypeBool parameters"
