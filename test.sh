#/space/pvelesko/install/llvm/18.0/bin/opt /space/pvelesko/chipStar/rocThrust/expand-hip-spirv64-generic-link.bc -load-pass-plugin /space/pvelesko/install/HIP/chipStar/test/lib/llvm/libLLVMHipSpvPasses.so -passes=hip-post-link-passes -o expand-hip-spirv64-generic-lower.bc
# /space/pvelesko/install/llvm/18.0/bin/opt /space/pvelesko/chipStar/rocThrust/scan_matrix_by_rows-hip-spirv64-generic-link.bc -load-pass-plugin /space/pvelesko/install/HIP/chipStar/test/lib/llvm/libLLVMHipSpvPasses.so -passes=hip-post-link-passes -o scan_matrix_by_rows-hip-spirv64-generic-opt.bc
# /space/pvelesko/install/llvm/18.0/bin/llvm-spirv  --spirv-ext=-all,+SPV_INTEL_subgroups  scan_matrix_by_rows-hip-spirv64-generic-opt.bc -o scan_matrix_by_rows-hip-spirv64-generic-opt.spv

# IR=~/chipStar/fix-promote-int-pass/small-i24-repro.ll
# llvm-as $IR
# /space/pvelesko/install/llvm/18.0/bin/opt -load-pass-plugin /space/pvelesko/chipStar/fix-promote-int-pass/build/install/lib/llvm/libLLVMHipSpvPasses.so -passes=hip-post-link-passes $IR -o $IR.bc -debug -debug-only=hip-promote-ints
# /space/pvelesko/install/llvm/18.0/bin/llvm-spirv $IR.bc -o $IR.spv

# IR=~/chipStar/fix-promote-int-pass/ext-repro.ll
# llvm-as $IR
# /space/pvelesko/install/llvm/18.0/bin/opt -load-pass-plugin /space/pvelesko/chipStar/fix-promote-int-pass/build/lib/libLLVMHipSpvPasses.so -passes=hip-post-link-passes $IR -o $IR.bc -debug -debug-only=hip-promote-ints
# /space/pvelesko/install/llvm/18.0/bin/llvm-spirv $IR.bc -o $IR.spv

# IR=~/chipStar/fix-promote-int-pass/func-return-nonstd.ll
# llvm-as $IR
# /space/pvelesko/install/llvm/18.0/bin/opt -load-pass-plugin /space/pvelesko/chipStar/fix-promote-int-pass/build/lib/libLLVMHipSpvPasses.so -passes=hip-post-link-passes $IR -o $IR.bc -debug -debug-only=hip-promote-ints
# /space/pvelesko/install/llvm/18.0/bin/llvm-spirv $IR.bc -o $IR.spv


#IR=~/chipStar/fix-promote-int-pass/dominance.ll
#llvm-as $IR
#/space/pvelesko/install/llvm/18.0/bin/opt -load-pass-plugin /space/pvelesko/chipStar/fix-promote-int-pass/build/lib/libLLVMHipSpvPasses.so -passes=hip-post-link-passes $IR -o $IR.bc -debug -debug-only=hip-promote-ints 
#/space/pvelesko/install/llvm/18.0/bin/llvm-spirv $IR.bc -o $IR.spv

#IR=~/chipStar/fix-promote-int-pass/illegalZext.ll
#llvm-as $IR
#/space/pvelesko/install/llvm/18.0/bin/opt -load-pass-plugin /space/pvelesko/chipStar/fix-promote-int-pass/build/lib/libLLVMHipSpvPasses.so -passes=hip-post-link-passes $IR -o $IR.bc -debug -debug-only=hip-promote-ints 
#/space/pvelesko/install/llvm/18.0/bin/llvm-spirv $IR.bc -o $IR.spv

IR=~/chipStar/fix-promote-int-pass/TestHipccInvalidBitWidth-hip-spirv64-generic.ll
llvm-as $IR
/space/pvelesko/install/llvm/18.0/bin/opt -load-pass-plugin /space/pvelesko/chipStar/fix-promote-int-pass/build/lib/libLLVMHipSpvPasses.so -passes=hip-post-link-passes $IR -o $IR.bc -debug -debug-only=hip-promote-ints 
/space/pvelesko/install/llvm/18.0/bin/llvm-spirv $IR.bc -o $IR.spv