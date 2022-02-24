// LLVM definitions for SPIR-V
//
// (c) 2022 Henry Linjamäki / Parmance for Argonne National Laboratory
#ifndef LLVM_PASSES_LLVMSPIRV_H
#define LLVM_PASSES_LLVMSPIRV_H

// LLVM address space numbers for SPIR-V storage classes.
#define SPIRV_CROSSWORKGROUP_AS 1
#define SPIRV_UNIFORMCONSTANT_AS 2
#define SPIRV_GENERIC_AS 4

// Address space numbers for OpenCL memory regions and objects.
#define OCL_GLOBAL_AS SPIRV_CROSSWORKGROUP_AS
#define OCL_GENERIC_AS SPIRV_GENERIC_AS
#define OCL_CONSTANT_AS SPIRV_UNIFORMCONSTANT_AS
#define OCL_IMAGE_AS OCL_GLOBAL_AS
#define OCL_SAMPLER_AS OCL_CONSTANT_AS

#endif
