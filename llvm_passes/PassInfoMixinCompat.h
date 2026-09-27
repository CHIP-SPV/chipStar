// LLVM 24 hides PassInfoMixin in llvm::detail; passes pick Required/Optional.
#ifndef LLVM_PASSES_PASS_INFO_MIXIN_COMPAT_H
#define LLVM_PASSES_PASS_INFO_MIXIN_COMPAT_H

#include "llvm/Config/llvm-config.h"
#include "llvm/IR/PassManager.h"

#if LLVM_VERSION_MAJOR >= 24
template <typename T>
using HipRequiredPassInfoMixin = llvm::RequiredPassInfoMixin<T>;
template <typename T>
using HipOptionalPassInfoMixin = llvm::OptionalPassInfoMixin<T>;
#else
template <typename T> using HipRequiredPassInfoMixin = llvm::PassInfoMixin<T>;
template <typename T> using HipOptionalPassInfoMixin = llvm::PassInfoMixin<T>;
#endif

#endif
