// LLVM 24 hides PassInfoMixin in llvm::detail. Every chipStar pass is a
// lowering that must run.
#ifndef LLVM_PASSES_PASS_INFO_MIXIN_COMPAT_H
#define LLVM_PASSES_PASS_INFO_MIXIN_COMPAT_H

#include "llvm/Config/llvm-config.h"
#include "llvm/IR/PassManager.h"

#if LLVM_VERSION_MAJOR >= 24
template <typename T> using HipPassInfoMixin = llvm::RequiredPassInfoMixin<T>;
#else
template <typename T> using HipPassInfoMixin = llvm::PassInfoMixin<T>;
#endif

#endif
