// PassPlugin moved from llvm/Passes/ to llvm/Plugins/ in LLVM 22.
#if LLVM_VERSION_MAJOR >= 22
#include "llvm/Plugins/PassPlugin.h"
#else
#include "llvm/Passes/PassPlugin.h"
#endif
