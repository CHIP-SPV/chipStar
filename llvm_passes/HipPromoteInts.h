#ifndef HIP_PROMOTE_INTS_H
#define HIP_PROMOTE_INTS_H

#include "llvm/IR/PassManager.h"
#include "llvm/IR/Module.h"

namespace llvm {

class HipPromoteIntsPass : public PassInfoMixin<HipPromoteIntsPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
  
  // Promote a non-standard integer type to the next larger standard size
  static unsigned getPromotedBitWidth(unsigned Original);
  
  // Check if the given bit width is a standard size (8, 16, 32, 64)
  static bool isStandardBitWidth(unsigned BitWidth);

  // Get the promoted type for a given type
  static Type* getPromotedType(Type* TypeToPromote);
};

} // namespace llvm

#endif // HIP_PROMOTE_INTS_H
