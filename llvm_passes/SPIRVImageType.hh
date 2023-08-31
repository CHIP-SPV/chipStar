
enum ImageAccessQualifier : unsigned { AQ_ro = 0, AQ_wo = 1, AQ_rw = 2 };

namespace llvm {
  class Type;
  class LLVMContext;
  class StringRef;
}


/// Construct a SPIR-V target extension type for the given OpenCL image type.
llvm::Type *getSPIRVImageType(llvm::LLVMContext &Ctx, llvm::StringRef BaseType,
                                     llvm::StringRef OpenCLName,
                                     unsigned ImageAccessQualifier);
