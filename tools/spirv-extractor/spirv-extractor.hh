#include <algorithm>
#include <cstring>
#include <elf.h>
#include <filesystem>
#include <fstream>
#include <hip/hip_fatbin.h>
#include <iostream>
#include <memory>
#include <spirv-tools/libspirv.hpp>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

// SPIR-V magic number (0x07230203)
const uint32_t SPIRV_MAGIC = 0x07230203;

enum class BinaryType { UNKNOWN, BUNDLED, SPIRV_ONLY };

struct MagicResult {
  void *ptr;
  BinaryType type;
};

template <class T>
static inline T _copyAs(const void *BaseAddr, size_t ByteOffset = 0) {
  T Res;
  std::memcpy(&Res, (const char *)BaseAddr + ByteOffset, sizeof(T));
  return Res;
}

// Helper function to find the .hip_fatbin section in an ELF file
std::pair<const void *, size_t> findHipFatbinSection(const void *data,
                                                     size_t size) {
  const Elf64_Ehdr *ehdr = static_cast<const Elf64_Ehdr *>(data);

  // Verify ELF magic
  if (size < sizeof(Elf64_Ehdr) ||
      memcmp(ehdr->e_ident, ELFMAG, SELFMAG) != 0) {
    return {nullptr, 0};
  }

  // Get section headers
  const Elf64_Shdr *shdr = reinterpret_cast<const Elf64_Shdr *>(
      static_cast<const char *>(data) + ehdr->e_shoff);

  // Get section names string table
  const char *strtab =
      static_cast<const char *>(data) + shdr[ehdr->e_shstrndx].sh_offset;

  // Find .hip_fatbin section
  for (size_t i = 0; i < ehdr->e_shnum; i++) {
    const char *name = strtab + shdr[i].sh_name;
    if (strcmp(name, ".hip_fatbin") == 0) {
      return {static_cast<const char *>(data) + shdr[i].sh_offset,
              shdr[i].sh_size};
    }
  }

  return {nullptr, 0};
}

MagicResult seekToMagic(const void *Bundle) {
  // First check if this is an ELF file
  const Elf64_Ehdr *ehdr = static_cast<const Elf64_Ehdr *>(Bundle);

  // Check if this looks like an ELF file (at least has the magic bytes)
  if (memcmp(ehdr->e_ident, ELFMAG, SELFMAG) == 0) {
    // Get section headers
    const Elf64_Shdr *shdr = reinterpret_cast<const Elf64_Shdr *>(
        static_cast<const char *>(Bundle) + ehdr->e_shoff);

    // Get section names string table
    const char *strtab =
        static_cast<const char *>(Bundle) + shdr[ehdr->e_shstrndx].sh_offset;

    // Find .hip_fatbin section
    for (size_t i = 0; i < ehdr->e_shnum; i++) {
      const char *name = strtab + shdr[i].sh_name;
      if (strcmp(name, ".hip_fatbin") == 0) {
        const char *data =
            static_cast<const char *>(Bundle) + shdr[i].sh_offset;
        size_t size = shdr[i].sh_size;

        // Search for magic identifiers in the section
        for (size_t j = 0; j < size - sizeof(CLANG_OFFLOAD_BUNDLER_MAGIC) + 1;
             ++j) {
          if (std::memcmp(data + j, CLANG_OFFLOAD_BUNDLER_MAGIC,
                          sizeof(CLANG_OFFLOAD_BUNDLER_MAGIC) - 1) == 0) {
            return {const_cast<void *>(static_cast<const void *>(data + j)),
                    BinaryType::BUNDLED};
          }
        }

        for (size_t j = 0; j < size - sizeof(uint32_t); ++j) {
          uint32_t potential_magic;
          std::memcpy(&potential_magic, data + j, sizeof(uint32_t));
          if (potential_magic == SPIRV_MAGIC) {
            return {const_cast<void *>(static_cast<const void *>(data + j)),
                    BinaryType::SPIRV_ONLY};
          }
        }
        break;
      }
    }
  }

  // Not an ELF file or no .hip_fatbin section found, try scanning the raw data
  const char *data = static_cast<const char *>(Bundle);
  constexpr size_t MAX_SCAN_SIZE = 1024 * 1024; // 1MB scan limit for safety

  for (size_t i = 0;
       i < MAX_SCAN_SIZE - sizeof(CLANG_OFFLOAD_BUNDLER_MAGIC) + 1; ++i) {
    if (std::memcmp(data + i, CLANG_OFFLOAD_BUNDLER_MAGIC,
                    sizeof(CLANG_OFFLOAD_BUNDLER_MAGIC) - 1) == 0) {
      return {const_cast<void *>(static_cast<const void *>(data + i)),
              BinaryType::BUNDLED};
    }
  }

  for (size_t i = 0; i < MAX_SCAN_SIZE - sizeof(uint32_t); ++i) {
    uint32_t potential_magic;
    std::memcpy(&potential_magic, data + i, sizeof(uint32_t));
    if (potential_magic == SPIRV_MAGIC) {
      return {const_cast<void *>(static_cast<const void *>(data + i)),
              BinaryType::SPIRV_ONLY};
    }
  }

  return {nullptr, BinaryType::UNKNOWN};
}

std::string_view extractSPIRVModule(const void *Bundle, std::string &ErrorMsg) {
  // Use seekToMagic to find the start of the bundle or SPIR-V
  auto magicResult = seekToMagic(Bundle);
  if (!magicResult.ptr) {
    ErrorMsg = "Could not find CLANG_OFFLOAD_BUNDLER_MAGIC or SPIR-V magic "
               "number in the binary";
    return std::string_view();
  }

  // If it's a direct SPIR-V binary, return it
  if (magicResult.type == BinaryType::SPIRV_ONLY) {
    // Get the size by reading the SPIR-V header
    // SPIR-V header is 5 words (20 bytes): Magic, Version, Generator, Bound,
    // Reserved
    const uint32_t *words = static_cast<const uint32_t *>(magicResult.ptr);
    size_t size = 0;
    // Scan through the SPIR-V binary to find its size
    // Each instruction's length is encoded in its first word
    size_t pos = 5;         // Start after header
    while (pos < 1000000) { // Reasonable upper limit to prevent infinite loop
      uint16_t wordCount = words[pos] >> 16;
      if (wordCount == 0)
        break;
      pos += wordCount;
    }
    size = pos * sizeof(uint32_t);
    return std::string_view(static_cast<const char *>(magicResult.ptr), size);
  }

  // Otherwise handle as bundled binary
  Bundle = magicResult.ptr;
  std::string Magic(reinterpret_cast<const char *>(Bundle),
                    sizeof(CLANG_OFFLOAD_BUNDLER_MAGIC) - 1);
  if (Magic != CLANG_OFFLOAD_BUNDLER_MAGIC) {
    ErrorMsg = "The bundled binaries are not Clang bundled "
               "(CLANG_OFFLOAD_BUNDLER_MAGIC is missing)";
    return std::string_view();
  }

  using HeaderT = __ClangOffloadBundleHeader;
  using EntryT = __ClangOffloadBundleDesc;
  const auto *Header = (const char *)Bundle;
  auto NumBundles = _copyAs<uint64_t>(Header, offsetof(HeaderT, numBundles));

  // std::cout << "Number of bundles: " << NumBundles << std::endl;

  const char *Desc = Header + offsetof(HeaderT, desc);
  for (size_t i = 0; i < NumBundles; i++) {
    auto Offset = _copyAs<uint64_t>(Desc, offsetof(EntryT, offset));
    auto Size = _copyAs<uint64_t>(Desc, offsetof(EntryT, size));
    auto TripleSize = _copyAs<uint64_t>(Desc, offsetof(EntryT, tripleSize));
    const char *Triple = Desc + offsetof(EntryT, triple);
    std::string_view EntryID(Triple, TripleSize);

    // std::cout << "Bundle " << i << ":" << std::endl;
    // std::cout << "  ID: " << EntryID << std::endl;
    // std::cout << "  Offset: " << Offset << std::endl;
    // std::cout << "  Size: " << Size << std::endl;

    // SPIR-V bundle entry ID for HIP-Clang 14+. Additional components
    // are ignored for now.
    std::string_view SPIRVBundleID = "hip-spirv64";
    if (EntryID.substr(0, SPIRVBundleID.size()) == SPIRVBundleID ||
        // Legacy entry ID used during early development.
        EntryID == "hip-spir64-unknown-unknown") {
      // std::cout << "Found SPIR-V bundle" << std::endl;
      const char *spirvData = Header + Offset;
      uint32_t magic;
      std::memcpy(&magic, spirvData, sizeof(uint32_t));
      // std::cout << "Magic at offset: 0x" << std::hex << magic << std::dec
                // << std::endl;
      if (magic == SPIRV_MAGIC) {
        // std::cout << "Valid SPIR-V magic number found" << std::endl;
        return std::string_view(spirvData, Size);
      }
      // std::cout << "Invalid SPIR-V magic number" << std::endl;
    }

    // std::cout << "Not a SPIR-V triple, ignoring" << std::endl;
    Desc = Triple + TripleSize; // Next bundle entry.
  }

  ErrorMsg = "Couldn't find SPIR-V binary in the bundle!";
  return std::string_view();
}

std::string disassembleSPIRV(const std::string_view &spirvBinary) {
  spv_context context = spvContextCreate(SPV_ENV_UNIVERSAL_1_1);
  spv_text text = nullptr;
  spv_diagnostic diagnostic = nullptr;

  // std::cout << "SPIR-V binary size: " << spirvBinary.size() << std::endl;
  // std::cout << "First few bytes: ";
  for (size_t i = 0; i < std::min(size_t(16), spirvBinary.size()); i++) {
    // std::cout << std::hex << (int)(unsigned char)spirvBinary[i] << " ";
  }
  // std::cout << std::dec << std::endl;

  spv_result_t result = spvBinaryToText(
      context, reinterpret_cast<const uint32_t *>(spirvBinary.data()),
      spirvBinary.size() / sizeof(uint32_t), // Convert bytes to words
      SPV_BINARY_TO_TEXT_OPTION_INDENT |
          SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES,
      &text, &diagnostic);

  if (result == SPV_SUCCESS) {
    std::string spirvText(text->str, text->length);
    spvTextDestroy(text);
    spvContextDestroy(context);
    return spirvText;
  } else {
    std::cerr << "Failed to disassemble SPIR-V: " << diagnostic->error
              << std::endl;
    spvDiagnosticDestroy(diagnostic);
    spvContextDestroy(context);
  }

  return "";
}

bool usesDoubles(const std::string &spirvHumanReadable) {
  std::istringstream iss(spirvHumanReadable);
  std::string line;
  while (std::getline(iss, line)) {
    if (line.find("OpTypeFloat 64") != std::string::npos) {
      return true;
    }
  }
  return false;
}

// SPIR-V Verification structures and functions
struct SPIRVVerificationResult {
  bool isValid;
  std::vector<std::string> errors;
  std::vector<std::string> warnings;
  
  void addError(const std::string& message) {
    errors.push_back("ERROR: " + message);
    isValid = false;
  }
  
  void addWarning(const std::string& message) {
    warnings.push_back("WARNING: " + message);
  }
  
  bool hasIssues() const {
    return !errors.empty() || !warnings.empty();
  }
  
  void printResults() const {
    for (const auto& error : errors) {
      std::cerr << error << std::endl;
    }
    for (const auto& warning : warnings) {
      std::cerr << warning << std::endl;
    }
  }
};

// SPIR-V verification functions
SPIRVVerificationResult verifySPIRVBinary(const std::string_view& spirvBinary);
SPIRVVerificationResult verifySPIRVText(const std::string& spirvText);
bool validateSPIRVHeader(const std::string_view& spirvBinary, SPIRVVerificationResult& result);
bool validateSPIRVInstructions(const std::string& spirvText, SPIRVVerificationResult& result);
bool checkHIPSpecificConstraints(const std::string& spirvText, SPIRVVerificationResult& result);
bool checkMemoryModel(const std::string& spirvText, SPIRVVerificationResult& result);
bool checkAddressingModel(const std::string& spirvText, SPIRVVerificationResult& result);
bool checkExecutionModel(const std::string& spirvText, SPIRVVerificationResult& result);
bool checkCapabilities(const std::string& spirvText, SPIRVVerificationResult& result);

// SPIR-V verification function implementations
SPIRVVerificationResult verifySPIRVBinary(const std::string_view& spirvBinary) {
  SPIRVVerificationResult result;
  result.isValid = true;
  
  // First validate using SPIR-V Tools validator
  spv_context context = spvContextCreate(SPV_ENV_UNIVERSAL_1_1);
  spv_diagnostic diagnostic = nullptr;
  
  spv_result_t validationResult = spvValidateBinary(
    context,
    reinterpret_cast<const uint32_t*>(spirvBinary.data()),
    spirvBinary.size() / sizeof(uint32_t),
    &diagnostic
  );
  
  if (validationResult != SPV_SUCCESS) {
    if (diagnostic) {
      result.addError("SPIR-V validation failed: " + std::string(diagnostic->error));
      spvDiagnosticDestroy(diagnostic);
    } else {
      result.addError("SPIR-V validation failed with unknown error");
    }
  }
  
  spvContextDestroy(context);
  
  // Validate header structure
  if (!validateSPIRVHeader(spirvBinary, result)) {
    return result;
  }
  
  // Convert to text for additional checks
  std::string spirvText = disassembleSPIRV(spirvBinary);
  if (spirvText.empty()) {
    result.addError("Failed to disassemble SPIR-V binary for text-based validation");
    return result;
  }
  
  // Perform text-based validation
  return verifySPIRVText(spirvText);
}

SPIRVVerificationResult verifySPIRVText(const std::string& spirvText) {
  SPIRVVerificationResult result;
  result.isValid = true;
  
  // Perform various validation checks
  validateSPIRVInstructions(spirvText, result);
  checkHIPSpecificConstraints(spirvText, result);
  checkMemoryModel(spirvText, result);
  checkAddressingModel(spirvText, result);
  checkExecutionModel(spirvText, result);
  checkCapabilities(spirvText, result);
  
  return result;
}

bool validateSPIRVHeader(const std::string_view& spirvBinary, SPIRVVerificationResult& result) {
  if (spirvBinary.size() < 20) { // SPIR-V header is 20 bytes (5 words)
    result.addError("SPIR-V binary too small to contain valid header");
    return false;
  }
  
  const uint32_t* words = reinterpret_cast<const uint32_t*>(spirvBinary.data());
  
  // Check magic number
  if (words[0] != SPIRV_MAGIC) {
    result.addError("Invalid SPIR-V magic number: 0x" + 
                   std::to_string(words[0]) + " (expected 0x07230203)");
    return false;
  }
  
  // Check version (word 1)
  uint32_t version = words[1];
  uint32_t major = (version >> 16) & 0xFF;
  uint32_t minor = (version >> 8) & 0xFF;
  
  if (major < 1 || (major == 1 && minor < 0)) {
    result.addWarning("SPIR-V version " + std::to_string(major) + "." + 
                     std::to_string(minor) + " may not be supported");
  }
  
  // Check generator magic number (word 2) - informational
  uint32_t generator = words[2];
  if (generator == 0) {
    result.addWarning("Generator magic number is 0 (unknown generator)");
  }
  
  // Check bound (word 3) - must be non-zero
  uint32_t bound = words[3];
  if (bound == 0) {
    result.addError("Invalid bound value: 0 (must be non-zero)");
    return false;
  }
  
  // Word 4 is reserved and should be 0
  if (words[4] != 0) {
    result.addWarning("Reserved field in header is non-zero: " + std::to_string(words[4]));
  }
  
  return true;
}

bool validateSPIRVInstructions(const std::string& spirvText, SPIRVVerificationResult& result) {
  std::istringstream iss(spirvText);
  std::string line;
  bool hasOpEntryPoint = false;
  bool hasOpMemoryModel = false;
  int lineNumber = 0;
  
  while (std::getline(iss, line)) {
    lineNumber++;
    
    // Trim whitespace
    line.erase(0, line.find_first_not_of(" \t"));
    line.erase(line.find_last_not_of(" \t") + 1);
    
    if (line.empty() || line[0] == ';') continue; // Skip comments and empty lines
    
    // Check for required instructions
    if (line.find("OpEntryPoint") != std::string::npos) {
      hasOpEntryPoint = true;
    }
    if (line.find("OpMemoryModel") != std::string::npos) {
      hasOpMemoryModel = true;
    }
    
    // Check for problematic patterns
    if (line.find("OpUndef") != std::string::npos) {
      result.addWarning("OpUndef instruction found at line " + std::to_string(lineNumber) + 
                       " - may cause undefined behavior");
    }
    
    // Check for deprecated instructions
    if (line.find("OpTypePointer") != std::string::npos && 
        line.find("Generic") != std::string::npos) {
      result.addWarning("Generic address space usage at line " + std::to_string(lineNumber) + 
                       " - consider using specific address spaces");
    }
  }
  
  // Check for required instructions
  if (!hasOpEntryPoint) {
    result.addError("Missing required OpEntryPoint instruction");
  }
  if (!hasOpMemoryModel) {
    result.addError("Missing required OpMemoryModel instruction");
  }
  
  return result.isValid;
}

bool checkHIPSpecificConstraints(const std::string& spirvText, SPIRVVerificationResult& result) {
  std::istringstream iss(spirvText);
  std::string line;
  bool hasKernelExecutionModel = false;
  int lineNumber = 0;
  
  while (std::getline(iss, line)) {
    lineNumber++;
    
    // Check for HIP/OpenCL kernel execution model
    if (line.find("OpEntryPoint") != std::string::npos && 
        line.find("Kernel") != std::string::npos) {
      hasKernelExecutionModel = true;
    }
    
    // Check for unsupported HIP features
    if (line.find("OpTypeImage") != std::string::npos) {
      result.addWarning("Image types found at line " + std::to_string(lineNumber) + 
                       " - ensure HIP runtime supports this feature");
    }
    
    if (line.find("OpTypeSampler") != std::string::npos) {
      result.addWarning("Sampler types found at line " + std::to_string(lineNumber) + 
                       " - ensure HIP runtime supports this feature");
    }
    
    // Check for atomic operations
    if (line.find("OpAtomic") != std::string::npos) {
      result.addWarning("Atomic operations found at line " + std::to_string(lineNumber) + 
                       " - verify memory scope and semantics are HIP-compatible");
    }
    
    // Check for barrier operations
    if (line.find("OpControlBarrier") != std::string::npos || 
        line.find("OpMemoryBarrier") != std::string::npos) {
      result.addWarning("Barrier operations found at line " + std::to_string(lineNumber) + 
                       " - verify scope and semantics are HIP-compatible");
    }
  }
  
  if (!hasKernelExecutionModel) {
    result.addWarning("No kernel execution model found - this may not be a valid HIP kernel");
  }
  
  return true;
}

bool checkMemoryModel(const std::string& spirvText, SPIRVVerificationResult& result) {
  std::istringstream iss(spirvText);
  std::string line;
  bool foundMemoryModel = false;
  
  while (std::getline(iss, line)) {
    if (line.find("OpMemoryModel") != std::string::npos) {
      foundMemoryModel = true;
      
      // Check for supported memory models
      if (line.find("OpenCL") != std::string::npos) {
        // OpenCL memory model is expected for HIP
        continue;
      } else if (line.find("GLSL450") != std::string::npos) {
        result.addWarning("GLSL450 memory model may not be fully compatible with HIP");
      } else if (line.find("Vulkan") != std::string::npos) {
        result.addWarning("Vulkan memory model may not be fully compatible with HIP");
      } else {
        result.addWarning("Unknown or unsupported memory model in: " + line);
      }
    }
  }
  
  if (!foundMemoryModel) {
    result.addError("No OpMemoryModel instruction found");
  }
  
  return foundMemoryModel;
}

bool checkAddressingModel(const std::string& spirvText, SPIRVVerificationResult& result) {
  std::istringstream iss(spirvText);
  std::string line;
  bool foundAddressingModel = false;
  
  while (std::getline(iss, line)) {
    if (line.find("OpMemoryModel") != std::string::npos) {
      foundAddressingModel = true;
      
      // Check for supported addressing models
      if (line.find("Physical64") != std::string::npos) {
        // Physical64 is expected for HIP on 64-bit systems
        continue;
      } else if (line.find("Physical32") != std::string::npos) {
        result.addWarning("Physical32 addressing model - ensure this matches target architecture");
      } else if (line.find("Logical") != std::string::npos) {
        result.addWarning("Logical addressing model may not be optimal for HIP kernels");
      } else {
        result.addWarning("Unknown addressing model in: " + line);
      }
    }
  }
  
  return foundAddressingModel;
}

bool checkExecutionModel(const std::string& spirvText, SPIRVVerificationResult& result) {
  std::istringstream iss(spirvText);
  std::string line;
  bool foundKernelModel = false;
  
  while (std::getline(iss, line)) {
    if (line.find("OpEntryPoint") != std::string::npos) {
      if (line.find("Kernel") != std::string::npos) {
        foundKernelModel = true;
      } else if (line.find("Vertex") != std::string::npos || 
                 line.find("Fragment") != std::string::npos ||
                 line.find("Geometry") != std::string::npos) {
        result.addError("Graphics execution models (Vertex/Fragment/Geometry) are not supported in HIP");
      } else if (line.find("GLCompute") != std::string::npos) {
        result.addWarning("GLCompute execution model may not be fully compatible with HIP");
      }
    }
  }
  
  if (!foundKernelModel) {
    result.addError("No Kernel execution model found - HIP requires kernel entry points");
  }
  
  return foundKernelModel;
}

bool checkCapabilities(const std::string& spirvText, SPIRVVerificationResult& result) {
  std::istringstream iss(spirvText);
  std::string line;
  std::vector<std::string> requiredCapabilities = {"Kernel", "Addresses"};
  std::vector<std::string> foundCapabilities;
  
  while (std::getline(iss, line)) {
    if (line.find("OpCapability") != std::string::npos) {
      for (const auto& cap : requiredCapabilities) {
        if (line.find(cap) != std::string::npos) {
          foundCapabilities.push_back(cap);
        }
      }
      
      // Check for potentially problematic capabilities
      if (line.find("Float64") != std::string::npos) {
        result.addWarning("Float64 capability found - ensure target device supports double precision");
      }
      if (line.find("Int64") != std::string::npos) {
        result.addWarning("Int64 capability found - ensure target device supports 64-bit integers");
      }
      if (line.find("ImageBasic") != std::string::npos || 
          line.find("ImageReadWrite") != std::string::npos) {
        result.addWarning("Image capabilities found - ensure HIP runtime supports image operations");
      }
    }
  }
  
  // Check for missing required capabilities
  for (const auto& required : requiredCapabilities) {
    bool found = std::find(foundCapabilities.begin(), foundCapabilities.end(), required) 
                 != foundCapabilities.end();
    if (!found) {
      result.addError("Missing required capability: " + required);
    }
  }
  
  return true;
}
