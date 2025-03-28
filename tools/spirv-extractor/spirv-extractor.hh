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
