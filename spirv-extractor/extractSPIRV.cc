#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <unordered_map>
#include <memory>
#include <string>
#include <sstream>
#include <spirv-tools/libspirv.hpp>
#include <string_view>
#include <filesystem>

#define CLANG_OFFLOAD_BUNDLER_MAGIC "__CLANG_OFFLOAD_BUNDLE__"
struct __ClangOffloadBundleDesc {
  uint64_t offset;
  uint64_t size;
  uint64_t tripleSize;
  const char triple[1];
};

struct __ClangOffloadBundleHeader {
  const char magic[sizeof(CLANG_OFFLOAD_BUNDLER_MAGIC) - 1];
  uint64_t numBundles;
  __ClangOffloadBundleDesc desc[1];
};

template <class T>
static T copyAs(const void *BaseAddr, size_t ByteOffset = 0) {
  T Res;
  std::memcpy(&Res, (const char *)BaseAddr + ByteOffset, sizeof(T));
  return Res;
}

std::string_view extractSPIRVModule(const char *Bundle, std::string &ErrorMsg,
                                    size_t bundleSize) {
  // Ensure that we don't go beyond the bundleSize while reading the binary.
  const char *BundleEnd = Bundle + bundleSize;

  // NOTE: This method is designed to read from possibly misaligned buffer.
  std::string Magic(reinterpret_cast<const char *>(Bundle),
                    sizeof(CLANG_OFFLOAD_BUNDLER_MAGIC) - 1);
  if (Magic != CLANG_OFFLOAD_BUNDLER_MAGIC) {
    ErrorMsg = "The bundled binaries are not Clang bundled "
               "(CLANG_OFFLOAD_BUNDLER_MAGIC is missing)";
    return std::string_view();
  }

  using HeaderT = __ClangOffloadBundleHeader;
  using EntryT = __ClangOffloadBundleDesc;

  auto NumBundles = copyAs<uint64_t>(Bundle, offsetof(HeaderT, numBundles));
  std::cout << "Number of bundles: " << NumBundles << std::endl;

  const char *Desc = Bundle + offsetof(HeaderT, desc);

  for (size_t i = 0; i < NumBundles; i++) {
    if (reinterpret_cast<const char *>(Desc) >= BundleEnd) {
      ErrorMsg = "Bundle descriptor exceeds allocated bundle size.";
      return std::string_view();
    }

    auto Offset = copyAs<uint64_t>(Desc, offsetof(EntryT, offset));
    auto Size = copyAs<uint64_t>(Desc, offsetof(EntryT, size));
    auto TripleSize = copyAs<uint64_t>(Desc, offsetof(EntryT, tripleSize));

    if (Bundle + Offset + Size > BundleEnd) {
      ErrorMsg = "SPIR-V module exceeds allocated bundle size.";
      return std::string_view();
    }

    const char *Triple = Desc + offsetof(EntryT, triple);
    std::string_view EntryID(Triple, TripleSize);
    std::cout << "Bundle entry ID " << i << " is: '" << EntryID << "'\n"
              << "  Offset: " << Offset << "\n"
              << "  Size: " << Size << "\n"
              << "  TripleSize: " << TripleSize << std::endl;

    std::string_view SPIRVBundleID = "hip-spirv64";
    if (EntryID.substr(0, SPIRVBundleID.size()) == SPIRVBundleID ||
        EntryID == "hip-spir64-unknown-unknown") {
      std::cout << "Found SPIR-V binary, offset: " << Offset
                << ", size: " << Size << std::endl;

      return std::string_view(Bundle + Offset, Size / sizeof(uint32_t));
    }
    std::cout << "Not a SPIR-V triple, ignoring\n";
    Desc +=
        offsetof(EntryT, triple) + TripleSize; // Move to the next descriptor
  }

  ErrorMsg = "Couldn't find SPIR-V binary in the bundle!";
  return std::string_view();
}

std::string_view disassembleSPIRV(const std::string_view &spirvBinary) {
  std::string_view spirvHumanReadable;
  spv_context context = spvContextCreate(SPV_ENV_UNIVERSAL_1_1);
  spv_text text = nullptr;
  spv_diagnostic diagnostic = nullptr;

  spv_result_t result = spvBinaryToText(
      context, reinterpret_cast<const uint32_t *>(spirvBinary.data()),
      spirvBinary.size(),
      SPV_BINARY_TO_TEXT_OPTION_INDENT |
          SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES,
      &text, &diagnostic);

  if (result == SPV_SUCCESS) {
    return std::string_view(text->str, text->length);
    spvTextDestroy(text);
  } else {
    std::cerr << "Failed to disassemble SPIR-V: " << diagnostic->error
              << std::endl;
    spvDiagnosticDestroy(diagnostic);
  }

  spvContextDestroy(context);
  return "";
}

bool usesDoubles(const std::string_view &spirvHumanReadable) {
  std::istringstream iss(spirvHumanReadable.data());
  std::string line;
  while (std::getline(iss, line)) {
    if (line.find("OpTypeFloat 64") != std::string::npos) {
      return true;
    }
  }
  return false;
}

int main(int argc, char *argv[]) {
  if (argc < 3 || argc > 5) {
    std::cerr
        << "Usage: " << argv[0]
        << " [--check-for-doubles] [-o <output_filename>] <fatbinary_path>"
        << std::endl;
    return 1;
  }

  std::string outputFilename;
  bool dumpToFile = false;
  bool checkForDoubles = false;

  for (int i = 1; i < argc - 1; ++i) {
    if (std::string(argv[i]) == "-o") {
      if (i + 1 < argc - 1) {
        dumpToFile = true;
        outputFilename = argv[++i];
      } else {
        std::cerr << "Expected output filename after -o option" << std::endl;
        return 1;
      }
    } else if (std::string(argv[i]) == "--check-for-doubles") {
      checkForDoubles = true;
    } else {
      std::cerr << "Invalid option: " << argv[i] << std::endl;
      return 1;
    }
  }

  std::string fatbinaryPath = argv[argc - 1];

  if (!dumpToFile) {
    outputFilename =
        std::filesystem::path(fatbinaryPath).stem().string() + ".spv";
  }

  std::ifstream file(fatbinaryPath, std::ios::binary);
  if (!file) {
    std::cerr << "Failed to open file: " << fatbinaryPath << std::endl;
    return 1;
  }

  file.seekg(0, std::ios::end);
  size_t fileSize = file.tellg();
  file.seekg(0, std::ios::beg);

  std::vector<char> buffer(fileSize);
  file.read(buffer.data(), fileSize);
  file.close();

  std::string errorMsg;
  const char *bundleStart = nullptr;
  size_t bundleLength = 0;

  // Find the start and calculate the end of a single bundle
  for (size_t i = 0; i < fileSize - sizeof(CLANG_OFFLOAD_BUNDLER_MAGIC); ++i) {
    if (std::memcmp(buffer.data() + i, CLANG_OFFLOAD_BUNDLER_MAGIC,
                    sizeof(CLANG_OFFLOAD_BUNDLER_MAGIC) - 1) == 0) {
      bundleStart = buffer.data() + i;
      // Assuming that immediately after the magic starts the header
      const auto *header =
          reinterpret_cast<const __ClangOffloadBundleHeader *>(bundleStart);
      if (fileSize - i >= sizeof(__ClangOffloadBundleHeader)) {
        // Calculate total size of all bundles (header size + data size)
        bundleLength = sizeof(__ClangOffloadBundleHeader) +
                       header->numBundles * sizeof(__ClangOffloadBundleDesc);
        for (size_t j = 0; j < header->numBundles; ++j) {
          const auto *desc = reinterpret_cast<const __ClangOffloadBundleDesc *>(
              bundleStart + sizeof(__ClangOffloadBundleHeader) +
              j * sizeof(__ClangOffloadBundleDesc));
          bundleLength += desc->size; // Add the size of each bundle's data
        }
      }
      break;
    }
  }

  if (bundleStart == nullptr || bundleLength == 0) {
    std::cerr << "Failed to find or calculate the size of the Clang offload "
                 "bundle in the binary."
              << std::endl;
    return 1;
  }

  std::string_view spirvBinary =
      extractSPIRVModule(bundleStart, errorMsg, bundleLength);

  if (spirvBinary.empty()) {
    std::cerr << "Failed to extract SPIR-V binary from the fatbinary: "
              << errorMsg << std::endl;
    return 1;
  }

  auto spirvText = disassembleSPIRV(spirvBinary);

  if (checkForDoubles) {
    std::cout << "SPIR-V uses doubles: "
              << (usesDoubles(spirvText) ? "true" : "false") << std::endl;
    return 1;
  } else if (dumpToFile) {
    std::ofstream outputFile(outputFilename);
    if (!outputFile) {
      std::cerr << "Failed to open file: " << outputFilename << std::endl;
      return 1;
    }
    outputFile << spirvText;
    outputFile.close();
  } else {
    std::cout << spirvText << std::endl;
  }

  return 0;
}
