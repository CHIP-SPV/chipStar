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

    std::string_view SPIRVBundleID = "hip-spirv64";
    if (EntryID.substr(0, SPIRVBundleID.size()) == SPIRVBundleID ||
        EntryID == "hip-spir64-unknown-unknown") {

      return std::string_view(Bundle + Offset, Size / sizeof(uint32_t));
    }
    Desc += offsetof(EntryT, triple) + TripleSize;
  }

  ErrorMsg = "Couldn't find SPIR-V binary in the bundle!";
  return std::string_view();
}

std::string_view disassembleSPIRV(const std::string_view &spirvBinary) {
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
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0]
              << " [--check-for-doubles] [-o <output_filename>] [-h] "
                 "<fatbinary_path> [<additional_args>...]"
              << std::endl;
    return 1;
  }

  std::string outputFilename;
  bool dumpToFile = false;
  bool checkForDoubles = false;
  bool helpRequested = false;
  std::vector<std::string> additionalArgs;

  int i = 1;
  while (i < argc) {
    if (std::string(argv[i]) == "-o") {
      if (i + 1 < argc) {
        dumpToFile = true;
        outputFilename = argv[++i];
      } else {
        std::cerr << "Expected output filename after -o option" << std::endl;
        return 1;
      }
    } else if (std::string(argv[i]) == "--check-for-doubles") {
      checkForDoubles = true;
    } else if (std::string(argv[i]) == "-h") {
      helpRequested = true;
    } else {
      break;
    }
    ++i;
  }

  if (helpRequested) {
    std::cerr << "Usage: " << argv[0]
              << " [--check-for-doubles] [-o <output_filename>] [-h] "
                 "<fatbinary_path> [<additional_args>...]"
              << std::endl;
    return 0;
  }

  if (i >= argc) {
    std::cerr << "Missing fatbinary path" << std::endl;
    return 1;
  }

  std::string fatbinaryPath = argv[i++];

  // Collect additional arguments
  while (i < argc) {
    additionalArgs.emplace_back(argv[i++]);
  }

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
  bool hasDoubles = usesDoubles(spirvText);

  int exitCode = 0;
  if (checkForDoubles) {
    if (hasDoubles)
      std::cout << "HIP_SKIP_THIS_TEST: Kernel uses doubles" << std::endl;
    else {
      // Execute the binary with additional arguments
      std::string command = fatbinaryPath;
      for (const auto &arg : additionalArgs) {
        command += " " + arg;
      }
      exitCode = system(command.c_str());
    }
    return exitCode;
  }

  if (dumpToFile) {
    std::ofstream outputFileText(outputFilename + ".txt");
    if (!outputFileText) {
      std::cerr << "Failed to open file: " << outputFilename + ".txt" << std::endl;
      return 1;
    }
    outputFileText << spirvText;
    outputFileText.close();


  spv_context context = spvContextCreate(SPV_ENV_UNIVERSAL_1_1);
  spv_binary binary = nullptr;
  spv_diagnostic diagnostic = nullptr;

  spv_result_t result = spvTextToBinary(
      context, spirvText.data(), spirvText.size(), &binary, &diagnostic);

  if (result == SPV_SUCCESS) {
    std::ofstream outputFileBinary(outputFilename, std::ios::binary);
    if (!outputFileBinary) {
      std::cerr << "Failed to open file: " << outputFilename << std::endl;
      return 1;
    }
    outputFileBinary.write(reinterpret_cast<const char*>(binary->code), binary->wordCount * sizeof(uint32_t));
    outputFileBinary.close();
    spvBinaryDestroy(binary);
  } else {
    std::cerr << "Failed to assemble SPIR-V: " << diagnostic->error << std::endl;
    spvDiagnosticDestroy(diagnostic);
    spvContextDestroy(context);
    return 1;
  }

  spvContextDestroy(context);
  } else {
    std::cout << spirvText << std::endl;
  }

  return hasDoubles;
}
