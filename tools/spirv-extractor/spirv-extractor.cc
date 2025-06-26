#include "spirv-extractor.hh"

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0]
              << " [--check-for-doubles] [--validate] [-o <output_filename>] [-h] "
                 "<fatbinary_path> [<additional_args>...]"
              << std::endl;
    return 1;
  }

  std::string outputFilename;
  bool dumpToFile = false;
  bool checkForDoubles = false;
  bool validateSpirv = false;
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
    } else if (std::string(argv[i]) == "--validate") {
      validateSpirv = true;
    } else if (std::string(argv[i]) == "-h") {
      helpRequested = true;
    } else {
      break;
    }
    ++i;
  }

  if (helpRequested) {
    std::cerr << "Usage: " << argv[0]
              << " [--check-for-doubles] [--validate] [-o <output_filename>] [-h] "
                 "<fatbinary_path> [<additional_args>...]" << std::endl;
    std::cerr << "Options:" << std::endl;
    std::cerr << "  --check-for-doubles  Check if SPIR-V uses double precision and skip test if so" << std::endl;
    std::cerr << "  --validate           Perform SPIR-V verification (syntax) and validation (spec)" << std::endl;
    std::cerr << "  -o <filename>       Output SPIR-V to specified file" << std::endl;
    std::cerr << "  -h                  Show this help message" << std::endl;
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
  std::string_view spirvBinary = extractSPIRVModule(buffer.data(), errorMsg);

  if (spirvBinary.empty()) {
    std::cerr << "Failed to extract SPIR-V binary from the fatbinary: "
              << errorMsg << std::endl;
    return 1;
  }

  auto spirvText = disassembleSPIRV(spirvBinary);
  bool hasDoubles = usesDoubles(spirvText);

  int exitCode = 0;
  
  // Perform SPIR-V validation if requested (lighter weight than full verify)
  if (validateSpirv) {
    std::cout << "Running SPIR-V validator..." << std::endl;
    spv_context context = spvContextCreate(SPV_ENV_UNIVERSAL_1_1);
    spv_diagnostic diagnostic = nullptr;

    spv_result_t validationResult = spvValidateBinary(
        context, reinterpret_cast<const uint32_t *>(spirvBinary.data()),
        spirvBinary.size() / sizeof(uint32_t), &diagnostic);

    if (validationResult != SPV_SUCCESS) {
      if (diagnostic) {
        std::cerr << "SPIR-V validation failed: " << diagnostic->error << std::endl;
        spvDiagnosticDestroy(diagnostic);
      } else {
        std::cerr << "SPIR-V validation failed with unknown error" << std::endl;
      }
      spvContextDestroy(context);
      return 1;
    }

    if (diagnostic && diagnostic->error && std::strlen(diagnostic->error) > 0) {
      std::cout << "SPIR-V validator warnings: " << diagnostic->error << std::endl;
    } else {
      std::cout << "SPIR-V validation passed successfully." << std::endl;
    }

    // After spec validation, run structural verification for completeness
    std::cout << "Performing SPIR-V structural verification..." << std::endl;
    auto verificationResult = verifySPIRVBinary(spirvBinary);

    if (verificationResult.hasIssues()) {
      verificationResult.printResults();
    }

    if (!verificationResult.isValid) {
      std::cerr << "SPIR-V verification failed!" << std::endl;
      return 1;
    } else if (verificationResult.hasIssues()) {
      std::cout << "SPIR-V verification completed with warnings." << std::endl;
    } else {
      std::cout << "SPIR-V verification passed successfully." << std::endl;
    }

    if (diagnostic)
      spvDiagnosticDestroy(diagnostic);
    spvContextDestroy(context);
  }
  
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
      std::cerr << "Failed to open file: " << outputFilename + ".txt"
                << std::endl;
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
      outputFileBinary.write(reinterpret_cast<const char *>(binary->code),
                             binary->wordCount * sizeof(uint32_t));
      outputFileBinary.close();
      spvBinaryDestroy(binary);
    } else {
      std::cerr << "Failed to assemble SPIR-V: " << diagnostic->error
                << std::endl;
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
