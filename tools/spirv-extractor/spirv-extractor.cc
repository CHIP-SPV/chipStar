#include "spirv-extractor.hh"

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0]
              << " [--check-for-doubles] [--validate] [--index N] [--all] [-o "
                 "<output_filename>] [-h] "
                 "<fatbinary_path> [<additional_args>...]"
              << std::endl;
    return 1;
  }

  std::string outputFilename;
  bool dumpToFile = false;
  bool checkForDoubles = false;
  bool validateSpirv = false;
  int moduleIndex =
      -1; // Which SPIR-V module to extract (-1 = none specified, 0+ = specific)
  bool extractAll = false; // Extract all modules when true
  bool helpRequested = false;
  std::vector<std::string> additionalArgs;
  std::string fatbinaryPath;

  // Parse all arguments, allowing options to appear anywhere
  for (int i = 1; i < argc; ++i) {
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
    } else if (std::string(argv[i]) == "--index") {
      if (i + 1 < argc) {
        moduleIndex = std::stoi(argv[++i]);
      } else {
        std::cerr << "Expected module index after --index option" << std::endl;
        return 1;
      }
    } else if (std::string(argv[i]) == "--all") {
      extractAll = true;
    } else if (std::string(argv[i]) == "-h") {
      helpRequested = true;
    } else {
      // Non-option argument
      if (fatbinaryPath.empty()) {
        fatbinaryPath = argv[i];
      } else {
        additionalArgs.emplace_back(argv[i]);
      }
    }
  }

  if (helpRequested) {
    std::cerr << "Usage: " << argv[0]
              << " [--check-for-doubles] [--validate] [--index N] [--all] [-o "
                 "<output_filename>] [-h] "
                 "<fatbinary_path> [<additional_args>...]"
              << std::endl;
    std::cerr << "Options:" << std::endl;
    std::cerr << "  --check-for-doubles  Check if SPIR-V uses double precision "
                 "and skip test if so"
              << std::endl;
    std::cerr << "  --validate           Perform SPIR-V verification (syntax) "
                 "and validation (spec)"
              << std::endl;
    std::cerr
        << "  --index N            Extract only the Nth SPIR-V module (0-based)"
        << std::endl;
    std::cerr << "  --all               Extract all SPIR-V modules"
              << std::endl;
    std::cerr << "                       Default: extract first module only"
              << std::endl;
    std::cerr << "  -o <filename>       Output SPIR-V to specified file(s)"
              << std::endl;
    std::cerr << "  -h                  Show this help message" << std::endl;
    return 0;
  }

  if (fatbinaryPath.empty()) {
    std::cerr << "Missing fatbinary path" << std::endl;
    return 1;
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

  // Check if the path is a directory
  if (std::filesystem::is_directory(fatbinaryPath)) {
    std::cerr << "Error: " << fatbinaryPath << " is a directory, not a file"
              << std::endl;
    return 1;
  }

  file.seekg(0, std::ios::end);
  std::streampos pos = file.tellg();
  if (pos == -1 || !file.good()) {
    std::cerr << "Failed to get file size for: " << fatbinaryPath
              << " (possibly a directory or invalid file)" << std::endl;
    return 1;
  }
  size_t fileSize = static_cast<size_t>(pos);
  file.seekg(0, std::ios::beg);

  std::vector<char> buffer(fileSize);
  file.read(buffer.data(), fileSize);
  file.close();

  std::string errorMsg;

  // Extract specific module if --index is specified
  if (moduleIndex >= 0) {
    // Extract specific module
    std::string_view spirvBinary =
        extractSPIRVModuleByIndex(buffer.data(), moduleIndex, errorMsg);

    if (spirvBinary.empty()) {
      std::cerr << "Failed to extract SPIR-V module " << moduleIndex << ": "
                << errorMsg << std::endl;
      return 1;
    }

    auto spirvText = disassembleSPIRV(spirvBinary);
    bool hasDoubles = usesDoubles(spirvText);

    int exitCode = 0;

    // Perform SPIR-V validation if requested
    if (validateSpirv) {
      std::cout << "Running SPIR-V validator..." << std::endl;
      spv_context context = spvContextCreate(SPV_ENV_UNIVERSAL_1_1);
      spv_diagnostic diagnostic = nullptr;

      spv_result_t validationResult = spvValidateBinary(
          context, reinterpret_cast<const uint32_t *>(spirvBinary.data()),
          spirvBinary.size() / sizeof(uint32_t), &diagnostic);

      if (validationResult != SPV_SUCCESS) {
        if (diagnostic) {
          std::cerr << "SPIR-V validation failed: " << diagnostic->error
                    << std::endl;
          spvDiagnosticDestroy(diagnostic);
        } else {
          std::cerr << "SPIR-V validation failed with unknown error"
                    << std::endl;
        }
        spvContextDestroy(context);
        return 1;
      }

      if (diagnostic && diagnostic->error &&
          std::strlen(diagnostic->error) > 0) {
        std::cout << "SPIR-V validator warnings: " << diagnostic->error
                  << std::endl;
      } else {
        std::cout << "SPIR-V validation passed successfully." << std::endl;
      }

      std::cout << "Performing SPIR-V structural verification..." << std::endl;
      auto verificationResult = verifySPIRVBinary(spirvBinary);

      if (verificationResult.hasIssues()) {
        verificationResult.printResults();
      }

      if (!verificationResult.isValid) {
        std::cerr << "SPIR-V verification failed!" << std::endl;
        return 1;
      } else if (verificationResult.hasIssues()) {
        std::cout << "SPIR-V verification completed with warnings."
                  << std::endl;
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
        std::string command = fatbinaryPath;
        for (const auto &arg : additionalArgs) {
          command += " " + arg;
        }
        exitCode = system(command.c_str());
      }
      return exitCode;
    }

    if (dumpToFile) {
      std::ofstream outputFileText(outputFilename + ".spvasm");
      if (!outputFileText) {
        std::cerr << "Failed to open file: " << outputFilename + ".spvasm"
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
        std::ofstream outputFileBinary(outputFilename + ".spv",
                                       std::ios::binary);
        if (!outputFileBinary) {
          std::cerr << "Failed to open file: " << outputFilename + ".spv"
                    << std::endl;
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
  } else if (extractAll) {
    // Extract all modules
    auto modules = extractAllSPIRVModules(buffer.data(), errorMsg);

    if (modules.empty()) {
      std::cerr << "Failed to extract SPIR-V modules: " << errorMsg
                << std::endl;
      return 1;
    }

    std::cout << "Found " << modules.size() << " SPIR-V module(s)" << std::endl;

    for (size_t i = 0; i < modules.size(); i++) {
      std::string moduleFilename =
          outputFilename + "_module" + std::to_string(i);
      std::cout << "\nProcessing module " << i << "..." << std::endl;

      auto spirvText = disassembleSPIRV(modules[i]);
      bool hasDoubles = usesDoubles(spirvText);

      if (hasDoubles) {
        std::cout << "  Module uses double precision" << std::endl;
      }

      // Write text format
      std::ofstream outputFileText(moduleFilename + ".spvasm");
      if (!outputFileText) {
        std::cerr << "Failed to open file: " << moduleFilename + ".spvasm"
                  << std::endl;
        continue;
      }
      outputFileText << spirvText;
      outputFileText.close();

      // Write binary format
      spv_context context = spvContextCreate(SPV_ENV_UNIVERSAL_1_1);
      spv_binary binary = nullptr;
      spv_diagnostic diagnostic = nullptr;

      spv_result_t result = spvTextToBinary(
          context, spirvText.data(), spirvText.size(), &binary, &diagnostic);

      if (result == SPV_SUCCESS) {
        std::ofstream outputFileBinary(moduleFilename + ".spv",
                                       std::ios::binary);
        if (!outputFileBinary) {
          std::cerr << "Failed to open file: " << moduleFilename + ".spv"
                    << std::endl;
        } else {
          outputFileBinary.write(reinterpret_cast<const char *>(binary->code),
                                 binary->wordCount * sizeof(uint32_t));
          outputFileBinary.close();
          std::cout << "  Written: " << moduleFilename << ".spv and "
                    << moduleFilename << ".spvasm" << std::endl;
        }
        spvBinaryDestroy(binary);
      }

      spvContextDestroy(context);
    }

    return 0;
  } else {
    // Default: Extract first module only
    std::string_view spirvBinary =
        extractSPIRVModuleByIndex(buffer.data(), 0, errorMsg);

    if (spirvBinary.empty()) {
      std::cerr << "Failed to extract SPIR-V module 0: " << errorMsg
                << std::endl;
      return 1;
    }

    auto spirvText = disassembleSPIRV(spirvBinary);
    bool hasDoubles = usesDoubles(spirvText);

    int exitCode = 0;

    // Perform SPIR-V validation if requested
    if (validateSpirv) {
      std::cout << "Running SPIR-V validator..." << std::endl;
      spv_context context = spvContextCreate(SPV_ENV_UNIVERSAL_1_1);
      spv_diagnostic diagnostic = nullptr;

      spv_result_t validationResult = spvValidateBinary(
          context, reinterpret_cast<const uint32_t *>(spirvBinary.data()),
          spirvBinary.size() / sizeof(uint32_t), &diagnostic);

      if (validationResult != SPV_SUCCESS) {
        if (diagnostic) {
          std::cerr << "SPIR-V validation failed: " << diagnostic->error
                    << std::endl;
          spvDiagnosticDestroy(diagnostic);
        } else {
          std::cerr << "SPIR-V validation failed with unknown error"
                    << std::endl;
        }
        spvContextDestroy(context);
        return 1;
      }

      if (diagnostic && diagnostic->error &&
          std::strlen(diagnostic->error) > 0) {
        std::cout << "SPIR-V validator warnings: " << diagnostic->error
                  << std::endl;
      } else {
        std::cout << "SPIR-V validation passed successfully." << std::endl;
      }

      std::cout << "Performing SPIR-V structural verification..." << std::endl;
      auto verificationResult = verifySPIRVBinary(spirvBinary);

      if (verificationResult.hasIssues()) {
        verificationResult.printResults();
      }

      if (!verificationResult.isValid) {
        std::cerr << "SPIR-V verification failed!" << std::endl;
        return 1;
      } else if (verificationResult.hasIssues()) {
        std::cout << "SPIR-V verification completed with warnings."
                  << std::endl;
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
        std::string command = fatbinaryPath;
        for (const auto &arg : additionalArgs) {
          command += " " + arg;
        }
        exitCode = system(command.c_str());
      }
      return exitCode;
    }

    if (dumpToFile) {
      std::ofstream outputFileText(outputFilename + ".spvasm");
      if (!outputFileText) {
        std::cerr << "Failed to open file: " << outputFilename + ".spvasm"
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
        std::ofstream outputFileBinary(outputFilename + ".spv",
                                       std::ios::binary);
        if (!outputFileBinary) {
          std::cerr << "Failed to open file: " << outputFilename + ".spv"
                    << std::endl;
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
}
