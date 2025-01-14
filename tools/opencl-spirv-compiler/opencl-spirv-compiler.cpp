#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <filesystem>

namespace fs = std::filesystem;

class SPVCompiler {
private:
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    
    enum class SPIRVFileType {
        Binary,
        Assembly,
        Unknown
    };
    
    SPIRVFileType getSPIRVFileType(const std::string& filename) {
        std::string cmd = "file " + filename;
        FILE* pipe = popen(cmd.c_str(), "r");
        if (!pipe) {
            throw std::runtime_error("Failed to execute file command");
        }
        
        char buffer[128];
        std::string result;
        while (!feof(pipe)) {
            if (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
                result += buffer;
            }
        }
        pclose(pipe);
        
        if (result.find("SPIR-V binary") != std::string::npos) {
            return SPIRVFileType::Binary;
        } else if (result.find("ASCII text") != std::string::npos) {
            return SPIRVFileType::Assembly;
        }
        return SPIRVFileType::Unknown;
    }
    
    std::vector<char> readBinaryFile(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary | std::ios::ate);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open file: " + filename);
        }
        
        size_t fileSize = file.tellg();
        std::vector<char> buffer(fileSize);
        
        file.seekg(0);
        file.read(buffer.data(), fileSize);
        file.close();
        
        return buffer;
    }

public:
    SPVCompiler() {
        cl_int error;
        
        // Get platform
        error = clGetPlatformIDs(1, &platform, nullptr);
        if (error != CL_SUCCESS) {
            throw std::runtime_error("Failed to get OpenCL platform");
        }
        
        // Get device
        error = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, nullptr);
        if (error != CL_SUCCESS) {
            throw std::runtime_error("Failed to get OpenCL device");
        }
        
        // Create context
        context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &error);
        if (error != CL_SUCCESS) {
            throw std::runtime_error("Failed to create OpenCL context");
        }
    }
    
    void compileSPVFile(const std::string& spvPath) {
        try {
            auto fileType = getSPIRVFileType(spvPath);
            if (fileType == SPIRVFileType::Unknown) {
                throw std::runtime_error("Input file is neither a SPIR-V binary nor SPIR-V assembly");
            }
            
            std::vector<char> spirvBinary;
            if (fileType == SPIRVFileType::Assembly) {
                // For assembly files, we need to first assemble them using spirv-as
                std::string outputPath = spvPath + ".spv";
                std::string cmd = "spirv-as " + spvPath + " -o " + outputPath;
                if (system(cmd.c_str()) != 0) {
                    throw std::runtime_error("Failed to assemble SPIR-V assembly file");
                }
                spirvBinary = readBinaryFile(outputPath);
                fs::remove(outputPath); // Clean up temporary binary
            } else {
                spirvBinary = readBinaryFile(spvPath);
            }
            
            cl_int error;
            const unsigned char* binary = reinterpret_cast<const unsigned char*>(spirvBinary.data());
            size_t binarySize = spirvBinary.size();
            
            cl_program program = clCreateProgramWithIL(context, binary, binarySize, &error);
            if (error != CL_SUCCESS) {
                throw std::runtime_error("Failed to create program from SPIR-V binary");
            }
            
            // Build the program
            error = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
            if (error != CL_SUCCESS) {
                // Get build log if compilation failed
                size_t logSize;
                clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
                std::vector<char> buildLog(logSize);
                clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, buildLog.data(), nullptr);
                
                std::cerr << "Build failed. Log:" << std::endl;
                std::cerr << buildLog.data() << std::endl;
                throw std::runtime_error("Failed to build program");
            }
            
            // Get binary for the device
            size_t binarySizes;
            error = clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, sizeof(size_t), &binarySizes, nullptr);
            if (error != CL_SUCCESS) {
                throw std::runtime_error("Failed to get program binary size");
            }
            
            std::vector<unsigned char> deviceBinary(binarySizes);
            unsigned char* binaries[] = {deviceBinary.data()};
            error = clGetProgramInfo(program, CL_PROGRAM_BINARIES, sizeof(binaries), binaries, nullptr);
            if (error != CL_SUCCESS) {
                throw std::runtime_error("Failed to get program binary");
            }
            
            // Save the device binary
            std::string outputPath = fs::path(spvPath).stem().string() + "_device.bin";
            std::ofstream outFile(outputPath, std::ios::binary);
            outFile.write(reinterpret_cast<char*>(deviceBinary.data()), binarySizes);
            outFile.close();
            
            std::cout << "Successfully compiled " << spvPath << " to " << outputPath << std::endl;
            
            clReleaseProgram(program);
        } catch (const std::exception& e) {
            std::cerr << "Error processing " << spvPath << ": " << e.what() << std::endl;
        }
    }
    
    ~SPVCompiler() {
        clReleaseContext(context);
    }
};

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <file(s) or directory>" << std::endl;
        std::cout << "Examples:" << std::endl;
        std::cout << "  " << argv[0] << " kernel.spv" << std::endl;
        std::cout << "  " << argv[0] << " kernel.txt" << std::endl;
        std::cout << "  " << argv[0] << " /path/to/directory" << std::endl;
        return 1;
    }
    
    try {
        SPVCompiler compiler;
        
        for (int i = 1; i < argc; i++) {
            std::string path = argv[i];
            
            if (fs::is_directory(path)) {
                // Handle directory
                for (const auto& entry : fs::recursive_directory_iterator(path)) {
                    if (fs::is_regular_file(entry.path())) {
                        compiler.compileSPVFile(entry.path().string());
                    }
                }
            } else if (fs::is_regular_file(path)) {
                compiler.compileSPVFile(path);
            } else {
                std::cerr << "Warning: Path not found or invalid: " << path << std::endl;
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}