#ifndef HIP_INTERCEPT_LAYER_UTIL_HH
#define HIP_INTERCEPT_LAYER_UTIL_HH

#define __HIP_PLATFORM_SPIRV__
#include "hip/hip_runtime_api.h"

#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <cxxabi.h>
#include <dlfcn.h>
#include <cstring>
#include <cerrno>
#include <cstdlib>
#include <cstddef>
#include <unistd.h>
#include <linux/limits.h>
#include <regex>
#include <memory>
#include <queue>
#include <set>
#include <fstream>
#include <sstream>
#include <cstdio>
#include <memory>
#include <link.h>


//Forward declarations
struct dim3;
size_t countKernelArgs(void** args);
void printKernelArgs(void** args, const std::string& kernelName, const void* function_address);
std::string getKernelSignature(const void* function_address);
std::string getKernelName(const void* function_address);


size_t countKernelArgs(void** args) {
    if (!args) return 0;
    
    size_t count = 0;
    while (args[count] != nullptr) {
        count++;
        // Safety check to prevent infinite loop
        if (count > 100) {  // reasonable max number of kernel arguments
            std::cerr << "Warning: Exceeded maximum expected kernel arguments\n";
            break;
        }
    }
    return count;
}

// Helper to print kernel arguments
void printKernelArgs(void** args, const std::string& kernelName, const void* function_address) {
    std::string signature = getKernelSignature(function_address);
    std::cout << "    kernel signature: " << signature << "\n";
    
    if (!args) {
        std::cout << "    args: nullptr\n";
        return;
    }
    
    // Parse signature to get argument types
    std::vector<std::string> argTypes;
    size_t start = signature.find('(');
    size_t end = signature.find(')');
    if (start != std::string::npos && end != std::string::npos) {
        std::string argsStr = signature.substr(start + 1, end - start - 1);
        
        // Handle template arguments more carefully
        size_t pos = 0;
        int template_depth = 0;
        std::string current_arg;
        
        for (char c : argsStr) {
            if (c == '<') template_depth++;
            else if (c == '>') template_depth--;
            else if (c == ',' && template_depth == 0) {
                // Only split on commas outside of template arguments
                if (!current_arg.empty()) {
                    // Trim whitespace
                    current_arg.erase(0, current_arg.find_first_not_of(" "));
                    current_arg.erase(current_arg.find_last_not_of(" ") + 1);
                    argTypes.push_back(current_arg);
                    current_arg.clear();
                }
                continue;
            }
            current_arg += c;
        }
        if (!current_arg.empty()) {
            current_arg.erase(0, current_arg.find_first_not_of(" "));
            current_arg.erase(current_arg.find_last_not_of(" ") + 1);
            argTypes.push_back(current_arg);
        }
    }
    
    std::cout << "    args:\n";
    for (size_t i = 0; i < argTypes.size() && args[i] != nullptr; i++) {
        std::cout << "      arg[" << i << "]: " << argTypes[i] << " = ";
        
        try {
            // Handle different argument types
            if (argTypes[i].find("*") != std::string::npos) {
                // Pointer type
                void* ptr = *(void**)args[i];
                std::cout << ptr;
            } else if (argTypes[i].find("HIP_vector_type") != std::string::npos) {
                // For vector types, print the first value
                float* values = (float*)args[i];
                std::cout << values[0];  // Print just the first component
            } else if (argTypes[i].find("int") != std::string::npos) {
                // Integer type
                std::cout << *(int*)args[i];
            } else if (argTypes[i].find("float") != std::string::npos) {
                // Float type
                std::cout << *(float*)args[i];
            } else {
                // Unknown type - show raw pointer
                std::cout << "[unknown type at " << args[i] << "]";
            }
        } catch (...) {
            std::cout << "[failed to read argument]";
        }
        std::cout << "\n";
    }
}

// Get kernel object file
std::string getKernelObjectFile(const void* function_address) {
    //std::cout << "\nSearching for kernel object file containing address " 
    //          << function_address << std::endl;
              
    std::queue<std::string> files_to_check;
    std::set<std::string> checked_files;
    
    // Start with /proc/self/exe
    files_to_check.push("/proc/self/exe");
    //std::cout << "Starting search with /proc/self/exe" << std::endl;
    
    // Helper function to get dependencies using ldd
    auto getDependencies = [](const std::string& path) {
        std::vector<std::string> deps;
        std::string cmd = "ldd " + path;
        //std::cout << "Running: " << cmd << std::endl;
        
        std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd.c_str(), "r"), pclose);
        if (!pipe) {
            std::cerr << "Failed to run ldd: " << strerror(errno) << std::endl;
            return deps;
        }
        
        char buffer[512];
        while (fgets(buffer, sizeof(buffer), pipe.get()) != nullptr) {
            std::string line(buffer);
            // Look for => in ldd output
            size_t arrow_pos = line.find("=>");
            if (arrow_pos != std::string::npos) {
                // Extract path after =>
                size_t path_start = line.find('/', arrow_pos);
                size_t path_end = line.find(" (", arrow_pos);
                if (path_start != std::string::npos && path_end != std::string::npos) {
                    std::string dep = line.substr(path_start, path_end - path_start);
                    deps.push_back(dep);
                    std::cout << "Found dependency: " << dep << std::endl;
                }
            }
        }
        return deps;
    };
    
    // Helper function to check if address is in file
    auto isAddressInFile = [](const std::string& path, const void* addr) {
        //std::cout << "Checking if address " << addr << " is in " << path << std::endl;
        
        struct CallbackData {
            const void* target_addr;
            bool found;
            std::string found_path;
        };
        
        CallbackData data = {addr, false, ""};
        
        // Callback for dl_iterate_phdr
        auto callback = [](struct dl_phdr_info* info, size_t size, void* data) {
            auto params = static_cast<CallbackData*>(data);
            const void* target_addr = params->target_addr;
            
            std::string lib_path = info->dlpi_name[0] ? info->dlpi_name : "/proc/self/exe";
            //std::cout << "Checking segments in " << lib_path
            //          << " at base address " << (void*)info->dlpi_addr << std::endl;
            
            for (int j = 0; j < info->dlpi_phnum; j++) {
                const ElfW(Phdr)* phdr = &info->dlpi_phdr[j];
                if (phdr->p_type == PT_LOAD) {
                    void* start = (void*)(info->dlpi_addr + phdr->p_vaddr);
                    void* end = (void*)((char*)start + phdr->p_memsz);
                    //std::cout << "  Segment " << j << ": " << start << " - " << end << std::endl;
                    
                    if (target_addr >= start && target_addr < end) {
                        //std::cout << "  Found address in this segment!" << std::endl;
                        params->found = true;
                        params->found_path = lib_path;
                        return 1;  // Stop iteration
                    }
                }
            }
            return 0;  // Continue iteration
        };
        
        dl_iterate_phdr(callback, &data);
        
        if (!data.found) {
            //std::cout << "Address not found in " << path << std::endl;
            return std::make_pair(false, std::string());
        }
        
        return std::make_pair(true, data.found_path);
    };
    
    while (!files_to_check.empty()) {
        std::string current_file = files_to_check.front();
        files_to_check.pop();
        
        if (checked_files.count(current_file)) {
            //std::cout << "Already checked " << current_file << ", skipping" << std::endl;
            continue;
        }
        
        //std::cout << "\nChecking file: " << current_file << std::endl;
        checked_files.insert(current_file);
        
        // Check if the function_address is in this file
        auto [found, actual_path] = isAddressInFile(current_file, function_address);
        if (found) {
            //std::cout << "Found kernel in " << actual_path << "!" << std::endl;
            return actual_path;
        }
        
        // Add dependencies to queue
        //std::cout << "Getting dependencies for " << current_file << std::endl;
        for (const auto& dep : getDependencies(current_file)) {
            if (!checked_files.count(dep)) {
                //std::cout << "Adding to queue: " << dep << std::endl;
                files_to_check.push(dep);
            } else {
                //std::cout << "Already checked dependency: " << dep << std::endl;
            }
        }
    }
    std::cerr << "Searched the following files for kernel address " << function_address << std::endl;
    for (const auto& file : checked_files) {
        std::cerr << "  " << file << std::endl;
    }
    std::abort();
}


// Helper to extract kernel signature from binary
std::string getKernelSignature(const void* function_address) {
    // Get the object file containing this kernel
    std::string object_file = getKernelObjectFile(function_address);
    if (object_file == "unknown") {
        std::cerr << "Failed to find object file containing kernel at " 
                  << function_address << std::endl;
        return "";
    }
    
    // Use nm to get symbol information from the object file
    std::string cmd = "nm -C " + object_file + " | grep __device_stub_";
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) {
        std::cerr << "Failed to run nm command: " << strerror(errno) << std::endl;
        return "";
    }
    
    char buffer[1024];
    std::string result;
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        result += buffer;
    }
    int status = pclose(pipe);
    if (status == -1) {
        std::cerr << "Failed to close pipe: " << strerror(errno) << std::endl;
    }
    std::cout << "nm output: " << result << std::endl;
    // Parse the nm output to extract function signature
    std::regex signature_regex(R"(__device_stub_([^\(]+)\((.*?)\))");
    std::smatch matches;
    if (std::regex_search(result, matches, signature_regex)) {
        auto signature = matches[1].str() + "(" + matches[2].str() + ")";
        std::cout << "Kernel signature: " << signature << std::endl;
        return signature;
    }
    
    std::cerr << "No kernel signature found in binary " << object_file << std::endl;
    return "";
}

// Helper to get type name as string
template<typename T>
std::string getTypeName() {
  std::string name = typeid(T).name();
  
  // Clean up type name
  if constexpr (std::is_pointer_v<T>) {
    name = getTypeName<std::remove_pointer_t<T>>() + "*";
  }
  else if constexpr (std::is_const_v<T>) {
    name = "const " + getTypeName<std::remove_const_t<T>>();
  }
  else {
    // Map common types to readable names
    if (name == "f") name = "float";
    else if (name == "i") name = "int";
    else if (name == "d") name = "double";
    // Add more type mappings as needed
  }
  return name;
}

template<typename... Args>
std::string getArgTypes() {
  std::vector<std::string> typeNames;
  (typeNames.push_back(getTypeName<Args>()), ...);
  
  std::stringstream ss;
  ss << "(";
  for (size_t i = 0; i < typeNames.size(); i++) {
    if (i > 0) ss << ", ";
    ss << typeNames[i];
  }
  ss << ")";
  return ss.str();
}

// Helper to demangle C++ names
std::string demangle(const char* name) {
    int status;
    std::unique_ptr<char, void(*)(void*)> demangled(
        abi::__cxa_demangle(name, nullptr, nullptr, &status),
        std::free
    );
    return status == 0 ? demangled.get() : name;
}

// Helper function to convert hipMemcpyKind to string
static const char* memcpyKindToString(hipMemcpyKind kind) {
  switch(kind) {
    case hipMemcpyHostToHost: return "hipMemcpyHostToHost";
    case hipMemcpyHostToDevice: return "hipMemcpyHostToDevice"; 
    case hipMemcpyDeviceToHost: return "hipMemcpyDeviceToHost";
    case hipMemcpyDeviceToDevice: return "hipMemcpyDeviceToDevice";
    case hipMemcpyDefault: return "hipMemcpyDefault";
    default: return "Unknown";
  }
}


// Helper function to convert dim3 to string
static std::string dim3ToString(dim3 d) {
  std::stringstream ss;
  ss << "{" << d.x << "," << d.y << "," << d.z << "}";
  return ss.str();
}

// Helper for hipDeviceProp_t
static std::string devicePropsToString(const hipDeviceProp_t* props) {
  if (!props) return "null";
  std::stringstream ss;
  ss << "{name=" << props->name << ", totalGlobalMem=" << props->totalGlobalMem << "}";
  return ss.str();
}

std::string getArgTypeFromSignature(const std::string& signature, size_t arg_index) {
    size_t start = signature.find('(');
    size_t end = signature.find(')');
    if (start == std::string::npos || end == std::string::npos) {
        return "";
    }
    
    std::string args_str = signature.substr(start + 1, end - start - 1);
    
    // Parse arguments handling nested template brackets
    std::vector<std::string> args;
    std::string current_arg;
    int template_depth = 0;
    
    for (char c : args_str) {
        if (c == '<') {
            template_depth++;
            current_arg += c;
        }
        else if (c == '>') {
            template_depth--;
            current_arg += c;
        }
        else if (c == ',' && template_depth == 0) {
            // Only split on commas outside of template brackets
            if (!current_arg.empty()) {
                // Trim whitespace
                current_arg.erase(0, current_arg.find_first_not_of(" "));
                current_arg.erase(current_arg.find_last_not_of(" ") + 1);
                args.push_back(current_arg);
                current_arg.clear();
            }
        }
        else {
            current_arg += c;
        }
    }
    
    // Add the last argument
    if (!current_arg.empty()) {
        current_arg.erase(0, current_arg.find_first_not_of(" "));
        current_arg.erase(current_arg.find_last_not_of(" ") + 1);
        args.push_back(current_arg);
    }
    
    // Return the requested argument type if index is valid
    if (arg_index < args.size()) {
        return args[arg_index];
    }
    
    return "";
}

struct KernelArgInfo {
    bool is_vector;
    size_t size;
};

struct KernelInfo {
    std::vector<KernelArgInfo> args;
};

static bool isVectorType(const std::string& type_name) {
    static const std::vector<std::string> vector_types = {
        "float4", "float3", "float2",
        "int4", "int3", "int2",
        "uint4", "uint3", "uint2",
        "double4", "double3", "double2",
        "long4", "long3", "long2",
        "ulong4", "ulong3", "ulong2",
        "char4", "char3", "char2",
        "uchar4", "uchar3", "uchar2",
        "HIP_vector_type"
    };
    
    for (const auto& vtype : vector_types) {
        if (type_name.find(vtype) != std::string::npos) {
            return true;
        }
    }
    return false;
}

std::string getKernelName(const void* function_address) {
    // First find which object file contains this kernel
    auto object_file = getKernelObjectFile(function_address);
    std::cout << "Kernel object file: " << object_file << std::endl;
    
    if (object_file == "unknown") {
        std::cerr << "Could not find object file containing kernel at " 
                  << function_address << std::endl;
        std::abort();
    }
    
    // If dladdr failed or didn't give us a symbol name, 
    // try to get it from the kernel signature
    std::string signature = getKernelSignature(function_address);
    if (!signature.empty()) {
        size_t end = signature.find('(');
        if (end != std::string::npos) {
            return signature.substr(0, end);
        }
    }
    
    // If we get here, we've failed to get the name through any method
    std::cerr << "Failed to get kernel name for address " << function_address 
              << " in file " << object_file << std::endl;
    
    std::abort();
}

#endif // HIP_INTERCEPT_LAYER_UTIL_HH