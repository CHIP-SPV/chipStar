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

// Helper to extract kernel signature from binary
std::string getKernelSignature(const void* function_address) {
    // Get the current executable's path from /proc
    char exe_path[PATH_MAX];
    ssize_t len = readlink("/proc/self/exe", exe_path, sizeof(exe_path)-1);
    if (len == -1) {
        std::cerr << "Failed to read /proc/self/exe: " << strerror(errno) << std::endl;
        return "";
    }
    exe_path[len] = '\0';
    
    // Use nm to get symbol information from the executable
    std::string cmd = "nm -C " + std::string(exe_path) + " | grep __device_stub_";
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
    
    // Parse the nm output to extract function signature
    std::regex signature_regex(R"(__device_stub_([^\(]+)\((.*?)\))");
    std::smatch matches;
    if (std::regex_search(result, matches, signature_regex)) {
        return matches[1].str() + "(" + matches[2].str() + ")";
    }
    
    std::cerr << "No kernel signature found in binary " << exe_path << std::endl;
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
    Dl_info info;
    if (dladdr(function_address, &info)) {
        // Add debug logging
        std::cout << "dladdr results:\n"
                  << "  fname: " << (info.dli_fname ? info.dli_fname : "null") << "\n"
                  << "  sname: " << (info.dli_sname ? info.dli_sname : "null") << "\n"
                  << "  fbase: " << info.dli_fbase << "\n"
                  << "  saddr: " << info.dli_saddr << std::endl;
                  
        if (info.dli_sname) {
            // Try to demangle the symbol name if it exists
            std::string demangled = demangle(info.dli_sname);
            if (demangled != info.dli_sname) {
                return demangled;
            }
            return info.dli_sname;
        }
        
        // If we have the filename but no symbol name, try to extract from signature
        if (info.dli_fname) {
            std::string signature = getKernelSignature(function_address);
            if (!signature.empty()) {
                // Extract just the function name from the signature
                size_t start = 0;
                size_t end = signature.find('(');
                if (end != std::string::npos) {
                    return signature.substr(start, end);
                }
            }
        }
    } else {
        std::cerr << "dladdr failed: " << dlerror() << std::endl;
    }
    std::abort();
}

#endif // HIP_INTERCEPT_LAYER_UTIL_HH