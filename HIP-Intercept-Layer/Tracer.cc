#include "Tracer.hh"
#include <iostream>
#include <filesystem>
#include <chrono>
#include <unistd.h>
#include <linux/limits.h>
#include <sys/stat.h>
#include <dlfcn.h>
#include <link.h>
#include <algorithm>
#include <regex>
#include <fstream>

namespace hip_intercept {

// MemoryState implementation
MemoryState::MemoryState(size_t s) : data(new char[s]), size(s) {}

MemoryState::MemoryState(const char* src, size_t s) : data(new char[s]), size(s) {
    memcpy(data.get(), src, s);
}

MemoryState::MemoryState() : size(0) {}

// Tracer implementation
Tracer& Tracer::instance() {
    static Tracer instance;
    return instance;
}

Tracer::Tracer() : initialized_(false) {
    initializeTraceFile();
}

Tracer::~Tracer() {
    if (trace_file_.is_open()) {
        trace_file_.close();
    }
}

void Tracer::initializeTraceFile() {
    if (initialized_) return;
    
    trace_path_ = getTraceFilePath();
    if (trace_path_.empty()) return; // Skip tracing for this process
    
    trace_file_.open(trace_path_, std::ios::binary);
    if (!trace_file_) {
        std::cerr << "Failed to open trace file: " << trace_path_ << std::endl;
        return;
    }
    
    // Add this more visible message
    std::cout << "\n=== HIP Trace File ===\n"
              << "Writing trace to: " << trace_path_ << "\n"
              << "===================\n\n";
    
    // Write header
    struct {
        uint32_t magic = TRACE_MAGIC;
        uint32_t version = TRACE_VERSION;
    } header;
    
    trace_file_.write(reinterpret_cast<char*>(&header), sizeof(header));
    initialized_ = true;
}

void Tracer::recordKernelLaunch(const KernelExecution& exec) {
    if (!initialized_) return;
    writeKernelExecution(exec);
}

void Tracer::recordMemoryOperation(const MemoryOperation& op) {
    if (!initialized_) return;
    writeMemoryOperation(op);
}

void Tracer::writeEvent(uint32_t type, const void* data, size_t size) {
    struct {
        uint32_t type;
        uint64_t timestamp;
        uint32_t size;
    } event_header = {
        type,
        static_cast<uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count()),
        static_cast<uint32_t>(size)
    };
    
    trace_file_.write(reinterpret_cast<char*>(&event_header), sizeof(event_header));
    trace_file_.write(reinterpret_cast<const char*>(data), size);
}

void Tracer::writeKernelExecution(const KernelExecution& exec) {
    // Serialize kernel execution data
    struct {
        void* function_address;
        uint32_t name_length;
        dim3 grid_dim;
        dim3 block_dim;
        size_t shared_mem;
        hipStream_t stream;
        uint64_t execution_order;
        uint32_t num_args;
        uint32_t num_changes;
    } kernel_data = {
        exec.function_address,
        static_cast<uint32_t>(exec.kernel_name.length()),
        exec.grid_dim,
        exec.block_dim,
        exec.shared_mem,
        exec.stream,
        exec.execution_order,
        static_cast<uint32_t>(exec.arg_ptrs.size()),
        0  // Will count total changes below
    };
    
    // Count total changes
    for (const auto& [arg_idx, changes] : exec.changes_by_arg) {
        kernel_data.num_changes += changes.size();
    }
    
    writeEvent(1, &kernel_data, sizeof(kernel_data));
    
    // Write kernel name
    trace_file_.write(exec.kernel_name.c_str(), exec.kernel_name.length());
    
    // Write argument pointers
    for (void* ptr : exec.arg_ptrs) {
        trace_file_.write(reinterpret_cast<const char*>(&ptr), sizeof(void*));
    }
    
    // Write memory changes
    for (const auto& [arg_idx, changes] : exec.changes_by_arg) {
        for (const auto& [element_index, values] : changes) {
            struct {
                int arg_idx;
                size_t element_index;
                float pre_val;
                float post_val;
            } change_data = {
                arg_idx,
                element_index,
                values.first,
                values.second
            };
            trace_file_.write(reinterpret_cast<const char*>(&change_data), 
                            sizeof(change_data));
        }
    }
}

void Tracer::writeMemoryOperation(const MemoryOperation& op) {
    struct {
        MemoryOpType type;
        void* dst;
        const void* src;
        size_t size;
        int value;
        hipMemcpyKind kind;
        uint64_t execution_order;
        size_t pre_state_size;
        size_t post_state_size;
    } mem_op_data = {
        op.type,
        op.dst,
        op.src,
        op.size,
        op.value,
        op.kind,
        op.execution_order,
        op.pre_state ? op.pre_state->size : 0,
        op.post_state ? op.post_state->size : 0
    };
    
    writeEvent(2, &mem_op_data, sizeof(mem_op_data));
    
    // Write pre-state if exists
    if (op.pre_state && op.pre_state->data) {
        trace_file_.write(op.pre_state->data.get(), op.pre_state->size);
    }
    
    // Write post-state if exists
    if (op.post_state && op.post_state->data) {
        trace_file_.write(op.post_state->data.get(), op.post_state->size);
    }
}

std::string Tracer::getTraceFilePath() const {
    static int trace_id = 0;
    const char* home = getenv("HOME");
    if (!home) home = "/tmp";
    
    // Get binary name
    char self_path[PATH_MAX];
    ssize_t len = readlink("/proc/self/exe", self_path, sizeof(self_path) - 1);
    if (len == -1) {
        return std::string(home) + "/hipTracer/unknown-" + 
               std::to_string(trace_id++) + ".trace";
    }
    self_path[len] = '\0';
    
    std::string binary_name = std::filesystem::path(self_path).filename();
    
    // Skip system utilities
    static const std::vector<std::string> ignore_list = {
        "grep", "dash", "nm", "ld", "as", "objdump", "readelf", "addr2line"
    };
    
    for (const auto& ignored : ignore_list) {
        if (binary_name.find(ignored) != std::string::npos) {
            return ""; // Skip tracing
        }
    }
    
    // Create tracer directory
    std::string tracer_dir = std::string(home) + "/hipTracer";
    std::cout << "Creating tracer directory: " << tracer_dir << std::endl;
    mkdir(tracer_dir.c_str(), 0755);
    
    // Find next available trace ID
    std::string base_path = tracer_dir + "/" + binary_name + "-";
    while (access((base_path + std::to_string(trace_id) + ".trace").c_str(), F_OK) != -1) {
        trace_id++;
    }
    
    auto trace_path = base_path + std::to_string(trace_id++) + ".trace";
    std::cout << "Trace file path: " << trace_path << std::endl;
    return trace_path;
}

void Tracer::flush() {
    if (trace_file_.is_open()) {
        trace_file_.flush();
    }
}

// Add helper function implementations
std::string dim3ToString(const dim3& d) {
    std::stringstream ss;
    ss << "{" << d.x << "," << d.y << "," << d.z << "}";
    return ss.str();
}

const char* memcpyKindToString(hipMemcpyKind kind) {
    switch(kind) {
        case hipMemcpyHostToHost: return "hipMemcpyHostToHost";
        case hipMemcpyHostToDevice: return "hipMemcpyHostToDevice"; 
        case hipMemcpyDeviceToHost: return "hipMemcpyDeviceToHost";
        case hipMemcpyDeviceToDevice: return "hipMemcpyDeviceToDevice";
        case hipMemcpyDefault: return "hipMemcpyDefault";
        default: return "Unknown";
    }
}

std::string devicePropsToString(const hipDeviceProp_t* props) {
    if (!props) return "null";
    std::stringstream ss;
    ss << "{name=" << props->name << ", totalGlobalMem=" << props->totalGlobalMem << "}";
    return ss.str();
}

std::string demangle(const char* name) {
    int status;
    std::unique_ptr<char, void(*)(void*)> demangled(
        abi::__cxa_demangle(name, nullptr, nullptr, &status),
        std::free
    );
    return status == 0 ? demangled.get() : name;
}

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

std::string getKernelName(const void* function_address) {
    Dl_info info;
    if (dladdr(function_address, &info) && info.dli_sname) {
        return info.dli_sname;
    }
    std::stringstream ss;
    ss << "kernel_" << std::hex << (uintptr_t)function_address;
    return ss.str();
}

// Add Tracer method implementations
void Tracer::registerKernelArg(const std::string& kernel_name, size_t arg_index, 
                              bool is_vector, size_t size) {
    auto& info = kernel_registry_[kernel_name];
    if (info.args.size() <= arg_index) {
        info.args.resize(arg_index + 1);
    }
    info.args[arg_index].is_vector = is_vector;
    info.args[arg_index].size = size;
}

bool Tracer::isVectorType(const std::string& type_name) const {
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
    
    return std::any_of(vector_types.begin(), vector_types.end(),
                      [&](const auto& vtype) {
                          return type_name.find(vtype) != std::string::npos;
                      });
}

void Tracer::printKernelArgs(void** args, const std::string& kernelName, 
                            const void* function_address) {
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

Trace Tracer::loadTrace(const std::string& path) {
    Trace trace;
    std::ifstream file(path, std::ios::binary);
    
    if (!file) {
        throw std::runtime_error("Failed to open trace file: " + path);
    }
    
    // Read and verify header
    struct {
        uint32_t magic;
        uint32_t version;
    } header;
    
    file.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (header.magic != TRACE_MAGIC) {
        throw std::runtime_error("Invalid trace file format: incorrect magic number");
    }
    if (header.version != TRACE_VERSION) {
        throw std::runtime_error("Unsupported trace file version");
    }
    
    // Read events until end of file
    while (file.good() && !file.eof()) {
        struct {
            uint32_t type;
            uint64_t timestamp;
            uint32_t size;
        } event_header;
        
        if (!file.read(reinterpret_cast<char*>(&event_header), sizeof(event_header))) {
            break;  // End of file or error
        }
        
        switch (event_header.type) {
            case 1: // Kernel execution
                trace.kernel_executions.push_back(readKernelExecution(file));
                break;
            case 2: // Memory operation
                trace.memory_operations.push_back(readMemoryOperation(file));
                break;
            default:
                throw std::runtime_error("Unknown event type in trace file");
        }
    }
    
    return trace;
}

KernelExecution Tracer::readKernelExecution(std::ifstream& file) {
    KernelExecution exec;
    
    struct {
        void* function_address;
        uint32_t name_length;
        dim3 grid_dim;
        dim3 block_dim;
        size_t shared_mem;
        hipStream_t stream;
        uint64_t execution_order;
        uint32_t num_args;
        uint32_t num_changes;
    } kernel_data;
    
    file.read(reinterpret_cast<char*>(&kernel_data), sizeof(kernel_data));
    
    // Read kernel name
    std::vector<char> name_buffer(kernel_data.name_length + 1);
    file.read(name_buffer.data(), kernel_data.name_length);
    name_buffer[kernel_data.name_length] = '\0';
    
    exec.function_address = kernel_data.function_address;
    exec.kernel_name = name_buffer.data();
    exec.grid_dim = kernel_data.grid_dim;
    exec.block_dim = kernel_data.block_dim;
    exec.shared_mem = kernel_data.shared_mem;
    exec.stream = kernel_data.stream;
    exec.execution_order = kernel_data.execution_order;
    
    // Read argument pointers
    for (uint32_t i = 0; i < kernel_data.num_args; i++) {
        void* arg_ptr;
        file.read(reinterpret_cast<char*>(&arg_ptr), sizeof(void*));
        exec.arg_ptrs.push_back(arg_ptr);
    }
    
    // Read memory changes
    for (uint32_t i = 0; i < kernel_data.num_changes; i++) {
        struct {
            int arg_idx;
            size_t element_index;
            float pre_val;
            float post_val;
        } change_data;
        
        file.read(reinterpret_cast<char*>(&change_data), sizeof(change_data));
        auto& changes = exec.changes_by_arg[change_data.arg_idx];
        changes.push_back(std::make_pair(
            change_data.element_index,
            std::make_pair(change_data.pre_val, change_data.post_val)
        ));
    }
    
    return exec;
}

MemoryOperation Tracer::readMemoryOperation(std::ifstream& file) {
    MemoryOperation op;
    
    struct {
        MemoryOpType type;
        void* dst;
        const void* src;
        size_t size;
        int value;
        hipMemcpyKind kind;
        uint64_t execution_order;
        size_t pre_state_size;
        size_t post_state_size;
    } mem_op_data;
    
    file.read(reinterpret_cast<char*>(&mem_op_data), sizeof(mem_op_data));
    
    op.type = mem_op_data.type;
    op.dst = mem_op_data.dst;
    op.src = mem_op_data.src;
    op.size = mem_op_data.size;
    op.value = mem_op_data.value;
    op.kind = mem_op_data.kind;
    op.execution_order = mem_op_data.execution_order;
    
    // Read pre-state if exists
    if (mem_op_data.pre_state_size > 0) {
        op.pre_state = std::make_shared<MemoryState>(mem_op_data.pre_state_size);
        file.read(op.pre_state->data.get(), mem_op_data.pre_state_size);
    }
    
    // Read post-state if exists
    if (mem_op_data.post_state_size > 0) {
        op.post_state = std::make_shared<MemoryState>(mem_op_data.post_state_size);
        file.read(op.post_state->data.get(), mem_op_data.post_state_size);
    }
    
    return op;
}

} // namespace hip_intercept
