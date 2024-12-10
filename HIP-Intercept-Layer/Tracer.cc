#include "Tracer.hh"

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
