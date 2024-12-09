#include "Comparator.hh"
#include <iostream>
#include <iomanip>
#include <cmath>

namespace hip_intercept {

Comparator::Comparator(float epsilon) : epsilon_(epsilon) {}

bool Comparator::compareFloats(float a, float b) const {
    if (std::isnan(a) && std::isnan(b)) return true;
    if (std::isinf(a) && std::isinf(b)) return a == b;
    return std::abs(a - b) <= epsilon_;
}

ValueDifference Comparator::compareMemoryChanges(
    const std::vector<MemoryChange>& changes1,
    const std::vector<MemoryChange>& changes2) {
    
    ValueDifference diff;
    diff.matches = true;
    
    std::set<size_t> indices1, indices2;
    for (const auto& change : changes1) indices1.insert(change.element_index);
    for (const auto& change : changes2) indices2.insert(change.element_index);
    
    // Find missing indices
    std::set_difference(indices1.begin(), indices1.end(),
                       indices2.begin(), indices2.end(),
                       std::back_inserter(diff.missing_indices));
    
    // Compare values for common indices
    for (const auto& change1 : changes1) {
        auto it = std::find_if(changes2.begin(), changes2.end(),
            [&](const MemoryChange& c) { return c.element_index == change1.element_index; });
        
        if (it != changes2.end()) {
            if (!compareFloats(change1.pre_value, it->pre_value)) {
                diff.matches = false;
                diff.pre_value_mismatches.push_back({
                    change1.element_index,
                    change1.pre_value,
                    it->pre_value
                });
            }
            if (!compareFloats(change1.post_value, it->post_value)) {
                diff.matches = false;
                diff.post_value_mismatches.push_back({
                    change1.element_index,
                    change1.post_value,
                    it->post_value
                });
            }
        }
    }
    
    return diff;
}

KernelComparisonResult Comparator::compareKernelExecutions(
    const KernelExecution& exec1,
    const KernelExecution& exec2) {
    
    KernelComparisonResult result;
    result.matches = true;
    result.kernel_name = exec1.kernel_name;
    
    // Compare basic kernel properties
    if (exec1.kernel_name != exec2.kernel_name) {
        result.matches = false;
        result.differences.push_back("Kernel names differ: " + 
            exec1.kernel_name + " vs " + exec2.kernel_name);
    }
    
    if (exec1.grid_dim.x != exec2.grid_dim.x ||
        exec1.grid_dim.y != exec2.grid_dim.y ||
        exec1.grid_dim.z != exec2.grid_dim.z) {
        result.matches = false;
        result.differences.push_back("Grid dimensions differ");
    }
    
    if (exec1.block_dim.x != exec2.block_dim.x ||
        exec1.block_dim.y != exec2.block_dim.y ||
        exec1.block_dim.z != exec2.block_dim.z) {
        result.matches = false;
        result.differences.push_back("Block dimensions differ");
    }
    
    if (exec1.shared_mem != exec2.shared_mem) {
        result.matches = false;
        result.differences.push_back("Shared memory size differs");
    }
    
    // Compare memory changes for each argument
    for (const auto& [arg_idx, changes1] : exec1.changes_by_arg) {
        auto it = exec2.changes_by_arg.find(arg_idx);
        if (it == exec2.changes_by_arg.end()) {
            result.matches = false;
            result.differences.push_back("Missing changes for argument " + 
                std::to_string(arg_idx) + " in second trace");
            continue;
        }
        
        std::vector<MemoryChange> changes1_vec, changes2_vec;
        for (const auto& [idx, vals] : changes1) {
            changes1_vec.push_back({idx, vals.first, vals.second});
        }
        for (const auto& [idx, vals] : it->second) {
            changes2_vec.push_back({idx, vals.first, vals.second});
        }
        
        auto diff = compareMemoryChanges(changes1_vec, changes2_vec);
        if (!diff.matches) {
            result.matches = false;
            result.value_differences[arg_idx] = diff;
        }
    }
    
    return result;
}

ComparisonResult Comparator::compare(const Trace& trace1, const Trace& trace2) {
    ComparisonResult result;
    result.traces_match = true;
    result.first_divergence_point = SIZE_MAX;

    // Compare kernel executions
    size_t min_kernels = std::min(trace1.kernel_executions.size(), 
                                 trace2.kernel_executions.size());
    
    for (size_t i = 0; i < min_kernels; ++i) {
        auto kernel_result = compareKernelExecutions(
            trace1.kernel_executions[i],
            trace2.kernel_executions[i]
        );
        
        result.kernel_results.push_back(kernel_result);
        
        if (!kernel_result.matches) {
            result.traces_match = false;
            if (result.first_divergence_point == SIZE_MAX) {
                result.first_divergence_point = i;
            }
        }
    }

    // Check for different number of kernel executions
    if (trace1.kernel_executions.size() != trace2.kernel_executions.size()) {
        result.traces_match = false;
        result.error_message += "Different number of kernel executions: " +
            std::to_string(trace1.kernel_executions.size()) + " vs " +
            std::to_string(trace2.kernel_executions.size()) + "\n";
    }

    // Compare memory operations
    if (trace1.memory_operations.size() != trace2.memory_operations.size()) {
        result.traces_match = false;
        result.error_message += "Different number of memory operations: " +
            std::to_string(trace1.memory_operations.size()) + " vs " +
            std::to_string(trace2.memory_operations.size()) + "\n";
    }

    // Compare individual memory operations
    size_t min_mem_ops = std::min(trace1.memory_operations.size(),
                                 trace2.memory_operations.size());
    
    for (size_t i = 0; i < min_mem_ops; ++i) {
        const auto& op1 = trace1.memory_operations[i];
        const auto& op2 = trace2.memory_operations[i];
        
        if (op1.type != op2.type || op1.size != op2.size ||
            op1.kind != op2.kind) {
            result.traces_match = false;
            result.error_message += "Memory operation " + std::to_string(i) +
                " differs in type, size, or kind\n";
        }
        
        // Compare memory states if they exist
        if ((op1.pre_state && !op2.pre_state) || (!op1.pre_state && op2.pre_state) ||
            (op1.post_state && !op2.post_state) || (!op1.post_state && op2.post_state)) {
            result.traces_match = false;
            result.error_message += "Memory operation " + std::to_string(i) +
                " differs in state availability\n";
        }
        
        if (op1.pre_state && op2.pre_state &&
            (op1.pre_state->size != op2.pre_state->size ||
             memcmp(op1.pre_state->data.get(), op2.pre_state->data.get(),
                    op1.pre_state->size) != 0)) {
            result.traces_match = false;
            result.error_message += "Memory operation " + std::to_string(i) +
                " differs in pre-state\n";
        }
        
        if (op1.post_state && op2.post_state &&
            (op1.post_state->size != op2.post_state->size ||
             memcmp(op1.post_state->data.get(), op2.post_state->data.get(),
                    op1.post_state->size) != 0)) {
            result.traces_match = false;
            result.error_message += "Memory operation " + std::to_string(i) +
                " differs in post-state\n";
        }
    }

    return result;
}

void Comparator::printComparisonResult(const ComparisonResult& result) {
    std::cout << "\nComparison Results:\n"
              << "==================\n";
    
    if (result.traces_match) {
        std::cout << "Traces match exactly!\n";
        return;
    }
    
    std::cout << "Traces differ.\n";
    if (!result.error_message.empty()) {
        std::cout << "Error: " << result.error_message << "\n";
    }
    
    if (result.first_divergence_point != SIZE_MAX) {
        std::cout << "First divergence at kernel execution "
                  << result.first_divergence_point << "\n\n";
    }
    
    for (size_t i = 0; i < result.kernel_results.size(); ++i) {
        const auto& kr = result.kernel_results[i];
        if (!kr.matches) {
            std::cout << "Kernel " << i << " (" << kr.kernel_name << ") differences:\n";
            
            for (const auto& diff : kr.differences) {
                std::cout << "  - " << diff << "\n";
            }
            
            for (const auto& [arg_idx, diff] : kr.value_differences) {
                std::cout << "  Argument " << arg_idx << ":\n";
                
                if (!diff.missing_indices.empty()) {
                    std::cout << "    Missing indices: ";
                    for (auto idx : diff.missing_indices) {
                        std::cout << idx << " ";
                    }
                    std::cout << "\n";
                }
                
                if (!diff.pre_value_mismatches.empty()) {
                    std::cout << "    Pre-execution mismatches:\n";
                    for (const auto& m : diff.pre_value_mismatches) {
                        std::cout << "      Index " << m.index << ": "
                                 << m.value1 << " vs " << m.value2 << "\n";
                    }
                }
                
                if (!diff.post_value_mismatches.empty()) {
                    std::cout << "    Post-execution mismatches:\n";
                    for (const auto& m : diff.post_value_mismatches) {
                        std::cout << "      Index " << m.index << ": "
                                 << m.value1 << " vs " << m.value2 << "\n";
                    }
                }
            }
            std::cout << "\n";
        }
    }
}

ComparisonResult Comparator::compare(const std::string& trace_path1, const std::string& trace_path2) {
    try {
        Trace trace1 = Tracer::loadTrace(trace_path1);
        Trace trace2 = Tracer::loadTrace(trace_path2);
        return compare(trace1, trace2);
    } catch (const std::exception& e) {
        ComparisonResult result;
        result.traces_match = false;
        result.error_message = std::string("Failed to load traces: ") + e.what();
        return result;
    }
}

} // namespace hip_intercept


int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <trace1> <trace2>\n";
        return 1;
    }

    hip_intercept::Comparator comparator;
    hip_intercept::ComparisonResult result = comparator.compare(argv[1], argv[2]);
    comparator.printComparisonResult(result);
    return 0;
}