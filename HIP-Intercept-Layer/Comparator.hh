#pragma once

#include "Tracer.hh"
#include <vector>
#include <string>
#include <unordered_map>
#include <set>

namespace hip_intercept {

// Structure to represent a memory change
struct MemoryChange {
    size_t element_index;
    float pre_value;
    float post_value;
};

// Structure to hold value mismatches
struct ValueMismatch {
    size_t index;
    float value1;
    float value2;
};

// Structure to hold differences between memory changes
struct ValueDifference {
    bool matches;
    std::vector<size_t> missing_indices;
    std::vector<ValueMismatch> pre_value_mismatches;
    std::vector<ValueMismatch> post_value_mismatches;
};

// Structure to hold kernel comparison results
struct KernelComparisonResult {
    bool matches;
    std::string kernel_name;
    std::vector<std::string> differences;
    std::unordered_map<int, ValueDifference> value_differences;
};

// Structure to hold overall comparison results
struct ComparisonResult {
    bool traces_match;
    size_t first_divergence_point;
    std::string error_message;
    std::vector<KernelComparisonResult> kernel_results;
};

class Comparator {
public:
    explicit Comparator(float epsilon = 1e-6);

    // Main comparison function
    ComparisonResult compare(const Trace& trace1, const Trace& trace2);

    // Print detailed comparison results
    void printComparisonResult(const ComparisonResult& result);

    // Add this new method declaration
    ComparisonResult compare(const std::string& trace_path1, const std::string& trace_path2);

private:
    float epsilon_;

    // Helper comparison functions
    KernelComparisonResult compareKernelExecutions(
        const KernelExecution& exec1, 
        const KernelExecution& exec2);

    ValueDifference compareMemoryChanges(
        const std::vector<MemoryChange>& changes1,
        const std::vector<MemoryChange>& changes2);

    bool compareFloats(float a, float b) const;
};

} // namespace hip_intercept 