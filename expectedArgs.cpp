
#include <hip/hip_runtime.h>

template <typename... FArgs>
std::tuple<FArgs...> getExpectedArgs(void(*)(FArgs...)) {};

template <typename F, typename... Args>
void validateArguments(F f, Args&&... args) {
    using expectedArgsTuple = decltype(getExpectedArgs(f));
    using providedArgsTuple = std::tuple<Args...>;

    static_assert(std::is_same<expectedArgsTuple, providedArgsTuple>::value,
                  "Kernel arguments types must match exactly!");
}

// General launchKernel function
template <typename... Typenames, typename Kernel, typename Dim, typename... Args>
void launchKernel(Kernel kernel, Dim numBlocks, Dim numThreads, std::uint32_t memPerBlock, hipStream_t stream, Args&&... args) {
    // Define a stateless, capture-free lambda that matches the kernel's signature.
    auto kernelWrapperLambda = [] (Args... args) {
        // This lambda is intentionally left empty as it's used solely for type validation.
    };

    // Convert the lambda to a function pointer.
    void (*kernelWrapper)(Args...) = kernelWrapperLambda;

    // Use the wrapper function pointer to validate arguments.
    validateArguments(kernelWrapper, std::forward<Args>(args)...);

    // Launch the kernel directly with the provided arguments.
    kernel<<<numBlocks, numThreads, memPerBlock, stream>>>(std::forward<Args>(args)...);

}

template <typename T> void __global__ vectorADD(const T* A_d, const T* B_d, T* C_d, size_t NELEM) {}

int main() {
    int LEN = 1;
    dim3 dimGrid(LEN / 512, 1, 1);
    dim3 dimBlock(512, 1, 1);
    float *A_d, *B_d, *C_d;


    
    launchKernel<float>(vectorADD<float>, dimGrid, dimBlock,
            0, 0, static_cast<const float*>(A_d),
            static_cast<float*>(B_d), C_d, static_cast<size_t>(LEN));
}         