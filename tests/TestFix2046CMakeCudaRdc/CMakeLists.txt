cmake_minimum_required(VERSION 3.28)
project(t LANGUAGES CXX CUDA)
find_package(CUDAToolkit REQUIRED)
# dev/main.cu shares main.cu's basename on purpose.
add_executable(t main.cu dev/main.cu)
set_target_properties(t PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(t CUDA::cudart)
