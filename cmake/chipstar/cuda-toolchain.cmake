get_filename_component(CHIPSTAR_PREFIX "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)

set(CMAKE_CUDA_COMPILER "${CHIPSTAR_PREFIX}/bin/cucc" CACHE FILEPATH "chipStar CUDA compiler driver")
set(CUDAToolkit_ROOT "${CHIPSTAR_PREFIX}" CACHE PATH "chipStar CUDA toolkit root")
